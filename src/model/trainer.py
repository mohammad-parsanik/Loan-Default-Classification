"""
Trainer for the DeepSets model.

Fixes vs. original:
  - scheduler.step(epoch + 1) — was step() which reset LR every epoch
  - Per-epoch checkpoint saves to <checkpoint_dir>/epoch_{N:03d}.pt
  - Returns loss/metric history for loss-curve plotting
  - Logs wall-clock time per epoch
"""

import copy
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.utils as nn_utils
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class TransformerTrainer:
    """
    Generic trainer — works with any model that has the same forward signature as
    DeepSets (features, padding_mask) → (logits, embedding).
    """

    def __init__(self, model, criterion, config, device: str = "cpu",
                 checkpoint_dir: Optional[Path] = None):
        self.model      = model.to(device)
        self.criterion  = criterion.to(device)
        self.device     = device
        self.config     = config
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        # CosineAnnealingWarmRestarts requires epoch index passed to step()
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

    # ── Single epoch ──────────────────────────────────────────────────────────

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="  train", leave=False):
            features     = batch["features"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            labels       = batch["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()

            logits, _ = self.model(features, padding_mask)
            loss = self.criterion(logits, labels)

            loss.backward()
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * features.size(0)

        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                features     = batch["features"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                labels       = batch["label"].to(self.device)

                logits, _ = self.model(features, padding_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * features.size(0)

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss  = total_loss / len(dataloader.dataset)
        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs  = torch.cat(all_probs).numpy()

        return avg_loss, compute_metrics(all_labels, all_preds, all_probs)

    # ── Full training loop ────────────────────────────────────────────────────

    def train(self, train_dl, val_dl) -> tuple[float, dict]:
        """
        Train for up to config.EPOCHS epochs with early stopping on val Macro F1.
        If validation optimization is disabled (val_dl is empty), trains for FIXED_EPOCHS.

        Returns:
            (best_val_f1, history)
            history: dict with lists 'train_loss', 'val_loss', 'val_f1'
        """
        best_val_f1    = -1.0
        best_state     = None
        patience_ctr   = 0
        history        = {"train_loss": [], "val_loss": [], "val_f1": []}

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        use_val = val_dl is not None and len(val_dl) > 0
        n_epochs = self.config.EPOCHS if use_val else getattr(self.config, "FIXED_EPOCHS", 15)

        logger.info(
            f"Starting training for up to {n_epochs} epochs "
            f"on {self.device} (patience={self.config.PATIENCE if use_val else 'N/A'})…"
        )

        for epoch in range(n_epochs):
            t0 = time.perf_counter()

            train_loss = self.train_epoch(train_dl)
            
            if use_val:
                val_loss, val_metrics = self.evaluate(val_dl)
                val_f1 = val_metrics["macro_f1"]
            else:
                val_loss, val_f1 = 0.0, 0.0

            # Fix: pass epoch index so cosine schedule advances correctly
            self.scheduler.step(epoch + 1)

            elapsed = time.perf_counter() - t0

            if use_val:
                logger.info(
                    f"Epoch {epoch+1:03d}/{n_epochs} | "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"val_f1={val_f1:.4f}  lr={self.scheduler.get_last_lr()[0]:.2e}  "
                    f"({elapsed:.1f}s)"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1:03d}/{n_epochs} | "
                    f"train_loss={train_loss:.4f}  "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e}  "
                    f"({elapsed:.1f}s)"
                )

            history["train_loss"].append(train_loss)
            if use_val:
                history["val_loss"].append(val_loss)
                history["val_f1"].append(val_f1)

            # Per-epoch checkpoint
            if self.checkpoint_dir:
                ckpt_path = self.checkpoint_dir / f"epoch_{epoch+1:03d}.pt"
                torch.save(
                    {
                        "epoch":       epoch + 1,
                        "model_state": self.model.state_dict(),
                        "optimizer":   self.optimizer.state_dict(),
                        "scheduler":   self.scheduler.state_dict(),
                        "val_f1":      val_f1,
                        "history":     history,
                    },
                    ckpt_path,
                )

            if use_val:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state  = copy.deepcopy(self.model.state_dict())
                    patience_ctr = 0
                    logger.info(f"  ↑ New best model (val_f1={val_f1:.4f})")
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.config.PATIENCE:
                        logger.info(f"Early stopping at epoch {epoch+1}.")
                        break
            else:
                best_state = copy.deepcopy(self.model.state_dict())
                best_val_f1 = 0.0

        if best_state:
            self.model.load_state_dict(best_state)

        return best_val_f1, history
