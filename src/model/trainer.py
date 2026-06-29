import torch
import torch.nn.utils as nn_utils
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import logging
import copy
from tqdm import tqdm
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

class TransformerTrainer:
    """Trainer for the Set-Transformer using PyTorch."""
    def __init__(self, model, criterion, config, device="cpu"):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.device = device
        self.config = config
        
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=1e-6
        )
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training", leave=False):
            features = batch['features'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(features, padding_mask)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item() * features.size(0)
            
        return total_loss / len(dataloader.dataset)
        
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                padding_mask = batch['padding_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, _ = self.model(features, padding_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item() * features.size(0)
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                
        avg_loss = total_loss / len(dataloader.dataset)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        return avg_loss, metrics

    def train(self, train_dl, val_dl):
        best_val_f1 = -1.0
        best_model_state = None
        patience_counter = 0
        
        logger.info(f"Starting training for {self.config.EPOCHS} epochs on {self.device}...")
        
        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(train_dl)
            val_loss, val_metrics = self.evaluate(val_dl)
            
            self.scheduler.step()
            
            val_f1 = val_metrics['macro_f1']
            logger.info(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                logger.info(f"  -> New best model! (Macro F1: {val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        return best_val_f1
