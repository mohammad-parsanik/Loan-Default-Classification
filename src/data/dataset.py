"""
PyTorch Dataset and DataLoader factory for customer loan portfolios.

Notes:
  - With MAX_LOANS ≤ 7 and ~64 features, each instance is tiny (7 × 64 × 4 = 1.8 KB).
  - Padding is applied once in __getitem__ — no pre-allocation needed.
  - num_workers defaults to 2; set to 0 if you observe IPC overhead with small payloads.
  - prefetch_factor=2 is set explicitly (requires num_workers > 0).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader


class CustomerPortfolioDataset(Dataset):
    """
    Returns per customer-snapshot:
      features:      (MAX_LOANS, N_features)  float32 — padded with zeros
      padding_mask:  (MAX_LOANS,)             bool   — True = padded slot
      label:         ()                       int64  — {0, 1, 2}
    """

    def __init__(self, instances: list[dict], max_loans: int):
        self.instances = instances
        self.max_loans = max_loans

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:
        inst     = self.instances[idx]
        features = inst["features"]          # (n_loans, n_features)  already float32
        n_loans, n_features = features.shape

        # Clamp: should already be truncated by data_loader, but guard here
        seq_len = min(n_loans, self.max_loans)

        padded = np.zeros((self.max_loans, n_features), dtype=np.float32)
        padded[:seq_len] = features[:seq_len]

        mask = np.ones(self.max_loans, dtype=bool)
        mask[:seq_len] = False               # False = valid, True = padded

        return {
            "features":     torch.from_numpy(padded),
            "padding_mask": torch.from_numpy(mask),
            "label":        torch.tensor(inst["label"], dtype=torch.long),
        }


def create_dataloaders(
    train_inst: list[dict],
    val_inst: list[dict],
    test_inst: list[dict],
    max_loans: int,
    batch_size: int,
    num_workers: int = 2,
) -> tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
    """
    Returns (train_dl, val_dl, test_dl).

    num_workers=2 is a good default for this workload (tiny per-sample payloads).
    Set num_workers=0 if you see slower throughput due to IPC overhead.
    """
    def _make_dl(instances, shuffle):
        ds = CustomerPortfolioDataset(instances, max_loans)
        use_workers = num_workers if len(instances) > 0 else 0
        kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=use_workers,
            persistent_workers=(use_workers > 0),
            pin_memory=False,            # CPU-only training; pin_memory wastes RAM
        )
        if use_workers > 0:
            kwargs["prefetch_factor"] = 2
        return TorchDataLoader(ds, **kwargs)

    train_dl = _make_dl(train_inst, shuffle=True)
    val_dl   = _make_dl(val_inst,   shuffle=False)
    test_dl  = _make_dl(test_inst,  shuffle=False)

    return train_dl, val_dl, test_dl
