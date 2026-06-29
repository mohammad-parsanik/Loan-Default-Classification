import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomerPortfolioDataset(Dataset):
    """
    PyTorch Dataset for customer loan portfolios.
    Handles padding up to MAX_LOANS and creating the mask.
    """
    def __init__(self, instances, max_loans):
        self.instances = instances
        self.max_loans = max_loans
        
    def __len__(self):
        return len(self.instances)
        
    def __getitem__(self, idx):
        instance = self.instances[idx]
        features = instance['features']  # (n_loans, n_features)
        label = instance['label']
        
        n_loans, n_features = features.shape
        
        # Determine actual length (already truncated in dataloader if necessary, but double check)
        seq_len = min(n_loans, self.max_loans)
        
        # Initialize padded array
        padded_features = np.zeros((self.max_loans, n_features), dtype=np.float32)
        padded_features[:seq_len] = features[:seq_len]
        
        # Create mask: True means padded (ignore), False means valid data
        padding_mask = np.ones(self.max_loans, dtype=bool)
        padding_mask[:seq_len] = False
        
        return {
            'features': torch.tensor(padded_features),
            'padding_mask': torch.tensor(padding_mask),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_dataloaders(train_inst, val_inst, test_inst, max_loans, batch_size, num_workers=4):
    """Create PyTorch DataLoaders for train, val, test."""
    train_ds = CustomerPortfolioDataset(train_inst, max_loans)
    val_ds = CustomerPortfolioDataset(val_inst, max_loans)
    test_ds = CustomerPortfolioDataset(test_inst, max_loans)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, persistent_workers=True if num_workers > 0 else False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                        num_workers=num_workers, persistent_workers=True if num_workers > 0 else False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                         num_workers=num_workers, persistent_workers=True if num_workers > 0 else False)
                         
    return train_dl, val_dl, test_dl
