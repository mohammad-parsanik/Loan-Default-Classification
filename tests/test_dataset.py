import pytest
import numpy as np
import torch
from src.data.dataset import CustomerPortfolioDataset

def test_dataset_padding_and_masking():
    max_loans = 5
    
    instances = [
        {
            'features': np.ones((2, 10)),
            'label': 1
        },
        {
            'features': np.ones((6, 10)),  # Will be truncated
            'label': 2
        }
    ]
    
    dataset = CustomerPortfolioDataset(instances, max_loans=max_loans)
    
    # First instance (padding)
    item1 = dataset[0]
    assert item1['features'].shape == (max_loans, 10)
    assert item1['padding_mask'].sum().item() == 3  # 5 - 2 = 3 padded elements
    assert torch.all(item1['padding_mask'][:2] == False)
    assert torch.all(item1['padding_mask'][2:] == True)
    assert item1['label'].item() == 1
    
    # Second instance (truncation)
    item2 = dataset[1]
    assert item2['features'].shape == (max_loans, 10)
    assert item2['padding_mask'].sum().item() == 0  # No padding
    assert torch.all(item2['padding_mask'] == False)
    assert item2['label'].item() == 2
