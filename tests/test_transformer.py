import pytest
import torch
from src.model.set_transformer import SetTransformer

def test_set_transformer_shapes():
    batch_size = 4
    max_loans = 10
    n_features = 64
    d_model = 32
    num_classes = 3
    
    model = SetTransformer(n_features=n_features, d_model=d_model, num_classes=num_classes)
    
    # Dummy input
    features = torch.randn(batch_size, max_loans, n_features)
    
    # Mask where last 2 in each sequence are padded
    padding_mask = torch.zeros(batch_size, max_loans, dtype=torch.bool)
    padding_mask[:, 8:] = True
    
    logits, embeddings = model(features, padding_mask)
    
    assert logits.shape == (batch_size, num_classes)
    assert embeddings.shape == (batch_size, d_model)
    
def test_masked_mean_pooling():
    # To test masked mean, we can pass ones and see if the mean is 1
    # despite having zeros in the padded positions
    model = SetTransformer(n_features=16, d_model=16, n_layers=1)
    
    features = torch.ones(2, 5, 16)
    mask = torch.tensor([
        [False, False, False, True, True],
        [False, True, True, True, True]
    ])
    
    # Bypassing the transformer layers just to test the pooling logic
    # In reality, the pooling logic is inside forward(). We can test if gradients flow 
    # and if padded elements don't affect the output.
    logits1, emb1 = model(features, mask)
    
    # Change padded elements
    features[0, 3:] = 999.0
    features[1, 1:] = 999.0
    
    logits2, emb2 = model(features, mask)
    
    assert torch.allclose(emb1, emb2)
    assert torch.allclose(logits1, logits2)
