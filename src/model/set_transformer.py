import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class SetTransformer(nn.Module):
    """
    Set-Transformer architecture for customer loan portfolios.
    No positional encoding since portfolios are unordered sets.
    """
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4, 
                 n_layers: int = 2, d_feedforward: int = 256, dropout: int = 0.15,
                 num_classes: int = 3):
        super().__init__()
        self.d_model = d_model
        
        # 1. Input Projection
        self.input_projection = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # 2. Transformer Encoder Layers (Pre-LN is norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Classification Head (used during training to learn representations)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self._init_weights()
        logger.info(f"Initialized Set-Transformer with {self._count_params():,} parameters.")
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, features, padding_mask):
        """
        Args:
            features: (B, MAX_LOANS, n_features)
            padding_mask: (B, MAX_LOANS) bool tensor, True where padded
        Returns:
            logits: (B, num_classes)
            embeddings: (B, d_model)
        """
        # (B, MAX_LOANS, d_model)
        x = self.input_projection(features)
        
        # Transformer processing
        # src_key_padding_mask requires True for padded elements
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Masked Mean Pooling
        # Invert mask: 1 for valid, 0 for padded
        valid_mask = (~padding_mask).float().unsqueeze(-1)  # (B, MAX_LOANS, 1)
        
        # Sum valid tokens / number of valid tokens
        sum_embeddings = (x * valid_mask).sum(dim=1)  # (B, d_model)
        num_valid = valid_mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        
        customer_embedding = sum_embeddings / num_valid  # (B, d_model)
        
        # Classification
        logits = self.head(customer_embedding)
        
        return logits, customer_embedding
        
    def extract_embeddings(self, features, padding_mask):
        """Helper method to just get the customer embedding for XGBoost."""
        _, embeddings = self.forward(features, padding_mask)
        return embeddings
