"""
DeepSets architecture for customer loan portfolios.

Architecture:
  phi  : per-loan MLP → transforms each loan to latent space independently
  pool : masked mean + masked max → concatenated (2 × hidden_dim)
         handles variable-length portfolios (1 to MAX_LOANS loans)
  rho  : customer-level MLP → produces a dense embedding
  head : linear classifier (used during training; frozen for XGBoost)

Why DeepSets over Transformer when MAX_LOANS ≤ 7:
  - Self-attention on 1-2 tokens degenerates to a weighted average
  - phi+pool is provably permutation-invariant (correct for unordered loan sets)
  - Far fewer parameters, faster on CPU
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class DeepSets(nn.Module):
    """
    DeepSets for customer loan portfolio classification.

    Args:
        n_features:    number of input loan features
        hidden_dim:    width of the phi (per-loan) MLP layers
        embedding_dim: dimension of the customer-level embedding fed to XGBoost
        dropout:       dropout probability in both phi and rho
        num_classes:   number of output classes (default 3)
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.15,
        num_classes: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # ── phi: per-loan feature encoder ───────────────────────────────────
        # Applied independently to each loan (like a shared MLP across the set)
        self.phi = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── rho: customer-level aggregation MLP ─────────────────────────────
        # Input: concat(masked_mean, masked_max) → 2 * hidden_dim
        self.rho = nn.Sequential(
            nn.Linear(2 * hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Classification head (training only; bypassed for XGBoost) ───────
        self.head = nn.Linear(embedding_dim, num_classes)

        self._init_weights()
        logger.info(
            f"Initialized DeepSets with {self._count_params():,} parameters "
            f"(hidden={hidden_dim}, embed={embedding_dim})."
        )

    # ── Weight initialisation ────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Core forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features:     (B, MAX_LOANS, n_features)  float32
            padding_mask: (B, MAX_LOANS)               bool — True where padded

        Returns:
            logits:     (B, num_classes)
            embeddings: (B, embedding_dim)  ← used by XGBoost
        """
        # 1. Per-loan encoding via phi  →  (B, MAX_LOANS, hidden_dim)
        encoded = self.phi(features)

        # 2. Masking: valid positions = True, padded = False
        valid = (~padding_mask).float().unsqueeze(-1)  # (B, MAX_LOANS, 1)

        # 3. Masked mean pooling
        n_valid = valid.sum(dim=1).clamp(min=1e-9)         # (B, 1)
        mean_pool = (encoded * valid).sum(dim=1) / n_valid   # (B, hidden_dim)

        # 4. Masked max pooling
        #    Fill padded positions with -inf so they never win the max
        masked_encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        max_pool, _ = masked_encoded.max(dim=1)              # (B, hidden_dim)
        # Guard against all-padded rows (edge case: single valid loan column)
        max_pool = torch.nan_to_num(max_pool, nan=0.0, neginf=0.0)

        # 5. Concatenate pooling signals  →  (B, 2 * hidden_dim)
        pooled = torch.cat([mean_pool, max_pool], dim=1)

        # 6. rho: customer-level embedding  →  (B, embedding_dim)
        embedding = self.rho(pooled)

        # 7. Classification head
        logits = self.head(embedding)

        return logits, embedding

    def extract_embeddings(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return only the customer embedding (B, embedding_dim) for XGBoost."""
        _, embeddings = self.forward(features, padding_mask)
        return embeddings
