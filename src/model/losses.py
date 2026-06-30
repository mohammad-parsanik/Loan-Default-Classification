import torch
import torch.nn as nn
import torch.nn.functional as F


class CostSensitiveFocalLoss(nn.Module):
    """
    Combines Focal Loss (gamma=2.0) with a cost matrix to heavily penalise
    missing high-risk customers (Cat 2).

    Cost Matrix (True × Predicted):
                   Pred 0   Pred 1   Pred 2
        True 0  [  0.0,    0.5,    1.0  ]
        True 1  [  1.5,    0.0,    0.5  ]
        True 2  [  4.0,    2.0,    0.0  ]

    Using register_buffer so .to(device) moves the matrix automatically.
    """

    def __init__(self, gamma: float = 2.0, num_classes: int = 3):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes

        cost_matrix = torch.tensor(
            [
                [0.0, 0.5, 1.0],  # True 0
                [1.5, 0.0, 0.5],  # True 1
                [4.0, 2.0, 0.0],  # True 2
            ],
            dtype=torch.float32,
        )
        # Registered buffers are moved automatically with .to(device)
        self.register_buffer("cost_matrix", cost_matrix)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, num_classes)  raw unnormalised scores
            targets: (B,)              class indices {0, 1, 2}
        Returns:
            scalar loss
        """
        probs = F.softmax(logits, dim=1)          # (B, num_classes)
        log_probs = F.log_softmax(logits, dim=1)  # (B, num_classes)

        # Cost row for each sample: (B, num_classes)
        costs = self.cost_matrix[targets]

        # One-hot: (B, num_classes)
        targets_oh = F.one_hot(targets, num_classes=self.num_classes).float()

        # Focal weight on the true class: (B,)
        pt = (probs * targets_oh).sum(dim=1)
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        # Standard focal cross-entropy on the true class: (B,)
        ce_loss = -(targets_oh * log_probs).sum(dim=1)
        focal_component = focal_weight * ce_loss

        # Expected cost: penalises probability mass on expensive wrong classes: (B,)
        expected_cost = (probs * costs).sum(dim=1)

        return (focal_component + expected_cost).mean()
