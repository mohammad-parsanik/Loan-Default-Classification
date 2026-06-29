import torch
import torch.nn as nn
import torch.nn.functional as F

class CostSensitiveFocalLoss(nn.Module):
    """
    Combines Focal Loss (gamma=2.0) with a predefined cost matrix to 
    heavily penalize missing high-risk customers (Cat 2).
    """
    def __init__(self, gamma: float = 2.0, num_classes: int = 3, device="cpu"):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        
        # Cost Matrix (True x Predicted)
        # 0: No Delay, 1: Current (0-30 DPD), 2: Past Due+ (31+ DPD)
        # We heavily penalize true=2 predicted=0 (cost 4.0)
        self.cost_matrix = torch.tensor([
            [0.0, 0.5, 1.0],  # True 0
            [1.5, 0.0, 0.5],  # True 1
            [4.0, 2.0, 0.0]   # True 2
        ], dtype=torch.float32, device=device)
        
    def to(self, device):
        self.cost_matrix = self.cost_matrix.to(device)
        return super().to(device)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) raw unnormalized scores
            targets: (B,) class indices
        """
        # Softmax probabilities: (B, num_classes)
        probs = F.softmax(logits, dim=1)
        
        # Log probabilities: (B, num_classes)
        log_probs = F.log_softmax(logits, dim=1)
        
        batch_size = targets.size(0)
        
        # Gather the cost for each predicted class given the true target
        # targets shape (B,), we want to index row `targets[i]` of cost_matrix
        # costs shape: (B, num_classes)
        costs = self.cost_matrix[targets]
        
        # Create one-hot targets for the standard focal part: (B, num_classes)
        targets_oh = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Focal weight for the true class: (1 - pt)^gamma
        # (B,)
        pt = (probs * targets_oh).sum(dim=1)
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        
        # Standard focal loss component (applied to true class only)
        # - (1 - p_t)^gamma * log(p_t)
        ce_loss = - (targets_oh * log_probs).sum(dim=1)
        focal_loss_component = focal_weight * ce_loss
        
        # Cost penalty component
        # We also penalize the probability assigned to wrong classes based on the cost matrix
        # sum_j [ C_yj * p_j ]
        expected_cost = (probs * costs).sum(dim=1)
        
        # Combine: focal loss scales the cross entropy, and we add the expected cost
        # The expected cost term guides the network away from high-cost mistakes 
        # even if it's confident
        total_loss = focal_loss_component + expected_cost
        
        return total_loss.mean()
