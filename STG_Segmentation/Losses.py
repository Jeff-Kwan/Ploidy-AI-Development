import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinationLoss(nn.Module):
    """
    Combines FocalLoss and SoftDiceLoss with user-defined weights.

    Args:
        dice_weight (float): Weight for the SoftDiceLoss component.
        focal_weight (float): Weight for the FocalLoss component.
        dice_params (dict): Keyword arguments for initializing the SoftDiceLoss.
        focal_params (dict): Keyword arguments for initializing the FocalLoss.
    
    Example:
        >>> combo_crit = CombinationLoss(
        ...     dice_weight=0.5,
        ...     focal_weight=0.5,
        ...     dice_params={"smooth": 1.0, "reduction": "mean"},
        ...     focal_params={"alpha": 0.25, "gamma": 2.0, "reduction": "mean"}
        ... )
        >>> logits = torch.randn(4, 1, 256, 256)
        >>> targets = torch.randint(0, 2, (4, 1, 256, 256))
        >>> loss_val = combo_crit(logits, targets)
        >>> print(loss_val.item())
    """
    def __init__(self,
                 dice_weight=0.5,
                 focal_weight=0.0,
                 bce_weight=0.5,
                 dice_params=None,
                 focal_params=None,
                 bce_params=None):
        super(CombinationLoss, self).__init__()
        if dice_params is None:
            dice_params = {}
        if focal_params is None:
            focal_params = {}
        if bce_params is None:
            bce_params = {}

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        
        # Instantiate the SoftDiceLoss and FocalLoss with user-specified params
        self.dice_loss = SoftDiceLoss(**dice_params)
        self.focal_loss = FocalLoss(**focal_params)
        self.bce_loss = nn.BCEWithLogitsLoss(**bce_params)

    def forward(self, logits, targets):
        """
        Compute the weighted sum of SoftDiceLoss and FocalLoss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (N, 1, H, W).
            targets (torch.Tensor): Binary ground-truth mask of shape
                                    (N, 1, H, W) or (N, H, W).

        Returns:
            torch.Tensor: Scalar representing the combined loss.
        """
        loss = torch.tensor(0.0, device=logits.device)
        if self.dice_weight:
            loss = loss + self.dice_loss(logits, targets) * self.dice_weight
        if self.focal_weight:
            loss = loss + self.focal_loss(logits, targets) * self.focal_weight
        if self.bce_weight:
            loss = loss + self.bce_loss(logits, targets) * self.bce_weight
        return loss


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for segmentation or any 2D binary classification map.

    Args:
        alpha (float): Weight for the positive (foreground) class. 
                       The negative (background) class gets (1-alpha).
                       E.g. alpha=0.25 up-weights the positive class.
        gamma (float): Focusing parameter. Typical default = 2.0
        reduction (str): 'none' | 'mean' | 'sum'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): shape (B,1,H,W) or (B,H,W)
                - Raw, un-sigmoided logits for each pixel.
            targets (torch.Tensor): shape (B,H,W), values in {0,1}.
        """
        # If logits has a channel-dim of 1, squeeze it to match targets shape
        if logits.dim() > targets.dim():
            logits = logits.squeeze(1)  # becomes (B,H,W)

        # Compute binary cross entropy (elementwise)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            reduction='none'
        )  # shape: (B,H,W)

        # pt = exp(-bce) = p if y=1 else (1-p) if y=0
        pt = torch.exp(-bce)

        # Compute alpha_t:
        # alpha for positives, (1-alpha) for negatives.
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # shape: (B,H,W)

        # Focal term = alpha_t * (1-pt)^gamma * BCE
        focal_term = alpha_t * ((1 - pt) ** self.gamma) * bce

        if self.reduction == 'mean':
            return focal_term.mean()
        elif self.reduction == 'sum':
            return focal_term.sum()
        else:
            return focal_term



class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Args:
        smooth (float): Smoothing constant to avoid division by zero. Default=1.0
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'. Default='mean'.
    """
    def __init__(self, smooth=1.0, reduction='mean'):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Forward pass for Soft Dice Loss.

        Args:
            logits (torch.Tensor): Model logits of shape (N, 1, H, W).
            targets (torch.Tensor): Ground-truth binary masks of shape (N, 1, H, W) or (N, H, W).

        Returns:
            torch.Tensor: Scalar loss by default (if reduction is 'mean' or 'sum'),
                          or per-sample loss (if reduction='none').
        """
        if logits.dim() != 4 or logits.size(1) != 1:
            raise ValueError("logits must be of shape (N, 1, H, W).")
        
        # If targets has shape (N, H, W), unsqueeze the channel dimension
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Convert targets to float for consistent multiplication
        targets = targets.float()

        # Apply sigmoid to get probabilities in [0,1]
        probs = torch.sigmoid(logits)

        # Flatten channel and spatial dimensions (batch dimension stays)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # Compute per-batch Dice
        intersection = (probs_flat * targets_flat).sum(dim=1)
        denominator = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice_score  # Soft Dice Loss = 1 - Dice Coefficient

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        elif self.reduction == 'none':
            return dice_loss
        else:
            raise ValueError(f"Unsupported reduction mode: {self.reduction}")
