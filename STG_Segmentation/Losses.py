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
                 focal_weight=0.5,
                 dice_params=None,
                 focal_params=None):
        super(CombinationLoss, self).__init__()
        if dice_params is None:
            dice_params = {}
        if focal_params is None:
            focal_params = {}

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Instantiate the SoftDiceLoss and FocalLoss with user-specified params
        self.dice_loss = SoftDiceLoss(**dice_params)
        self.focal_loss = FocalLoss(**focal_params)

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
        dice_val = self.dice_loss(logits, targets)
        focal_val = self.focal_loss(logits, targets)

        loss = self.dice_weight * dice_val + self.focal_weight * focal_val
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation.
    
    Args:
        alpha (float): Weighting factor for the rare class (0 < alpha <= 1). 
                       Often set to 0.25 for the minority class (e.g., foreground).
        gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted. 
                       Typically ranges from 1.0 to 5.0. Commonly set to 2.0.
        reduction (str): Specifies the reduction to apply to the output: 
                         'none' | 'mean' | 'sum'.
                         'none': no reduction will be applied.
                         'mean': the sum of the output will be divided by the number of elements in the output.
                         'sum': the output will be summed.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for focal loss.

        Args:
            inputs (torch.Tensor): Predicted logits of shape (N, 1, H, W) or (N, H, W).
                                   - If shape is (N, H, W), targets should be similarly shaped.
                                   - If shape is (N, 1, H, W), you may need to squeeze the channel dimension to match the targets.
            targets (torch.Tensor): Binary ground-truth mask of shape (N, H, W) with values in {0,1}.

        Returns:
            torch.Tensor: Focal loss value.
        """
        # Ensure inputs and targets are the same shape for element-wise operations
        # If inputs has an extra channel dimension, we can squeeze it
        if inputs.dim() > targets.dim():
            inputs = inputs.squeeze(1)

        # Compute binary cross entropy with logits, without reduction (element-wise)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, 
            targets.float(),  # ensure targets are float
            reduction='none'
        )
        
        # For numerical stability, compute the "pt" term, where
        # pt = p if y=1 and pt = (1-p) otherwise
        # After BCE, we can interpret exp(-bce_loss) as "pt"
        pt = torch.exp(-bce_loss)
        
        # Focal loss term
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss

        # Apply the requested reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



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
