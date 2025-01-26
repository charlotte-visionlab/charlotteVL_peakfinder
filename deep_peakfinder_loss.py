
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparsePeakLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super(SparsePeakLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha  # Weight for false positive penalty
        self.beta = beta    # Weight for sparsity

    def forward(self, outputs, targets):
        # BCE loss for detection
        bce_loss = self.bce(outputs, targets)

        # Sparsity penalty (L1 norm to encourage few peaks)
        sparsity_loss = torch.mean(torch.abs(outputs))

        # Combine losses
        total_loss = bce_loss + self.alpha * sparsity_loss
        return total_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class SparsePeakDetectionLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(SparsePeakDetectionLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.sparsity_weight = 0.1  # Weight for sparsity penalty

    def forward(self, outputs, targets):
        # Focal Loss for detection
        focal_loss = self.focal_loss(outputs, targets)

        # Sparsity penalty to encourage sparse predictions
        sparsity_loss = torch.mean(torch.abs(outputs))

        # Total loss
        total_loss = focal_loss + self.sparsity_weight * sparsity_loss
        return total_loss


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Computes IoU loss between predicted and target binary masks.

        Args:
            predictions: Tensor of shape (batch_size, length), predicted binary values (0 or 1).
            targets: Tensor of shape (batch_size, length), ground truth binary values (0 or 1).

        Returns:
            IoU loss: 1 - mean IoU score across the batch.
        """
        # Ensure predictions are probabilities (apply sigmoid if logits are used)
        predictions = torch.sigmoid(predictions)  # Convert logits to probabilities
        # predictions = (predictions > 0.5).float()  # Threshold probabilities to binary

        # Compute intersection and union
        intersection = torch.sum(predictions * targets, dim=1)
        union = torch.sum(predictions + targets, dim=1) - intersection

        # Compute IoU
        iou = intersection / (union + 1e-8)  # Add epsilon to avoid division by zero

        # Compute IoU loss
        iou_loss = 1 - iou.mean()  # Mean IoU across the batch
        return iou_loss


class WeightedIoULoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        """
        Initializes the weighted IoU loss function.

        Args:
            pos_weight: Weight for the positive class (peaks).
            neg_weight: Weight for the negative class (background).
        """
        super(WeightedIoULoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, predictions, targets):
        """
        Computes the weighted IoU loss between predicted probabilities and target binary masks.

        Args:
            predictions: Tensor of shape (batch_size, length), predicted probabilities (0 to 1).
            targets: Tensor of shape (batch_size, length), ground truth binary values (0 or 1).

        Returns:
            Weighted IoU loss: 1 - weighted mean IoU score across the batch.
        """
        # Apply sigmoid to predictions if they are logits
        predictions = torch.sigmoid(predictions)

        # Compute weights for positive and negative regions
        weights = targets * self.pos_weight + (1 - targets) * self.neg_weight

        # Compute weighted intersection and union
        intersection = torch.sum(weights * predictions * targets, dim=1)
        union = torch.sum(weights * (predictions + targets) - weights * predictions * targets, dim=1)

        # Compute IoU
        iou = intersection / (union + 1e-8)  # Add epsilon to prevent division by zero

        # Compute weighted IoU loss Returns 
        # 1−meanIoU, where higher IoU corresponds to lower loss.
        iou_loss = 1 - iou.mean()  # Mean IoU across the batch
        return iou_loss