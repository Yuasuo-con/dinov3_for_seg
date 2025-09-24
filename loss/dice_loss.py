import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for 2D segmentation (input: [B, C, H, W], target: [B, H, W])
    """

    def __init__(self, smooth=1e-6, reduction="mean"):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] (logits, 未经过 softmax/sigmoid)
            targets: [B, H, W] (long 类型, 存类别索引)
        """
        B, C, H, W = inputs.shape

        if C == 1:
            # 二分类 -> sigmoid
            probs = torch.sigmoid(inputs)  # [B, 1, H, W]
            probs = probs.squeeze(1)       # [B, H, W]
            targets_onehot = targets.float()
        else:
            # 多分类 -> softmax + one-hot
            probs = F.softmax(inputs, dim=1)  # [B, C, H, W]
            targets_onehot = F.one_hot(targets, num_classes=C)  # [B, H, W, C]
            targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # 展平
        probs = probs.contiguous().view(B, C, -1)
        targets_onehot = targets_onehot.contiguous().view(B, C, -1)

        # Dice score
        intersection = 2.0 * (probs * targets_onehot).sum(-1) + self.smooth
        denominator = probs.sum(-1) + targets_onehot.sum(-1) + self.smooth
        dice_score = intersection / denominator  # [B, C]

        loss = 1 - dice_score

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
