import torch.nn as nn
from .dice_loss import DiceLoss

class DiceCELoss(nn.Module):
    """
    Dice Loss + CrossEntropy Loss
    """

    def __init__(self, weight_dice=0.5, weight_ce=0.5, smooth=1e-6):
        super(DiceCELoss, self).__init__()
        self.dice = DiceLoss(smooth=smooth, reduction="mean")
        self.ce = nn.CrossEntropyLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        ce_loss = self.ce(inputs, targets)
        return self.weight_dice * dice_loss + self.weight_ce * ce_loss
