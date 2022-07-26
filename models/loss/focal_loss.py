import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2., reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCELoss(reduction='none')
        self.reduction = reduction

    def forward(self, input, target):
        bce_loss = self.bce_loss(input, target)

        alpha_factor = torch.where(torch.eq(target, 1.), self.alpha, 1.-self.alpha)
        focal_weight = torch.where(torch.eq(target, 1.), 1.-input, input)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

        loss = focal_weight * bce_loss

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss
        