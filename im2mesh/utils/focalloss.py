import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if self.alpha is not None:
            self.alpha = torch.FloatTensor([1 - alpha, alpha]).to(device)

    def forward(self, pred, target):
        batch_size, n_pts = pred.size()
        pos = torch.sigmoid(pred)  # N x T
        neg = 1 - pos

        pt = torch.stack([neg, pos], dim=-1).view(-1, 2)
        index = target.view(-1, 1).long()

        pt = pt.gather(-1, index).view(-1)
        logpt = pt.log()

        if self.alpha is not None:
            at = self.alpha.gather(0, index.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt

        return loss.view(batch_size, n_pts)
