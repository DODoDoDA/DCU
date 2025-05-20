import torch.nn as nn
import torch.nn.functional as F
import torch

# kd loss
class SoftTarget(nn.Module):
    def __init__(self, T=2):
        super(SoftTarget, self).__init__()
        self.T = T
    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss
    
# weighted kd loss
class loss_KD(nn.Module):
    def __init__(self, T=2):
        super(loss_KD, self).__init__()
        self.T = T
    def forward(self, out_s, out_t, weight = None):
        if weight is None:
            loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T
            return loss
        else:
            loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                            F.softmax(out_t/self.T, dim=1),
                            reduction='none') * self.T * self.T
            weighted_loss = torch.mean(weight * torch.sum(loss,dim=1))
            return weighted_loss
        
class loss_PreCE(nn.Module):
    def __init__(self):
        super(loss_PreCE, self).__init__()
        self.loss_normal = nn.CrossEntropyLoss()
        self.loss_none = nn.CrossEntropyLoss(reduction="none")
    def forward(self, pre, tat, weight = None):
        if weight is None:
            return self.loss_normal(pre, tat)
        else:
            loss = self.loss_none(pre, tat)
            weighted_loss = torch.mean(weight*loss)
            return weighted_loss

def kd_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]