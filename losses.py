'''
Implementation of losses for interlead conversion.


Code by Sofia C. Beco, João Ribeiro Pinto, and Jaime S. Cardoso
in "Electrocardiogram lead conversion from single-lead blindly-segmented signals",
BMC Medical Informatics and Decision Making, 22(1): 314, 2022.
https://doi.org/10.1186/s12911-022-02063-6

'''


import torch 

from torch import nn


def AE_loss(X_rec_leads, X_leads, loss_fn, reduce=True):
    '''Computes a given loss function between a batch of reconstructed signals and the corresponding references.'''
    train_losses = list()
    train_loss=0
    for lead in range(len(X_leads)):
        train_losses.append(loss_fn(X_rec_leads[lead], X_leads[lead]))
        train_loss += 1/len(X_leads) * train_losses[-1]
    if reduce:
        return train_loss
    else:
        return train_losses


def corr(X_rec_leads, X_leads, reduce=True):
    '''Computes the correlation r between pairs of signals'''
    corrs = list()
    corr_coef=0
    for lead in range(len(X_leads)):
        vx = X_rec_leads[lead] - torch.mean(X_rec_leads[lead])
        vy = X_leads[lead] - torch.mean(X_leads[lead])
        corrs.append(torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-9))
        corr_coef += 1/len(X_leads) * corrs[lead]
    if reduce:
        return corr_coef 
    else:
        return corrs


class RMSELoss(nn.Module):
    '''Root mean square loss'''
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss