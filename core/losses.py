'''
    Custom losses for the model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
    '''
        Custom loss function to penalize false positives more than false negatives

    '''
    def __init__(self, weight_fp: float = 1., weight_fn: float = 1.):
        '''
            Initialize the loss function with a weight for false positives
                
            Parameters:
                * weight_fp (float): The weight to apply to false positives
                (the higher the weight, the more false positives are penalized)
                * weight_fn (float): The weight to apply to false negatives
                (the higher the weight, the more false negatives are penalized)
        '''
        super(WeightedBCELoss, self).__init__()
        self.weight_fp = weight_fp
        self.weight_fn = weight_fn

    def forward(self, y_pred, y_true):
        loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        weights_fp = torch.where(y_true == 0., self.weight_fp, 1.)
        weights_fn = torch.where(y_true == 1., self.weight_fn, 1.)
        weighted_loss = loss * weights_fp * weights_fn
        return weighted_loss.mean()