import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def dice_loss_org_weights(pred, target, weights):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat=weights.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(torch.mul(iflat, tflat), weights_flat))

    #A_sum = torch.sum(torch.mul(torch.mul(iflat, iflat), weights_flat))
    #B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat), weights_flat))
    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_org(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(iflat, tflat))

    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))


def dice_loss_org_individually(pred, target):
    """
    Computes the sum of dice loss of every sample in the minibatch.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1

    # get batchsize
    N = pred.size(0)
    # have to use contiguous since they may from a torch.view op
    # iflat and tflat are of size (N, C*X*Y*Z)
    iflat = pred.contiguous().view(N, -1)
    tflat = target.contiguous().view(N, -1)

    intersection = 2. * torch.sum(torch.mul(iflat, tflat), dim=1)

    A_sum = torch.sum(torch.mul(iflat, iflat), dim=1)
    B_sum = torch.sum(torch.mul(tflat, tflat), dim=1)

    return torch.mean(1 - ((intersection + smooth) / (A_sum + B_sum + smooth)))


def balanced_cross_entropy(pred, target):
    N = pred.size(0)
    iflat = pred.contiguous().view(N, -1)
    tflat = target.contiguous().view(N, -1)
    # parameter for weighting positive and negatives, beta is the percentage of non-edge voxels to edge voxels
    edge_percentage = torch.sum(tflat, dim=1) / tflat.size(dim=1)
    beta = 1. - edge_percentage

    tflat_inverted = 1 - tflat

    weight = torch.unsqueeze(beta, dim=1).expand_as(tflat) * tflat + \
             torch.unsqueeze(edge_percentage, dim=1).expand_as(tflat_inverted) * tflat_inverted

    return F.binary_cross_entropy(iflat, tflat, weight=weight)


def dice_loss_II_weights(pred, target, weights):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat=weights.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(torch.mul(iflat/(iflat+delta), tflat),weights_flat))

    #A_sum = torch.sum(torch.mul(torch.mul(iflat/(iflat+delta), iflat/(iflat+delta)),weights_flat))
    #B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat),weights_flat))
    A_sum = torch.sum(torch.mul(iflat/(iflat+delta), iflat/(iflat+delta)))
    B_sum = torch.sum(torch.mul(tflat, tflat))

    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_II(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(iflat/(iflat+delta), tflat))

    A_sum = torch.sum(torch.mul(iflat/(iflat+delta), iflat/(iflat+delta)))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_III_weights(pred, target, weights, alpha=2):
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat=weights.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(torch.mul(torch.pow(iflat, alpha), tflat),weights_flat))

    A_sum = torch.sum(torch.mul(torch.mul(torch.pow(iflat, alpha), torch.pow(iflat, alpha)),weights_flat))
    B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat),weights_flat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_III(pred, target, alpha=2):
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(torch.pow(iflat, alpha), tflat))

    A_sum = torch.sum(torch.mul(torch.pow(iflat, alpha), torch.pow(iflat, alpha)))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_accuracy(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(iflat, tflat))

    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return (intersection) / (A_sum + B_sum + 0.0001)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class WeightedCrossEntropyLoss(nn.Module):
    # from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # reshape from (batch, 1, 64, 64, 64) to (batch, 64, 64, 64)
        #inp = torch.reshape(input, (input.shape[0],
        #                            input.shape[2],
        #                            input.shape[3],
        #                            input.shape[4]))
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights