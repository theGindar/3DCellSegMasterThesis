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

# ohne non-edge class
def dice_loss_org_individually(pred, target):
    """
    Computes the sum of dice loss of every sample in the minibatch.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    epsilon = 1e-7

    # get batchsize
    N = pred.size(0)
    # have to use contiguous since they may from a torch.view op
    # iflat and tflat are of size (N, C*X*Y*Z)
    iflat = pred.contiguous().view(N, -1)
    tflat = target.contiguous().view(N, -1)

    intersection = 2. * torch.sum(torch.mul(iflat, tflat), dim=1)

    A_sum = torch.sum(torch.mul(iflat, iflat), dim=1)
    B_sum = torch.sum(torch.mul(tflat, tflat), dim=1)

    return torch.mean(1 - ((intersection) / (A_sum + B_sum + epsilon)))


# ohne non-edge class
def dice_loss_org_individually_with_weights(pred, target, weights):
    """
    Computes the sum of dice loss of every sample in the minibatch.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    epsilon = 1e-7

    # get batchsize
    N = pred.size(0)
    # have to use contiguous since they may from a torch.view op
    # iflat and tflat are of size (N, C*X*Y*Z)
    iflat = pred.contiguous().view(N, -1)
    tflat = target.contiguous().view(N, -1)
    weights_flat = weights.contiguous().view(N, -1)



    intersection = 2. * torch.sum(torch.mul(torch.mul(iflat, tflat), weights_flat), dim=1)

    A_sum = torch.sum(torch.mul(iflat, iflat), dim=1)
    B_sum = torch.sum(torch.mul(tflat, tflat), dim=1)

    return torch.mean(1 - ((intersection) / (A_sum + B_sum + epsilon)))


# ohne non-edge class
def dice_loss_org_individually_with_cellsegloss_and_weights(pred, target, weights):
    """
    Computes the sum of dice loss of every sample in the minibatch.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    epsilon = 1e-7
    delta = .1

    # get batchsize
    N = pred.size(0)
    # have to use contiguous since they may from a torch.view op
    # iflat and tflat are of size (N, C*X*Y*Z)
    iflat = pred.contiguous().view(N, -1)
    tflat = target.contiguous().view(N, -1)
    weights_flat = weights.contiguous().view(N, -1)

    intersection = 2. * torch.sum(torch.mul(torch.mul(iflat / (iflat + delta), tflat), weights_flat), dim=1)

    A_sum = torch.sum(torch.mul(iflat, iflat), dim=1)
    B_sum = torch.sum(torch.mul(tflat, tflat), dim=1)

    return torch.mean(1 - ((intersection) / (A_sum + B_sum + epsilon)))


# ohne non-edge class
def dice_loss_org_individually_with_cellsegloss(pred, target):
    """
    Computes the sum of dice loss of every sample in the minibatch.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    epsilon = 1e-7
    delta = .1

    # get batchsize
    N = pred.size(0)
    # have to use contiguous since they may from a torch.view op
    # iflat and tflat are of size (N, C*X*Y*Z)
    iflat = pred.contiguous().view(N, -1)
    tflat = target.contiguous().view(N, -1)

    intersection = 2. * torch.sum(torch.mul(iflat / (iflat + delta), tflat), dim=1)

    A_sum = torch.sum(torch.mul(iflat, iflat), dim=1)
    B_sum = torch.sum(torch.mul(tflat, tflat), dim=1)

    return torch.mean(1 - ((intersection) / (A_sum + B_sum + epsilon)))


# ohne non-edge class
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


# ohne non-edge class
def balanced_cross_entropy_with_weights(pred, target, boundary):
    N = pred.size(0)
    iflat = pred.contiguous().view(N, -1)
    tflat = target.contiguous().view(N, -1)
    boundary_flat = boundary.contiguous().view(N, -1)
    # parameter for weighting positive and negatives, beta is the percentage of non-edge voxels to edge voxels
    edge_percentage = torch.sum(tflat, dim=1) / tflat.size(dim=1)
    beta = 1. - edge_percentage

    tflat_inverted = 1 - tflat

    weight = torch.unsqueeze(beta, dim=1).expand_as(tflat) * tflat + \
             torch.unsqueeze(edge_percentage, dim=1).expand_as(tflat_inverted) * tflat_inverted

    # add boundary weights
    weight[boundary_flat > 0] = 0.5

    return F.binary_cross_entropy(iflat, tflat, weight=weight)


# ohne non-edge class
def balanced_cross_entropy_with_weights_II(pred, target, boundary):
    N = pred.size(0)
    iflat = pred.contiguous().view(N, -1)
    tflat = target.contiguous().view(N, -1)
    boundary_flat = boundary.contiguous().view(N, -1)
    # parameter for weighting positive and negatives, beta is the percentage of non-edge voxels to edge voxels
    edge_percentage = torch.sum(tflat, dim=1) / tflat.size(dim=1)
    beta = 1. - edge_percentage

    tflat_inverted = 1 - tflat

    weight = torch.unsqueeze(beta, dim=1).expand_as(tflat) * tflat + \
             torch.unsqueeze(edge_percentage, dim=1).expand_as(tflat_inverted) * tflat_inverted

    # add boundary weights
    weight[boundary_flat > 0] = beta

    return F.binary_cross_entropy(iflat, tflat, weight=weight)


# ohne non-edge class
def cross_entropy_with_weights(pred, target, boundary, boundary_edge):
    # target should be foreground
    N = pred.size(0)
    iflat = pred.contiguous().view(N, -1)
    tflat = target.contiguous().view(N, -1)
    boundary_flat = boundary.contiguous().view(N, -1)
    boundary_edge_flat = boundary_edge.contiguous().view(N, -1)
    # parameter for weighting positive and negatives, beta is the percentage of non-edge voxels to edge voxels
    weight = torch.ones_like(iflat) * 0.5


    # add boundary weights
    weight[boundary_flat > 0] = 1.0
    weight[boundary_edge_flat > 0] = 1.0

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


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target):
        return compute_per_channel_dice(input, target, weight=self.weight)



class WeightedCrossEntropyLoss(nn.Module):
    # from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py#L199
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
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
        return