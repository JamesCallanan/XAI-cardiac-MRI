import torch
import torch.nn as nn


def dice_loss(pred, target, smooth = 1.):
    """Calculate dice loss
    
    Args:
        - pred - predictions
        - target - ground truth
        - smooth [1.] - smoothing component so the dice won't crash
    
    Returns:
        - loss.mean() - avarage of losses from the single instance"""
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()
