"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples"
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt


class FocalLoss(nn.Module):
    """ Focal Loss - https://arxiv.org/abs/1708.02002
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Taken from https://github.com/NVIDIA/retinanet-examples/blob/f1671353cb657126354543e9e94ecb6b65c5ddfd/retinanet/loss.py
    """

    def __init__(self, alpha=0.25, gamma=2):
        """
        Args:
            alpha:  Float between 0 and 1. weight factor for positive examples
            gamma:  Float 1 or higher to indicate the emphasis on misclassified examples
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target):
        target = target.float()
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        pt = torch.where(target == 1,  pred, 1 - pred)
        loss = alpha * (1. - pt) ** self.gamma * ce
        return torch.sum(loss)



def test_val():

    labels = torch.tensor([[0],
                           [0]]).float()
    predictions = torch.tensor([[0.8042],
                                [0.217]]).float()

    logits = np.log(predictions / (1 - predictions))

    gamma = 5.0
    alpha = 0.9

    cb_loss_function = FocalLoss( alpha, gamma)

    cb_loss = cb_loss_function(logits, labels)

    print(cb_loss)


if __name__ == '__main__':

    test_val()

    #plot_values()