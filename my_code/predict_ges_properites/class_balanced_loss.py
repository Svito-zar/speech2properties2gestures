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


class BasicLoss(nn.Module):
    """
    Cross-Validation
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_logits, target):
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        return ce


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
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        pt = torch.where(target == 1,  pred, 1 - pred)
        return alpha * (1. - pt) ** self.gamma * ce


class ClassBalancedLoss(nn.Module):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
    """

    def __init__(self, samples_per_class, numb_of_classes, beta=0.95, alpha=0.25, gamma=2):
        """
        Args:
              samples_per_class: A python list of size [no_of_classes].
              numb_of_classes: total number of classes. int
              beta: float. Hyperparameter for Class balanced loss.
              gamma: float. Hyperparameter for Focal loss.
              alpha: float. Hyperparameter for Focal loss.
        """
        super().__init__()
        self.FocalLoss = FocalLoss(alpha, gamma)

        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * numb_of_classes
        self.cb_weights = weights

    def forward(self, pred_logits, labels):

        labels_one_hot = labels.float()

        weights = torch.tensor(self.cb_weights).float().to(pred_logits.device)

        cb_loss = self.FocalLoss(pred_logits, labels_one_hot)

        weighted_loss = weights * cb_loss

        return torch.sum(weighted_loss)


def test_val():

    no_of_classes = 3

    labels = torch.tensor([[0, 1, 1],
                           [0, 0, 0]]).float()
    predictions = torch.tensor([[0.2642, 0.3076, 0.2010],
                                [0.1917, 0.9089, 0.6967]]).float()

    logits = np.log(predictions / (1 - predictions))

    beta = 0.95
    gamma = 5.0
    samples_per_cls = [20, 300, 1]
    alpha = 0.9

    cb_loss_function = ClassBalancedLoss(samples_per_cls, no_of_classes, beta, alpha, gamma)

    cb_loss = cb_loss_function(logits, labels)

    print(cb_loss)


def plot_values():
    no_of_classes = 3

    labels = torch.tensor([1, 0, 0]).float()
    predictions = torch.tensor([0.3642, 0.01, 0.01]).float()

    beta = 0.95
    gamma = 5.0
    samples_per_cls = [20, 300, 1]  # ,2,2]
    alpha = 0.8

    cb_loss_function = ClassBalancedLoss(samples_per_cls, no_of_classes, beta, alpha, gamma)

    x = np.linspace(0.0001, 0.999, 10)

    y = [0 for range in range(len(x))]

    for i in range(len(x)):
        predictions[0] = x[i]
        logits = np.log(predictions / (1 - predictions))
        cb_loss = cb_loss_function(logits, labels)
        y[i] = cb_loss

    # Create plots with pre-defined labels.
    fig, ax = plt.subplots()

    ax.plot(x, y, label='line 1')  # + str(gamma)

    gamma = 1.0

    cb_loss_function = ClassBalancedLoss(samples_per_cls, no_of_classes, beta, alpha, gamma)

    x = np.linspace(0.0001, 0.999, 10)

    y = [0 for range in range(len(x))]

    for i in range(len(x)):
        predictions[0] = x[i]
        logits = np.log(predictions / (1 - predictions))
        cb_loss = cb_loss_function(logits, labels)
        y[i] = cb_loss

    ax.plot(x, y, label='line 2')  # + str(gamma)

    plt.show()


if __name__ == '__main__':

    test_val()

    #plot_values()