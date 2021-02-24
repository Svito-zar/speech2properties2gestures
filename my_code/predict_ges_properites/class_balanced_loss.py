"""Pytorch implementation of Class-Balanced-Loss: https://github.com/vandit15/Class-balanced-loss-pytorch
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



def focal_loss(labels, logits, alpha, gamma, positive_weight):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
      positive_weight:   a float scalar amplifying loss for positive examples

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    positive_w = torch.tensor([positive_weight for _ in range(labels.shape[1])])
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels,reduction="none", pos_weight=positive_w)

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, positive_weight):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = labels.float()

    weights = torch.tensor(weights).float().to(logits.device)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma, positive_weight)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss



if __name__ == '__main__':
    no_of_classes = 3
    logits = torch.rand(5,no_of_classes).float()
    labels = torch.randint(0,2, size = (5,no_of_classes))
    labels = torch.tensor([[0, 1, 1],
                            [0, 0, 0]]).float()
    logits = torch.tensor([[0.3642, 0.4676, 0.3010],
                           [0.2917, 0.1789, 0.2967]]).float()
    beta = 0.95
    gamma = 2.0
    samples_per_cls = [20,300,1] #,2,2]
    positive_weight = 4
    loss_type = "focal"
    print(labels)
    print(logits)
    cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma, positive_weight)
    print(cb_loss)
