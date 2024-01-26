import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ScaledMSELoss(nn.Module):
    """mse loss scaled by 0.01"""
    def __init__(self, reduction='none'):
        super(ScaledMSELoss, self).__init__()
        self.reduction = reduction


    def forward(self, inputs, target):
        """
        out = 0.01 * mse_loss(inputs, target)
        """
        out = 0.01 * torch.nn.functional.mse_loss(inputs, target, reduction=self.reduction) 
        return out


class Loss(nn.Module):
    """Evaluation of two losses"""
    def __init__(self, loss_name_list):
        super(Loss, self).__init__()
        self.implemented_loss = ["MSELoss", "L1Loss", "ScaledMSELoss"]

        self.loss_list = []
        for loss_name in loss_name_list:
            if loss_name not in self.implemented_loss:
                raise NotImplementedError("{} not implemented. Implemented losses are: {}".format(loss_name, self.implemented_loss))
            elif loss_name == "MSELoss":
                self.loss_list.append( torch.nn.MSELoss(reduction='none') )
            elif loss_name == "L1Loss":
                self.loss_list.append( torch.nn.L1Loss(reduction='none') )
            elif loss_name == "ScaledMSELoss":
                self.loss_list.append( ScaledMSELoss(reduction='none') )

    def forward(self, inputs, target):
        """
        out_list = list of losses, where each loss is a tensor of losses for each sample
        """
        assert(target.shape[1] == len(self.loss_list))
        target = target.to(inputs.device)
        out_list = []
        for i, loss_fn in enumerate(self.loss_list):
            out = loss_fn(inputs, target[:, i][:, None])
            out_list.append(out.view(-1))

        return out_list


