import torch
import torch.nn as nn


class NCCLoss(nn.Module):
    """Normalized Cross-Correlation
    output = sum( (input - input_mean) * (target - target_mean) )/
             sqrt( sum( (input - input_mean)**2 ) + sum( (target - target_mean)**2 ) )

    original implementation from:
    https://github.com/yuta-hi/pytorch_similarity
    """
    def __init__(self, reduction='mean'):
        super(NCCLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        b = input.shape[0]

        # reshape
        input = input.view(b, -1)
        target = target.view(b, -1)

        # mean
        input_mean = torch.mean(input, dim=1, keepdim=True)
        target_mean = torch.mean(target, dim=1, keepdim=True)

        # deviation
        input = input - input_mean
        target = target - target_mean

        dev_xy = torch.mul(input, target)
        dev_xx = torch.mul(input, input)
        dev_yy = torch.mul(target, target)

        dev_xy_sum = torch.sum(dev_xy, dim=1, keepdim=True)
        dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
        dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

        eps = 1e-6
        ncc = torch.div(dev_xy_sum + eps,
                    torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + eps)

        if self.reduction=='mean':
            ncc = 1 - torch.mean(ncc)
        else:
            ncc = 1 - ncc
        return ncc.view(-1)


class BendingEnergyLoss(nn.Module):
    """Regularization loss for DIR
    Defined as spatial gradients of the DVF;
    Approximated by difference formula
    """
    def __init__(self, reduction='mean'):
        super(BendingEnergyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs):
        """
        inputs: dvf with dimesions b, h, w, 2
        """
        b, h, w, _ = inputs.shape
        # mag = torch.norm(inputs, dim=3)
        dh = torch.roll(inputs, 1, dims=1) - inputs
        dw = torch.roll(inputs, 1, dims=2) - inputs
        dhh = torch.roll(dh, 1, dims=1) - dh
        dww = torch.roll(dw, 1, dims=2) - dw
        dhw = torch.roll(dh, 1, dims=2) - dh

        loss = dhh.pow(2) + 2 * dhw.pow(2) + dww.pow(2)
        # summation over dvf
        loss = torch.sum(loss, dim=3)
        # reduction over h*w
        loss = torch.mean(loss, dim=(1, 2))
        # reduction over batchsize
        if self.reduction=='mean':
            loss = torch.mean(loss)
        return loss.view(-1)


class TransformationLoss(nn.Module):
    """Regularization loss for DIR
    Defined as magnitude of displacement;
    """
    def __init__(self, reduction='mean'):
        super(TransformationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs):
        """
        inputs: dvf with dimesions b, h, w, 2 or b, d, h, w, 3
        """
        ndim = inputs.ndim
        b = inputs.shape[0]
        mag = torch.norm(inputs, dim=ndim-1)
        mag = mag.view(b, -1)
        # reduction over h*w or d*h*w
        loss = torch.mean(mag, dim=1)
        # reduction
        if self.reduction=='mean':
            loss = torch.mean(loss)
        return loss.view(-1)
        
        
class Loss(nn.Module):
    """Evaluation of two losses"""
    def __init__(self, lossname_list):
        super(Loss, self).__init__()
        implemented_loss = ["NCCLoss", "TransformationLoss", "BendingEnergyLoss"]
        self.loss_list = []
        for loss in lossname_list:
            if loss not in implemented_loss:
                raise NotImplementedError("{} not implemented. Implemented losses are: {}".format(loss, implemented_loss))
            elif loss == "NCCLoss":
                self.loss_list.append(NCCLoss(reduction='none'))
            elif loss == "TransformationLoss":
                self.loss_list.append(TransformationLoss(reduction='none'))
            elif loss == "BendingEnergyLoss":
                self.loss_list.append(BendingEnergyLoss(reduction='none'))


    def forward(self, inputs, target):
        target = target.to(inputs[0].device)
        loss1 = self.loss_list[0](inputs[0], target)
        loss2 = self.loss_list[1](inputs[1])
        return (loss1, loss2)