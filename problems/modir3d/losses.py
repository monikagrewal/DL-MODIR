import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class NCCLoss(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    original source: https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/torch/losses.py
    """

    def __init__(self, win=None, reduction="mean"):
        super(NCCLoss, self).__init__()
        self.win = win
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_true.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        if self.reduction=='mean':
            ncc = 1 - torch.mean(cc)
        else:
            ncc = 1 - torch.mean(cc, dim=(1,2,3,4))
        return ncc


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
        inputs: dvf with dimesions b, 2, h, w
        """
        inputs = inputs.permute(0, 2, 3, 1)
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


class SpatialGradientLoss3D(nn.Module):
    """Regularization loss for DIR
    Defined as spatial gradients of the DVF;
    Approximated by difference formula

    original source: https://github.com/voxelmorph/voxelmorph/tree/204b87fd6147ba6c7fed7e441b2f3e85ba3a6b74/voxelmorph/
    """
    def __init__(self, reduction='mean', penalty='l1'):
        super(SpatialGradientLoss3D, self).__init__()
        self.reduction = reduction
        self.penalty = penalty

    def forward(self, inputs):
        """
        inputs: dvf with dimesions b, 3, d, h, w
        """
        inputs = inputs.permute(0, 2, 3, 4, 1)
        dy = torch.abs(inputs[:, 1:, :, :, :] - inputs[:, :-1, :, :, :])
        dx = torch.abs(inputs[:, :, 1:, :, :] - inputs[:, :, :-1, :, :])
        dz = torch.abs(inputs[:, :, :, 1:, :] - inputs[:, :, :, :-1, :])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx, dim=(1,2,3,4)) + torch.mean(dy, dim=(1,2,3,4)) + torch.mean(dz, dim=(1,2,3,4))
        loss = d / 3.0
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
        inputs: dvf with dimesions b, 2, h, w or b, 3, d, h, w
        """
        b = inputs.shape[0]
        mag = torch.norm(inputs, dim=1)
        mag = mag.view(b, -1)
        # reduction over h*w or d*h*w
        loss = torch.mean(mag, dim=1)
        # reduction
        if self.reduction=='mean':
            loss = torch.mean(loss)
        return loss.view(-1)


class SegSimilarityLoss(nn.Module):
    """Soft Dice loss for segmentation masks
    """
    def __init__(self, reduction='mean'):
        super(SegSimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs):
        """
        inputs: fixed_seg, moving_seg_warped
        """
        fixed_seg, moving_seg_w = inputs[0], inputs[1]
        # ndims = fixed_seg.ndim - 2
        # vol_axes = list(range(2, ndims + 2))
        # top = 2 * (fixed_seg * moving_seg_w).sum(dim=vol_axes)
        # bottom = torch.clamp((fixed_seg + moving_seg_w).sum(dim=vol_axes), min=1e-5)
        # dice = torch.mean(top / bottom, dim=1)

        smooth = 1e-5
        ndims = fixed_seg.ndim - 1
        vol_axes = list(range(1, ndims + 1))
        top = 2 * (fixed_seg * moving_seg_w).sum(dim=vol_axes) + smooth
        bottom = fixed_seg.sum(dim=vol_axes) + moving_seg_w.sum(dim=vol_axes) + smooth
        dice = top / bottom
        # reduction
        if self.reduction=='mean':
            loss = 1 - torch.mean(dice)
        else:
            loss = 1 - dice
        return loss.view(-1)

        
class Loss(nn.Module):
    """Evaluation of two losses"""
    def __init__(self, lossname_list):
        super(Loss, self).__init__()
        implemented_loss = ["NCCLoss", "TransformationLoss", "BendingEnergyLoss", "SpatialGradientLoss3D", 
                            "SegSimilarityLoss"]
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
            elif loss == "SpatialGradientLoss3D":
                self.loss_list.append(SpatialGradientLoss3D(reduction='none'))
            elif loss == "SegSimilarityLoss":
                self.loss_list.append(SegSimilarityLoss(reduction='none'))


    def forward(self, inputs, targets):
        """
        inputs is a list of:
        [moving_im_warped, dvf, moving_seg_warped (Optional)]

        targets is a dict of:
        [fixed, moving, fixed_seg (Optional), moving_seg (Optional)]
        """
        target = target.to(inputs[0].device)
        losses = []
        for loss_fn in self.loss_list:
            if loss_fn.__class__.__name__ in ["NCCLoss", "NCCVoxelMorph"]:
                loss = loss_fn(inputs[0], target)
            elif loss_fn.__class__.__name__ in ["TransformationLoss",
                                              "BendingEnergyLoss",
                                              "SpatialGradientLoss3D"]:
                loss = loss_fn(inputs[1])
            elif loss_fn.__class__.__name__=="SegSimilarityLoss":
                loss = loss_fn([inputs[2], inputs[3]])
            else:
                raise ValueError(f"loss function {loss_fn.__class__.__name__} unknown.")
            
            losses.append( loss )

        return losses



