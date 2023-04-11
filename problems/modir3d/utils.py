import os
import cv2
import torch
import numpy as np
import matplotlib
import logging


def postprocess():
    pass


def calculate_dice(y_true, y_pred):
    """
    y_true, y_pred = batchsize * num_class * d * h * w tensors
    """
    ndims = y_pred.ndim - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = top / bottom
    return dice


def normalize_z(val):
    color = int(31 / 2 + val)
    color = np.clip(color, 0, 31)
    return color


def dvf_to_im(dvf, n_control_points=32, scale_factor=1):
    """
    input: 2D dvf (h*w*3)
    output: vizualization of dvf (h*w*3)
            dvf_x, dvf_y define angle of arrow
            dvf_z
    """
    cmap1 = matplotlib.colormaps['spring'](np.linspace(0, 1, 32))
    cmap = matplotlib.colormaps['cool'](np.linspace(0, 1, 32))
    cmap[16:] = cmap1[:16]
    cmap = cmap*255

    h, w, _ = dvf.shape
    grid_x, grid_y = np.meshgrid(np.linspace(0, w-1, n_control_points), np.linspace(0, h-1, n_control_points), indexing="xy")
    grid_x, grid_y = grid_x.astype(np.int32).reshape(-1), grid_y.astype(np.int32).reshape(-1)

    image = np.zeros((h, w, 3)).astype(np.uint8)
    for i in range(grid_x.size):
        x, y = grid_x[i], grid_y[i]
        x_val = -1 * int(dvf[y, x, 0] * scale_factor)  #convert dvf to pixels
        y_val = -1 * int(dvf[y, x, 1] * scale_factor)
        z_val = -1 * dvf[y, x, 2] * scale_factor
        z_color = cmap[normalize_z(z_val)][:-1][::-1]
        image = cv2.arrowedLine(image, (x, y), (x + x_val, y + y_val),
                                        z_color, 1)
    
    # image = cv2.resize(image, (h, w))
    return image


def validation(ensemble_class, validation_dataloader, criterion, cache, visualize=True):
    """
    runs on entire validation data
    """
    nsol = ensemble_class.n_mo_sol
    nobj = ensemble_class.n_mo_obj

    # eval mode on in each network
    net_list = ensemble_class.net_list
    for i, net in enumerate(net_list):
        net.eval()

    loss_per_sample_list = [[] for i in range(nsol)]
    for batch_no, data in enumerate(validation_dataloader):
        inputs = data["X"]
        targets = data["Y"]
        img1, img2 = inputs[0].numpy(), inputs[1].numpy()
        outs = []
        dvfs = []
        for i_mo_sol in range(0, nsol):
            with torch.no_grad():
                out = net_list[i_mo_sol](inputs)
            loss_per_sample = criterion(out, targets)
            loss_per_sample = torch.stack(loss_per_sample, dim=0)
            loss_per_sample_list[i_mo_sol].append(loss_per_sample.data.cpu().numpy())
            outs.append(out[0].data.cpu().numpy())
            dvf = out[1].permute(0, 2, 3, 4, 1)
            dvfs.append(dvf.data.cpu().numpy())

        if visualize:
            # sorting outs for visualization
            losses = [item[-1] for item in loss_per_sample_list]
            losses = np.array(losses) #nsol * nobj * nsample
            sort_indices = [np.argsort(losses[:, 0, i]) for i in range(losses.shape[2])]

            batchsize = img1.shape[0]
            for i in range(batchsize):
                outs_i = [item[i, 0, :, :, :] for item in outs]
                outs_i = [outs_i[idx] for idx in sort_indices[i]] #sort outs_i acc to sort indices
                dvfs_i = [item[i, :, :, :, :] for item in dvfs]
                dvfs_i = [dvfs_i[idx] for idx in sort_indices[i]] #sort outs_i acc to sort indices
                im1 = img1[i, 0, :, :, :]
                im2 = img2[i, 0, :, :, :]

                nslices = im1.shape[0]
                for i_slice in range(nslices):
                    outs_slice = [item[i_slice] for item in outs_i]
                    ims = [im1[i_slice]] + outs_slice + [im2[i_slice]]
                    dvfs_slice = [item[i_slice] for item in dvfs_i]
                    dvfs_slice_im = [dvf_to_im(item) for item in dvfs_slice]
                    empty_im = np.zeros_like(dvfs_slice_im[0])
                    ims_dvf = [empty_im] + dvfs_slice_im + [empty_im]
                    img11 = np.concatenate(ims, axis=1)
                    img11 = cv2.cvtColor((img11*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    img22 = np.concatenate(ims_dvf, axis=1)
                    img = np.concatenate((img11, img22), axis=0)

                    cv2.imwrite(os.path.join(cache.out_dir_val, "im{}_slice{}.jpg".format(batch_no*batchsize + i, i_slice)), img)

    loss_per_sample = [np.concatenate(arr_list, axis=1) for arr_list in  loss_per_sample_list] #list of obj * samples
    mo_obj_val_sample = np.array(loss_per_sample).transpose(2,1,0)  #samples * nobj * nsol
    assert mo_obj_val_sample.ndim==3
    assert(mo_obj_val_sample.shape[1:] == (nobj, nsol))
    m_obj_val_mean = np.mean(mo_obj_val_sample, axis=0)

    for i, net in enumerate(net_list):
        net.train()
    
    metrics = {"loss": mo_obj_val_sample}
    return metrics