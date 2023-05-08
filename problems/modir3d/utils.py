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
        x_val = int(dvf[y, x, 0] * scale_factor)  #convert dvf to pixels
        y_val = int(dvf[y, x, 1] * scale_factor)
        z_val = dvf[y, x, 2] * scale_factor
        z_color = cmap[normalize_z(z_val)][:-1][::-1]
        image = cv2.arrowedLine(image, (x, y), (x + x_val, y + y_val),
                                        z_color, 1)
    
    # image = cv2.resize(image, (h, w))
    return image


def seg_to_im(seg:np.array, num_classes:int=5) -> np.array:
    """
    input: 2D segmentation mask with integer value for each class
    """
    class_to_color = {1:(0,0,1), 2:(0,1,0), 3:(0,1,1), 4:(1,0,0)}
    seg_im = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
    for class_idx in range(1, num_classes):
        rr, cc = np.where(seg==class_idx)
        seg_im[rr, cc, :] = class_to_color[class_idx]
    return seg_im


def validation(ensemble_class, validation_dataloader, criterion, cache, visualize=True):
    """
    runs on entire validation data
    """
    nsol = ensemble_class.n_mo_sol
    nobj = ensemble_class.n_mo_obj

    # eval mode on in each network
    if ensemble_class.__class__.__name__=="DeepEnsemble":
        net_list = ensemble_class.net_list
        for i, net in enumerate(net_list):
            net.eval()
        use_segmentation = getattr(net_list[0], "use_segmentation", False)
    elif ensemble_class.__class__.__name__=="KHeadEnsemble":
        ensemble_class.model.eval()
        use_segmentation = getattr(ensemble_class.model, "use_segmentation", False)

    loss_per_sample_list = [[] for i in range(nsol)]
    if not use_segmentation:
        for batch_no, data in enumerate(validation_dataloader):
            inputs = data["X"]
            targets = data["Y"]
            img1, img2 = inputs[0].numpy(), inputs[1].numpy()
            outs = []
            dvfs = []
            if ensemble_class.__class__.__name__=="DeepEnsemble":
                for i_mo_sol in range(0, nsol):
                    with torch.no_grad():
                        out = net_list[i_mo_sol](inputs)
                    loss_per_sample = criterion(out, targets)
                    loss_per_sample = torch.stack(loss_per_sample, dim=0)
                    loss_per_sample_list[i_mo_sol].append(loss_per_sample.data.cpu().numpy())
                    outs.append(out[0].data.cpu().numpy())
                    dvf = out[1].permute(0, 2, 3, 4, 1)
                    dvfs.append(dvf.data.cpu().numpy())
            elif ensemble_class.__class__.__name__=="KHeadEnsemble":
                with torch.no_grad():
                    outs_torch = ensemble_class.model(inputs)
                for i_mo_sol in range(0, nsol):
                    out = outs_torch[i_mo_sol]
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
    else:
        for batch_no, data in enumerate(validation_dataloader):
            inputs = data["X"]
            targets = data["Y"]
            img1, img2 = inputs[0].numpy(), inputs[1].numpy()
            img1_seg = torch.argmax(inputs[2], dim=1).float().numpy()
            img2_seg = torch.argmax(inputs[3], dim=1).float().numpy()
            outs = []
            dvfs = []
            segs = []
            if ensemble_class.__class__.__name__=="DeepEnsemble":
                for i_mo_sol in range(0, nsol):
                    with torch.no_grad():
                        out = net_list[i_mo_sol](inputs)
                    loss_per_sample = criterion(out, targets)
                    loss_per_sample = torch.stack(loss_per_sample, dim=0)
                    loss_per_sample_list[i_mo_sol].append(loss_per_sample.data.cpu().numpy())
                    outs.append(out[0].data.cpu().numpy())
                    dvf = out[1].permute(0, 2, 3, 4, 1)
                    dvfs.append(dvf.data.cpu().numpy())
                    seg = torch.argmax(out[3], dim=1).float()
                    segs.append(seg.data.cpu().numpy())
            elif ensemble_class.__class__.__name__=="KHeadEnsemble":
                with torch.no_grad():
                    outs_torch = ensemble_class.model(inputs)
                for i_mo_sol in range(0, nsol):
                    out = outs_torch[i_mo_sol]
                    loss_per_sample = criterion(out, targets)
                    loss_per_sample = torch.stack(loss_per_sample, dim=0)
                    loss_per_sample_list[i_mo_sol].append(loss_per_sample.data.cpu().numpy())
                    outs.append(out[0].data.cpu().numpy())
                    dvf = out[1].permute(0, 2, 3, 4, 1)
                    dvfs.append(dvf.data.cpu().numpy())
                    seg = torch.argmax(out[3], dim=1).float()
                    segs.append(seg.data.cpu().numpy())

            if visualize:
                # sorting outs for visualization
                losses = [item[-1] for item in loss_per_sample_list]
                losses = np.array(losses) #nsol * nobj * nsample
                sort_indices = [np.argsort(losses[:, 0, i]) for i in range(losses.shape[2])]

                batchsize = img1.shape[0]
                for i in range(batchsize):
                    #sort outs_i acc to sort indices, and slice batch
                    outs_i = [outs[idx][i, 0, :, :, :] for idx in sort_indices[i]] 
                    dvfs_i = [dvfs[idx][i, :, :, :, :] for idx in sort_indices[i]]
                    segs_i = [segs[idx][i, :, :, :] for idx in sort_indices[i]]
                    im1 = img1[i, 0, :, :, :]
                    im2 = img2[i, 0, :, :, :]
                    im1_seg = img1_seg[i, :, :, :]
                    im2_seg = img2_seg[i, :, :, :]

                    nslices = im1.shape[0]
                    for i_slice in range(nslices):
                        outs_slice = [item[i_slice] for item in outs_i]
                        ims = [im1[i_slice]] + outs_slice + [im2[i_slice]]

                        dvfs_slice = [item[i_slice] for item in dvfs_i]
                        dvfs_slice_im = [dvf_to_im(item) for item in dvfs_slice]
                        empty_im = np.zeros_like(dvfs_slice_im[0])
                        ims_dvf = [empty_im] + dvfs_slice_im + [empty_im]

                        segs_slice = [item[i_slice] for item in segs_i]
                        segs_slice_im = [seg_to_im(item) for item in segs_slice]
                        ims_segs = [seg_to_im(im1_seg[i_slice])] + segs_slice_im + [seg_to_im(im2_seg[i_slice])]

                        img_row1 = np.concatenate(ims, axis=1)
                        img_row1 = cv2.cvtColor((img_row1*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                        img_row2 = np.concatenate(ims_dvf, axis=1)
                        img_row3 = (np.concatenate(ims_segs, axis=1)*255).astype(np.uint8)
                        alpha = 0.4
                        mask = img_row3 > 0
                        img_row3_masked = img_row1.copy()
                        img_row3_masked[mask] = alpha * img_row1[mask] + (1 - alpha) * img_row3[mask]
                        img = np.concatenate((img_row1, img_row2, img_row3_masked), axis=0)

                        cv2.imwrite(os.path.join(cache.out_dir_val, "im{}_slice{}.jpg".format(batch_no*batchsize + i, i_slice)), img)


    loss_per_sample = [np.concatenate(arr_list, axis=1) for arr_list in  loss_per_sample_list] #list of obj * samples
    mo_obj_val_sample = np.array(loss_per_sample).transpose(2,1,0)  #samples * nobj * nsol
    assert mo_obj_val_sample.ndim==3
    assert(mo_obj_val_sample.shape[1:] == (nobj, nsol))
    m_obj_val_mean = np.mean(mo_obj_val_sample, axis=0)

    if ensemble_class.__class__.__name__=="DeepEnsemble":
        for i, net in enumerate(net_list):
            net.train()
    elif ensemble_class.__class__.__name__=="KHeadEnsemble":
        ensemble_class.model.train()
    
    metrics = {"loss": mo_obj_val_sample}
    return metrics