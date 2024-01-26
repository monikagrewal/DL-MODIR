import os
import cv2
import torch
import numpy as np
import matplotlib
from scipy import signal
import logging

from problems.modir3d.utils import *
from problems.modir3d.models.vm_layers import transform_points


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
    elif ensemble_class.__class__.__name__=="KHeadEnsemble":
        ensemble_class.model.eval()
    
    use_segmentation = getattr(validation_dataloader.dataset, "use_segmentation", False)

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
                if nsol>15:
                    sort_indices = [np.random.choice(indices, 15) for indices in sort_indices]

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
            if batch_no==5:
                break
            inputs = data["X"]
            targets = data["Y"]
            img1, img2 = inputs[0].numpy(), inputs[1].numpy()
            img1_seg = torch.argmax(targets[1], dim=1).float().numpy()
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
                    seg = torch.argmax(out[2], dim=1).float()
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
                    seg = torch.argmax(out[2], dim=1).float()
                    segs.append(seg.data.cpu().numpy())

            if visualize:
                # sorting outs for visualization
                losses = [item[-1] for item in loss_per_sample_list]
                losses = np.array(losses) #nsol * nobj * nsample
                sort_indices = [np.argsort(losses[:, 0, i]) for i in range(losses.shape[2])]
                if nsol>15:
                    sort_indices = [np.random.choice(indices, 15) for indices in sort_indices]

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
                        segs_slice_im = [seg_to_im(item, text=str(sort_indices[i][idx])) for idx, item in enumerate(segs_slice)]
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


def full_im_inference(model, inputs, pts=None):
    """
    sliding window inference on full image
    """
    if pts is not None:
        pts = pts.float().to(model.device)
    inputs = [item.to(model.device) for item in inputs]
    target_im = inputs[0]

    min_depth = 8
    image_depth = 32
    slice_weighting = False
    batchsize = target_im.shape[0]
    nslices = target_im.shape[2]
    flow_fields = [torch.zeros(
        batchsize, 3, *target_im.shape[2:]).to(model.device)
        for _ in range(model.K)]
    slice_overlaps = torch.zeros(1, 1, nslices, 1, 1).to(model.device)
    start = 0
    while start + min_depth <= nslices:
        if start + image_depth >= nslices:
            indices = slice(nslices - image_depth, nslices)
            start = nslices
        else:
            indices = slice(start, start + image_depth)
            start += image_depth//3

        mini_inputs = [item[:, :, indices, :, :] for item in inputs]
        mini_flows = model.unet_model(*mini_inputs)

        if slice_weighting:
            actual_slices = mini_inputs[0].shape[2]
            weights = signal.gaussian(actual_slices, std=actual_slices / 6)
            weights = torch.tensor(weights, dtype=torch.float32, device=model.device)
            slice_overlaps[0, 0, indices, 0, 0] += weights

            for i, flow in enumerate(mini_flows):
                flow_fields[i][:, :, indices, :, :] += flow * weights.view(1, 1, actual_slices, 1, 1)
      
        else:
            slice_overlaps[0, 0, indices, 0, 0] += 1
            for i, flow in enumerate(mini_flows):
                if i==len(flow_fields)-1:
                    print("flow ", flow.max(), flow.min())
                flow_fields[i][:, :, indices, :, :] += flow

    print("flow after sum", flow_fields[-1].max(), flow_fields[-1].min())
    flow_fields = [output / slice_overlaps for output in flow_fields]
    print("slice overlaps ", slice_overlaps.max(), slice_overlaps.min())
    print("flow after divide", flow_fields[-1].max(), flow_fields[-1].min())

    outputs_list = []
    for flow in flow_fields:
        y_source_transformed = model.transformer(inputs[1], flow)
        outputs = [y_source_transformed, flow]
        
        if len(inputs)==4:  #seg masks provided
            source_seg_warped = model.transformer(inputs[3], flow)   
            outputs += [source_seg_warped]
        
        # transform pts if present
        if pts is not None:
            outputs.append(transform_points(pts, flow))

        outputs_list.append(outputs)
    
    if model.K==1:
        return outputs_list[0]
    else:
        return outputs_list


def testing(ensemble_class, dataloader, criterion, cache, visualize=True, save=True):
    """
    runs on test data

    dataloader gives the following data:
    data["X"] : target_image, source_image, target_seg, source_seg, target_pts, source_pts
    data["Y"] : target_image, target_seg
    """
    nsol = ensemble_class.n_mo_sol
    nobj = ensemble_class.n_mo_obj

    # eval mode on in each network
    if ensemble_class.__class__.__name__=="DeepEnsemble":
        net_list = ensemble_class.net_list
        for i, net in enumerate(net_list):
            net.eval()
    elif ensemble_class.__class__.__name__=="KHeadEnsemble":
        ensemble_class.model.eval()
    
    use_segmentation = getattr(dataloader.dataset, "use_segmentation", False)

    loss_per_sample_list = [[] for i in range(nsol)]
    sol_sort_indices_list = []
    
    if not use_segmentation:
        raise NotImplementedError("testing not implemented for use_segmentation=False.")
    else:
        for batch_no, data in enumerate(dataloader):
            logging.info(f"testing batch: {batch_no}")
            # if batch_no==2:
            #     break
            inputs = data["X"]
            targets = data["Y"]
            pts = data["pts"]
            img1, img2 = inputs[0].numpy(), inputs[1].numpy()
            img1_seg = torch.argmax(inputs[2], dim=1).float().numpy()
            img2_seg = torch.argmax(inputs[3], dim=1).float().numpy()
            img1_pts = pts[0].numpy()
            img2_pts = pts[1].numpy()
            outs = []
            dvfs = []
            segs = []
            pts_outs = []
            if ensemble_class.__class__.__name__=="DeepEnsemble":
                for i_mo_sol in range(0, nsol):
                    with torch.no_grad():
                        out = full_im_inference(ensemble_class.net_list[i_mo_sol], inputs, pts=pts[0])
                    loss_per_sample = criterion(out, targets)
                    loss_per_sample = torch.stack(loss_per_sample, dim=0)
                    loss_per_sample_list[i_mo_sol].append(loss_per_sample.data.cpu().numpy())
                    outs.append(out[0].data.cpu().numpy())
                    dvf = out[1].permute(0, 2, 3, 4, 1)
                    dvfs.append(dvf.data.cpu().numpy())
                    seg = torch.argmax(out[2], dim=1).float()
                    segs.append(seg.data.cpu().numpy())
                    pts_outs.append(out[3].data.cpu().numpy())
            elif ensemble_class.__class__.__name__=="KHeadEnsemble":
                with torch.no_grad():
                    outs_torch = full_im_inference(ensemble_class.model, inputs, pts=pts[0])
                for i_mo_sol in range(0, nsol):
                    out = outs_torch[i_mo_sol]
                    loss_per_sample = criterion(out, targets)
                    loss_per_sample = torch.stack(loss_per_sample, dim=0)
                    loss_per_sample_list[i_mo_sol].append(loss_per_sample.data.cpu().numpy())
                    outs.append(out[0].data.cpu().numpy())
                    dvf = out[1].permute(0, 2, 3, 4, 1)
                    dvfs.append(dvf.data.cpu().numpy())
                    seg = torch.argmax(out[2], dim=1).float()
                    segs.append(seg.data.cpu().numpy())
                    pts_outs.append(out[3].data.cpu().numpy())

            # sorting outs for visualization
            losses = [item[-1] for item in loss_per_sample_list]
            losses = np.array(losses) #nsol * nobj * nsample
            sort_indices = [np.argsort(losses[:, 0, i]) for i in range(losses.shape[2])]
            sol_sort_indices_list.append(np.array(sort_indices))

            if save:
                # calculate percent folding
                percent_foldings = [calculate_percent_folding(dvf[0]) for dvf in dvfs]
                tre_before = calculate_tre(img1_pts, img2_pts)
                tre_sols = [calculate_tre(pts_outs[idx], img2_pts) for idx in range(nsol)]

                filepath = os.path.join(cache.out_dir_test, "im{}.npz".format(batch_no))
                if batch_no < 1:
                    data = {"target_image": img1,
                            "source_image": img2,
                            "target_seg": img1_seg,
                            "source_seg": img2_seg,
                            "target_pts": img1_pts,
                            "source_pts": img2_pts,
                            "dvfs": dvfs,
                            "transformed_source_images": outs,
                            "transformed_source_seg": segs,
                            "transformed_target_pts": pts_outs,
                            "sort_indices": sort_indices,
                            "losses": losses,
                            "tre_before": tre_before,
                            "tre_sols": tre_sols,
                            "percent_foldings": percent_foldings
                    }
                else:
                    data = {"target_pts": img1_pts,
                            "source_pts": img2_pts,
                            "transformed_target_pts": pts_outs,
                            "losses": losses,
                            "tre_before": tre_before,
                            "tre_sols": tre_sols,
                            "percent_foldings": percent_foldings
                    }

                np.savez(filepath, **data)


            if visualize and batch_no < 10:
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
                        segs_slice_im = [seg_to_im(item, text=str(sort_indices[i][idx])) for idx, item in enumerate(segs_slice)]
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

                        cv2.imwrite(os.path.join(cache.out_dir_test, "im{}_slice{}.jpg".format(batch_no*batchsize + i, i_slice)), img)


    loss_per_sample = [np.concatenate(arr_list, axis=1) for arr_list in  loss_per_sample_list] #list of obj * samples
    mo_obj_val_sample = np.array(loss_per_sample).transpose(2,1,0)  #samples * nobj * nsol
    assert mo_obj_val_sample.ndim==3
    # assert(mo_obj_val_sample.shape[1:] == (nobj, nsol))
    m_obj_val_mean = np.mean(mo_obj_val_sample, axis=0)
    sol_sort_indices = np.concatenate(sol_sort_indices_list, axis=0) #samples * nsol
    
    metrics = {"loss": mo_obj_val_sample,
               "sol_sort_indices": sol_sort_indices}
    return metrics