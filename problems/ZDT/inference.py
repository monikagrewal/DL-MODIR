import os
import cv2
import torch
import numpy as np
import matplotlib
import logging


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

    loss_per_sample_list = [[] for i in range(nsol)]

    for batch_no, data in enumerate(validation_dataloader):
        if batch_no==10:
            break
        inputs = data["X"]
        targets = data["Y"]

        outs = []
        if ensemble_class.__class__.__name__=="DeepEnsemble":
            for i_mo_sol in range(0, nsol):
                with torch.no_grad():
                    out = net_list[i_mo_sol](inputs)
                loss_per_sample = criterion(out, targets)
                loss_per_sample = torch.stack(loss_per_sample, dim=0)
                loss_per_sample_list[i_mo_sol].append(loss_per_sample.data.cpu().numpy())
                outs.append(out[0].data.cpu().numpy())
        elif ensemble_class.__class__.__name__=="KHeadEnsemble":
            with torch.no_grad():
                outs_torch = ensemble_class.model(inputs)
            for i_mo_sol in range(0, nsol):
                out = outs_torch[i_mo_sol]
                loss_per_sample = criterion(out, targets)
                loss_per_sample = torch.stack(loss_per_sample, dim=0)
                loss_per_sample_list[i_mo_sol].append(loss_per_sample.data.cpu().numpy())
                outs.append(out[0].data.cpu().numpy())

        if visualize:
            # sorting outs for visualization
            pass


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