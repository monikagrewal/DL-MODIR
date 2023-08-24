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


def seg_to_im(seg:np.array, num_classes:int=5, text:str="") -> np.array:
    """
    input: 2D segmentation mask with integer value for each class
    """
    class_to_color = {1:(0,0,1), 2:(0,1,0), 3:(0,1,1), 4:(1,0,0)}
    seg_im = np.zeros(seg.shape, dtype=np.uint8)
    seg_im = cv2.cvtColor(seg_im, cv2.COLOR_GRAY2BGR)
    for class_idx in range(1, num_classes):
        rr, cc = np.where(seg==class_idx)
        seg_im[rr, cc, :] = class_to_color[class_idx]
    seg_im = cv2.putText(seg_im, text, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (1,1,1))
    return seg_im


def convert_points_to_image(samp_pts, d, H, W):
    """
    Inputs:-
    samp_pts: b, 1, 1, k, 3
    """

    b, _, _, K, _ = samp_pts.shape
    # Convert pytorch -> numpy.
    samp_pts = samp_pts.data.cpu().numpy().reshape(b, K, 3)
    samp_pts = (samp_pts + 1.) / 2.
    samp_pts = np.round(samp_pts * np.array([float(W-1), float(H-1), float(d-1)]).reshape(1, 1, 3), 0)
    return samp_pts.astype(np.int32)


def convert_points_to_torch(pts, d, H, W, device="cuda:0"):
    """
    Inputs:-
    pts: k, 3 (W, H, d)
    """

    samp_pts = torch.from_numpy(pts.astype(np.float32))
    samp_pts[:, 0] = (samp_pts[:, 0] * 2. / (W-1)) - 1.
    samp_pts[:, 1] = (samp_pts[:, 1] * 2. / (H-1)) - 1.
    samp_pts[:, 2] = (samp_pts[:, 2] * 2. / (d-1)) - 1.
    samp_pts = samp_pts.view(1, 1, 1, -1, 3)
    samp_pts = samp_pts.float().to(device)
    return samp_pts


def calculate_determinant_of_jacobian(self, deformation_field:np.array):
    """
    Calculate the determinant of the Jacobian matrix for a deformation vector field.
    
    Parameters:
    deformation_field (numpy.ndarray): A 3D array representing the deformation vector field.
                                    Each element is a 3D vector representing the displacement at a point.
                                    
    Returns:
    determinant (numpy.ndarray): A 3D array containing the determinants of the Jacobian matrices for each point.
    """
    shape = deformation_field.shape
    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError("Invalid deformation field shape. Expected (X, Y, Z, 3).")
    
    deformation_grad = np.gradient(deformation_field, axis=(0, 1, 2))
    
    determinant = np.zeros((shape[0], shape[1], shape[2]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                jac_matrix = np.array([[deformation_grad[0][i,j,k,0], deformation_grad[0][i,j,k,1], deformation_grad[0][i,j,k,2]],
                                    [deformation_grad[1][i,j,k,0], deformation_grad[1][i,j,k,1], deformation_grad[1][i,j,k,2]],
                                    [deformation_grad[2][i,j,k,0], deformation_grad[2][i,j,k,1], deformation_grad[2][i,j,k,2]]])
                determinant[i, j, k] = np.linalg.det(jac_matrix)
    
    return determinant
    

def calculate_tre(self, points_in_target, points_in_source, deformation_field:np.array):
    #TODO: change to torch
    """
    Calculate the TRE
    
    Parameters:
    deformation_field (numpy.ndarray): TODO
                                    
    Returns:
    determinant (numpy.ndarray): TODO
    """
    shape = deformation_field.shape
    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError("Invalid deformation field shape. Expected (X, Y, Z, 3).")
    
    pts2_torch = convert_points_to_torch(np.array(pts2_pred), *shape, device=device)   #1, 1, 1, k, 3

    deformation_torch = torch.from_numpy(deformation).to(device).permute(0, 4, 1, 2, 3)  #b, 3, d, h, w
    pts1_actual = F.grid_sample(deformation_torch, pts2_torch) #b, 3, 1, 1, k
    pts1_actual = pts1_actual.permute(0, 2, 3, 4, 1) #b, 1, 1, k, 3
    pts1_actual = convert_points_to_image(pts1_actual, *shape)
    pts1_actual = pts1_actual.reshape(-1, 3)
    
    return tre


def calculate_jac_analytics(detOfJacobian):
	npixels = detOfJacobian.size
	less_than_0 = (detOfJacobian <= 0).sum()
	more_than_2 = (detOfJacobian > 2).sum()
	jac_analytics = {"npixels": float(npixels),
					"less_than_0": float(less_than_0),
					"more_than_2": float(more_than_2)}
	return jac_analytics


def target_registration_errors(pts1, pts2, spacing1=(1, 1, 1), spacing2=(1, 1, 1)):
	pts1_mm = np.array(pts1) * np.array(spacing1).reshape(1, -1)
	pts2_mm = np.array(pts2) * np.array(spacing2).reshape(1, -1)
	errors = np.linalg.norm(pts1_mm -  pts2_mm, axis=1)
	return errors.tolist()