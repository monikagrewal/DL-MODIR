import os
import cv2
import torch
import numpy as np
import matplotlib
from scipy.ndimage import median_filter
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


def calculate_percent_folding(deformation_field, spacing=(1,1,1)):
	"""
	Calculate the determinant of the Jacobian matrix for a deformation vector field.
	and then percent folding

	Parameters:
	deformation_field (numpy.ndarray): A 3D array representing the deformation vector field.
										Each element is a 3D vector representing the displacement at a point.
										
	Returns:
	determinant (numpy.ndarray): A 2D array containing the determinants of the Jacobian matrices for each point.

	det([ (1.0+dx/dx) dx/dy dx/dz ; dy/dx (1.0+dy/dy) dy/dz; dz/dx dz/dy (1.0+dz/dz) ])
	information from https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1DisplacementFieldJacobianDeterminantFilter.html

	"""
	shape = deformation_field.shape
	if len(shape) != 4 or shape[-1] != 3:
		raise ValueError("Invalid deformation field shape. Expected (Z, Y, X, 3).")

	deformation_field = deformation_field * np.array(spacing).reshape(1,1,1,3)
	deformation_grad0 = np.gradient(deformation_field[...,0], *spacing)
	deformation_grad1 = np.gradient(deformation_field[...,1], *spacing)
	deformation_grad2 = np.gradient(deformation_field[...,2], *spacing)
	deformation_grad0[0] += 1
	deformation_grad1[1] += 1
	deformation_grad2[2] += 1

	A1 = deformation_grad0[0]
	A1_cofactor = deformation_grad1[1] * deformation_grad2[2] - deformation_grad2[1] * deformation_grad1[2]
	A2 = deformation_grad0[1]
	A2_cofactor = (-1) * ( deformation_grad1[0] * deformation_grad2[2] - deformation_grad2[0] * deformation_grad1[2] )
	A3 = deformation_grad0[2]
	A3_cofactor = deformation_grad1[0] * deformation_grad2[1] - deformation_grad2[0] * deformation_grad1[1]

	determinant = A1 * A1_cofactor +\
				A2 * A2_cofactor +\
				A3 * A3_cofactor

	number_folding = (determinant < 0).sum()
	total_voxels = np.prod(shape)
	percent_folding = number_folding * 100 / float(total_voxels)
	return percent_folding


def postprocess_dvf(dvf):
	smooth_dvf = np.zeros_like(dvf)
	for i in range(dvf.shape[3]):
		smooth_dvf[..., i] = median_filter(dvf[..., i], size=(1,3,3))
	return smooth_dvf
    

def calculate_tre(transformed_pts, source_pts, spacing=(1, 1, 1)):
    transformed_pts = transformed_pts.reshape(-1, 3)
    source_pts = source_pts.reshape(-1, 3)
    spacing = np.array(spacing).reshape(1, 3)
    dist = transformed_pts*spacing - source_pts*spacing
    tre = np.linalg.norm(dist, axis=1)
    return tre


def calculate_jac_analytics(detOfJacobian):
	npixels = detOfJacobian.size
	less_than_0 = (detOfJacobian <= 0).sum()
	more_than_2 = (detOfJacobian > 2).sum()
	jac_analytics = {"npixels": float(npixels),
					"less_than_0": float(less_than_0),
					"more_than_2": float(more_than_2)}
	return jac_analytics