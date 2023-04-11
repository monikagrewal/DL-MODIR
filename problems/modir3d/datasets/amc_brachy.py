import logging
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os, glob
import numpy as np, cv2
import pandas as pd
import json
import SimpleITK as sitk
from scipy.ndimage import zoom
import skimage

from problems.modir3d.sitk_utils import *
# from sitk_utils import *


def generate_mask(annotations, sorted_metadata_list, target_image, 
                  classes=['background', 'bowel', 'bladder', 'rectum', 'sigmoid']):
    target_shape = target_image.GetSize()[:2]       
    
    # order in which to layer class annotations in case they overlap
    class_layering = ['background', 'bowel', 'bladder', 'rectum', 'sigmoid']
    class2idx = dict(zip(classes, range(len(classes))))    
    class2layeridx = dict(zip(class_layering, range(len(classes))))
    class_layer_indici = np.array([class2layeridx[class_name] for class_name in classes])

    uid_to_slice_idx = dict([(meta['uid'], i) for i, meta in enumerate(sorted_metadata_list)])    
    mask_volume = np.zeros((len(sorted_metadata_list), target_shape[0], target_shape[1]), dtype=np.int32)  
    for item in sorted(annotations, key=lambda x: uid_to_slice_idx.get(x['uid'], -1)):        
        uid = item["uid"]
        slice_idx = uid_to_slice_idx.get(uid, None)
        if slice_idx is None:
            continue

        coords_pix = np.array(item["coords"])
        label = item["label_name"]
        label_idx = class2idx.get(label)
        if label_idx is None:
            continue
            
        rr, cc = skimage.draw.polygon(coords_pix[:,0], coords_pix[:,1], shape=(target_shape[0], target_shape[1]))

        # determine whether to overwrite existing annotation label based on predefined ordering 
        # (for example bladder takes precedence bowel bag)
        overwrite_mask = class_layer_indici[mask_volume[slice_idx, cc,rr]] < class_layer_indici[label_idx]        
        rr, cc = rr[overwrite_mask], cc[overwrite_mask]
        mask_volume[slice_idx, cc, rr] = label_idx

    # generate sitk image
    mask_image = sitk.GetImageFromArray(mask_volume)
    mask_image.SetOrigin(target_image.GetOrigin())
    mask_image.SetSpacing(target_image.GetSpacing())
    mask_image.SetDirection(target_image.GetDirection())
    
    return mask_image

def arr_resample_voxel_spacing(arr, original_spacing, required_spacing, order=1):
    zoom_factor = original_spacing / required_spacing
    arr = zoom(arr, zoom_factor, order=order)
    return arr


def preprocess_modir_data(root, csv_path, output_foldername="preprocessed", output_spacing=(1, 1, 3)):
    output_path = os.path.join(root, output_foldername)
    os.makedirs(output_path, exist_ok=True)

    info = pd.read_csv(csv_path)
    info = info[:100]  #take only 100 in the beginning

    filepaths_fixed = info["fixed"]
    filepaths_moving = info["moving"]

    for i, (fixed_infopath, moving_infopath) in enumerate(zip(filepaths_fixed, filepaths_moving)):
        print("Processing file: {}".format(fixed_infopath))

        info = json.load(open(fixed_infopath, "r"))
        fixed_meta = info["meta"]
        fixed_annotations = info["annotation"]
        fixed_meta = sorted(fixed_meta, key=lambda x: x["SliceLocation"])
        fixed_impaths = [item["original_path"] for item in fixed_meta]
        
        info = json.load(open(moving_infopath, "r"))
        moving_meta = info["meta"]
        moving_annotations = info["annotation"]
        moving_meta = sorted(moving_meta, key=lambda x: x["SliceLocation"])
        moving_impaths = [item["original_path"] for item in moving_meta]

        fixed_image = read_image(fixed_impaths, output_spacing=None, crop_depth=False, rescaling=True)
        moving_image = read_image(moving_impaths, output_spacing=None, crop_depth=False, rescaling=True)
        print(f"Spacings: fixed image = {fixed_image.GetSpacing()}, moving image = {moving_image.GetSpacing()}")

        # load annotations and generate mask
        fixed_label = generate_mask(fixed_annotations, fixed_meta, fixed_image)
        moving_label = generate_mask(moving_annotations, moving_meta, moving_image)

        # resample voxel spacing
        fixed_image = resample_voxel_spacing(fixed_image, output_spacing=output_spacing, output_size=[192, 192, 0])
        moving_image = resample_voxel_spacing(moving_image, output_spacing=output_spacing, output_size=[192, 192, 0])
        fixed_label = resample_voxel_spacing(fixed_label, 
                                             output_spacing=output_spacing,
                                             output_size=[192, 192, 0],
                                             interpolator='nearest')
        moving_label = resample_voxel_spacing(moving_label,
                                              output_spacing=output_spacing,
                                              output_size=[192, 192, 0],
                                              interpolator='nearest')

        try:
            moving_image_aligned, status, outTx = rigid_registration(fixed_image, moving_image)
            print("Sizes: Fixed image: {}, Moving image: {} --> {}".format(fixed_image.GetSize(),\
                                moving_image.GetSize(), moving_image_aligned.GetSize()))
            moving_label_aligned = resample_image(moving_label, fixed_image, outTx, interpolator='nearest')
        except Exception as e:
            logging.warning(e)
            status = False
            continue
        status = True

        if status:
            img1 = sitk.GetArrayFromImage(fixed_image)
            img2 = sitk.GetArrayFromImage(moving_image_aligned)
            fixed_label = sitk.GetArrayFromImage(fixed_label)
            print("fixed label: ", np.unique(fixed_label), fixed_label.shape)
            moving_label = sitk.GetArrayFromImage(moving_label_aligned)
            print("moving label aligned: ", np.unique(moving_label), moving_label.shape)

            # flag images that need rechecking
            size_diff = np.array(img1.shape[1:]) - np.array(img2.shape[1:])
            if size_diff.any():
                print(f"in-plane sizes are different: {img1.shape} and {img2.shape}. Recheck.")
                continue

            # save
            filename = "{0:03d}_Fixed".format(i)
            np.save(os.path.join(output_path, filename), img1)
            filename = "{0:03d}_Fixed_label".format(i)
            np.save(os.path.join(output_path, filename), fixed_label)

            filename = "{0:03d}_Moving".format(i)
            np.save(os.path.join(output_path, filename), img2)
            filename = "{0:03d}_Moving_label".format(i)
            np.save(os.path.join(output_path, filename), moving_label)


class AMCBrachy():
    """
    Brachytherapy MRI data
    """
    def __init__(self, root, use_segmentation=True, max_depth=32, num_classes=5):
        self.num_classes = num_classes
        self.root = os.path.join(root, "preprocessed")
        self.data = glob.glob(self.root + "/*_Fixed.npy")
        self.data.sort()

        self.use_segmentation = use_segmentation
        self.max_depth = max_depth

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        impath = self.data[index]
        logging.debug(f"impath: {impath}")
        outs = self.load_images(impath)
        data = {"X": outs, "Y": outs[0]}
        return data

    def load_images(self, impath):
        # check if moving image exists
        moving_impath = impath.replace("Fixed", "Moving")
        if not os.path.isfile(moving_impath):
            raise RuntimeError("corresponding moving image does not exist for: {}".format(impath))
        img1 = np.load(impath) #trans, cor, sag
        img2 = np.load(moving_impath)
        logging.debug(f"image sizes: {img1.shape}, {img2.shape}")

        # to tensor
        d1, h1, w1 = img1.shape
        img1 = torch.from_numpy(img1).float().view(1, d1, h1, w1)
        d2, h2, w2 = img2.shape
        img2 = torch.from_numpy(img2).float().view(1, d2, h2, w2)
        
        if self.use_segmentation:
            fixed_seg = np.load(impath.replace("Fixed", "Fixed_label"))
            moving_seg = np.load(impath.replace("Fixed", "Moving_label"))
            # to tensor
            fixed_seg = torch.from_numpy(fixed_seg).long()
            moving_seg = torch.from_numpy(moving_seg).long()
            # to onehot
            y = torch.eye(self.num_classes, dtype=torch.float32)
            fixed_seg_onehot = y[fixed_seg].permute(3, 0, 1, 2)
            moving_seg_onehot = y[moving_seg].permute(3, 0, 1, 2)

            outputs = (img1, img2, fixed_seg_onehot, moving_seg_onehot)
        else:
            outputs = (img1, img2)

        # crop depth
        if self.max_depth < 100:
            start_idx = np.random.choice(list(range(0, d1 - self.max_depth)), 
                                         1)[0]
            end_idx = start_idx + self.max_depth
            outputs = [item[:, start_idx:end_idx, :, :] for item in outputs]
 
        return outputs

    def partition(self, indices):
        """
        slice the data and targets for a given list of indices
        """
        self.indices = indices
        self.data = [self.data[i] for i in indices]
        return self      