import logging
import torch
from torch.utils.data import Dataset
import os, glob
import SimpleITK as sitk
from problems.modir3d.sitk_utils import *


def preprocess_Empire10_data(root, output_foldername="preprocessed", output_spacing=(4, 4, 4)):
    output_path = os.path.join(root, output_foldername)
    os.makedirs(output_path, exist_ok=True)

    input_path = os.path.join(root, "scans")
    filepaths = glob.glob(input_path + "/*_Fixed.mhd")
    filepaths.sort()

    for fixed_impath in filepaths:
        print("Processing image: {}".format(fixed_impath))
        moving_impath = fixed_impath.replace("Fixed", "Moving")
        if not os.path.isfile(moving_impath):
            raise RuntimeError("corresponding moving image does not exist for: {}".format(fixed_impath))
        fixed_image = read_image(fixed_impath, output_spacing=None, crop_depth=False, rescaling=False)
        moving_image = read_image(moving_impath, output_spacing=None, crop_depth=False, rescaling=False)
        moving_image_aligned, status = affine_registration(fixed_image, moving_image)
        print("Fixed image size: {}, Moving image size: {} --> {}".format(fixed_image.GetSize(),\
                         moving_image.GetSize(), moving_image_aligned.GetSize()))

        img1 = sitk.GetArrayFromImage(fixed_image)
        print("fixed image: ", img1.max(), img1.min(), img1.mean())
        img2 = sitk.GetArrayFromImage(moving_image)
        print("moving image: ", img2.max(), img2.min(), img2.mean())
        img3 = sitk.GetArrayFromImage(moving_image_aligned)
        print("moving image aligned: ", img3.max(), img3.min(), img3.mean())

        # resample voxel spacing and rescale intensity and save
        fixed_image = resample_voxel_spacing(fixed_image, output_spacing=output_spacing)
        fixed_image = rescale_intensity(fixed_image)
        _, fixed_filename = os.path.split(fixed_impath)
        save_sitk_image(fixed_image, os.path.join(output_path, fixed_filename))

        moving_image_aligned = resample_voxel_spacing(moving_image_aligned, output_spacing=output_spacing)
        moving_image_aligned = rescale_intensity(moving_image_aligned)
        _, moving_filename = os.path.split(moving_impath)
        save_sitk_image(moving_image_aligned, os.path.join(output_path, moving_filename))

     
class Empire10(Dataset):
    """Dataset class defining dataloader for Empire10"""

    def __init__(self, root="./", n_samples=100, train=True):
        super(Empire10, self).__init__()
        """
        Args:- root = input path
        n_samples = len(data) or smaller if trial run
        """
        self.root = os.path.join(root, "preprocessed")
        self.data = glob.glob(self.root + "/*_Fixed.mhd")
        self.data.sort()
        self.data = self.data[:n_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        impath = self.data[index]
        # print(impath)
        img1, img2 = self.load_images(impath)
        data = {"X": (img1, img2), "Y": img1}
        return data

    def load_images(self, impath):
        # check if moving image exists
        moving_impath = impath.replace("Fixed", "Moving")
        if not os.path.isfile(moving_impath):
            raise RuntimeError("corresponding moving image does not exist for: {}".format(impath))
        img1 = read_image(impath, output_spacing=None, windowing=False, rescaling=False, crop_depth=True, max_depth=64)
        img1 = sitk.GetArrayFromImage(img1)  #trans, cor, sag

        img2 = read_image(moving_impath, output_spacing=None, windowing=False, rescaling=False, crop_depth=True, max_depth=64)
        img2 = sitk.GetArrayFromImage(img2)  #trans, cor, sag
        
        # print("fixed image: ", img1.max(), img1.min(), img1.mean())
        # print("moving image: ", img2.max(), img2.min(), img2.mean())    

        # to tensor
        d, h, w = img1.shape
        img1 = torch.from_numpy(img1).float().view(1, d, h, w)
        img2 = torch.from_numpy(img2).float().view(1, d, h, w)
        return img1, img2

    def partition(self, indices):
        """
        slice the data and targets for a given list of indices
        """
        self.indices = indices
        self.data = [self.data[i] for i in indices]
        return self