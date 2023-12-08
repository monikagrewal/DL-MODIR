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
        label = item["label_name"]
        if label in classes:
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


def generate_pts_list(annotations, sorted_metadata_list):
    pts_names = ['int_urethral_os', 'ext_urethral_os', 'uterus_top', 'cervix_os',
                 'isthmus', 'IU_canal_top', 'ureteral_os_right', 'ureteral_os_left',
                 'int_anal_sfinct', 'coccygis', 'S1S2', 'S2S3', 'S3S4', 'ASBS',
                 'PIBS_os', 'FH_right', 'FH_left', 'AC_left', 'AC_right', 'rotundum_left',
                 'a_uterina_right', 'a_uterina_left', 'rotundum_right']
    
    uid_to_slice_idx = dict([(meta['uid'], i) for i, meta in enumerate(sorted_metadata_list)])    
    
    all_pts = []
    all_pts_names = []
    for pts_name in pts_names:
        item = [item for item in annotations if item["label_name"].lower()==pts_name.lower()]
        if len(item)==1:
            uid = item[0]["uid"]
            slice_idx = uid_to_slice_idx.get(uid, None)
            if slice_idx is None:
                logging.warning("This shouldn't happen. \
                                If annotation is there, it should correspond to a slice number.")
                continue

            coords_pix = item[0]["coords"]
            coords_pix = coords_pix[0] + [slice_idx]
            label = item[0]["label_name"].lower()

            all_pts.append(coords_pix)
            all_pts_names.append(label)

    return all_pts, all_pts_names


def arr_resample_voxel_spacing(arr, original_spacing, required_spacing, order=1):
    zoom_factor = original_spacing / required_spacing
    arr = zoom(arr, zoom_factor, order=order)
    return arr


def preprocess_modir_data(root, csv_path, output_foldername="preprocessed", output_spacing=(1, 1, 3), output_size="image", load_pts=False):
    output_path = os.path.join(root, output_foldername)
    os.makedirs(output_path, exist_ok=True)

    info = pd.read_csv(csv_path)
    # info = info[:100]  #take only 100 in the beginning

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
        print(f"Original spacings: fixed image = {fixed_image.GetSpacing()}, moving image = {moving_image.GetSpacing()}")
        print(f"Original sizes: fixed image = {fixed_image.GetSize()}, moving image = {moving_image.GetSize()}")


        if load_pts:
            # generate pts_list from annotations and convert to mm
            fixed_pts, fixed_pts_names = generate_pts_list(fixed_annotations, fixed_meta)
            moving_pts, moving_pts_names = generate_pts_list(moving_annotations, moving_meta)

            # make sure both images have same number of landmarks
            common_names = list(set(fixed_pts_names) & set(moving_pts_names))
            if len(common_names) != len(fixed_pts_names) or \
                len(common_names) != len(moving_pts_names):
                fixed_pts = [pts for i, pts in enumerate(fixed_pts) if fixed_pts_names[i] in common_names]
                moving_pts = [pts for i, pts in enumerate(moving_pts) if moving_pts_names[i] in common_names]
                fixed_pts_names = [name for name in fixed_pts_names if name in common_names]
                moving_pts_names = [name for name in moving_pts_names if name in common_names]

            fixed_pts_physical = [fixed_image.TransformContinuousIndexToPhysicalPoint(p) for p in fixed_pts]
            moving_pts_physical = [moving_image.TransformContinuousIndexToPhysicalPoint(p) for p in moving_pts]

        # load annotations and generate mask
        fixed_label = generate_mask(fixed_annotations, fixed_meta, fixed_image)
        moving_label = generate_mask(moving_annotations, moving_meta, moving_image)

        # resample voxel spacing
        fixed_image = resample_voxel_spacing(fixed_image, output_spacing=output_spacing, output_size=output_size)
        moving_image = resample_voxel_spacing(moving_image, output_spacing=output_spacing, output_size=output_size)
        fixed_label = resample_voxel_spacing(fixed_label, 
                                             output_spacing=output_spacing,
                                             output_size=output_size,
                                             interpolator='nearest')
        moving_label = resample_voxel_spacing(moving_label,
                                              output_spacing=output_spacing,
                                              output_size=output_size,
                                              interpolator='nearest')
        
        if load_pts:
            # convert pts to new voxel spacing
            fixed_pts = [fixed_image.TransformPhysicalPointToContinuousIndex(p) for p in fixed_pts_physical]
            moving_pts = [moving_image.TransformPhysicalPointToContinuousIndex(p) for p in moving_pts_physical]

        print(f"Resampled spacings: fixed image = {fixed_image.GetSpacing()}, moving image = {moving_image.GetSpacing()}")
        print(f"Resampled sizes: fixed image = {fixed_image.GetSize()}, moving image = {moving_image.GetSize()}")

        status = True
        try:
            moving_image_aligned, status, outTx = rigid_registration(fixed_image, moving_image)
        except Exception as e:
            logging.warning(e)
            if "The images do not sufficiently overlap" in str(e):
                if load_pts and len(fixed_pts)>0:
                    fixed_pts_physical = [fixed_image.TransformContinuousIndexToPhysicalPoint(p) for p in fixed_pts]
                    moving_pts_physical = [moving_image.TransformContinuousIndexToPhysicalPoint(p) for p in moving_pts]
                    fixed_pts_flat = [c for p in fixed_pts_physical for c in p]        
                    moving_pts_flat = [c for p in moving_pts_physical for c in p]
                    initial_transform = sitk.LandmarkBasedTransformInitializer(sitk.VersorRigid3DTransform(), 
                                                                                    fixed_pts_flat, 
                                                                                    moving_pts_flat)
                    moving_image_aligned, status, outTx = rigid_registration(fixed_image, moving_image, initialTx=initial_transform)
            else:
                import pdb; pdb.set_trace()
                status = False
                continue

        if status:
            print("Sizes: Fixed image: {}, Moving image: {} --> {}".format(fixed_image.GetSize(),\
                                moving_image.GetSize(), moving_image_aligned.GetSize()))
            moving_label_aligned = resample_image(moving_label, fixed_image, outTx, interpolator='nearest')
            
            if load_pts and len(fixed_pts)>0:
                inverseTx = outTx.GetInverse()
                moving_pts_physical = [moving_image.TransformContinuousIndexToPhysicalPoint(p) for p in moving_pts]
                moving_pts_physical_transformed = [inverseTx.TransformPoint(p) for p in moving_pts_physical]
                moving_pts_transformed = [fixed_image.TransformPhysicalPointToContinuousIndex(p) for p in \
                                         moving_pts_physical_transformed]
            else:
                moving_pts_transformed = []

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
            if load_pts:
                filename = "{0:03d}_Fixed_points.json".format(i)
                obj = {"pts": fixed_pts, "names": fixed_pts_names}
                json.dump(obj, open(os.path.join(output_path, filename), "w"))
                filename = "{0:03d}_Moving_points.json".format(i)
                obj = {"pts": moving_pts_transformed, "names": moving_pts_names}
                json.dump(obj, open(os.path.join(output_path, filename), "w"))


def embed_seg(x: torch.Tensor, xs: torch.Tensor):
    """
    x = input image, 1 * d * h * w
    xs = input image one-hot encoded seg mask, seg_classes * d * h * w
    """
    seg_classes = xs.shape[0]
    for i in range(1, seg_classes):  # remove background
        mask = xs[i, :, :, :] == 1
        mean_intensity = x[0][mask].mean()
        x[0][mask] = mean_intensity
    
    return x

class AMCBrachy():
    """
    Brachytherapy MRI data
    """
    def __init__(self, root, train=True, use_segmentation=True, max_depth=32, num_classes=5, inplane_size=192, 
                 classes_to_include=[0,1,2,3,4], load_pts=False, processed_foldername="preprocessed"):
        self.root = os.path.join(root, processed_foldername)
        self.data = glob.glob(self.root + "/*_Fixed.npy")
        self.data.sort()

        # -----------------------------------------------------------
        # remove image 88: rectum segmentation missing in some slices
        #         and 107: bladder segmentation missing in some slices
        # For details, check data_preparation/meta/LUMC_cervical_train_pairs_annotation.csv
        # -----------------------------------------------------------
        if train:
            filename_to_remove = glob.glob(self.root + "/088_Fixed.npy") +\
                                glob.glob(self.root + "/107_Fixed.npy*")
            for filename in filename_to_remove:
                self.data.remove(filename)

        self.max_depth = max_depth
        self.inplane_size = inplane_size
        self.use_segmentation = use_segmentation
        self.num_classes = num_classes
        self.classes_to_include = classes_to_include
        self.load_pts = load_pts
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        impath = self.data[index]
        logging.debug(f"impath: {impath}")
        outs = self.load_images(impath)
        if len(outs)==2:
            data = {"X": [outs[0], outs[1]], 
                    "Y": outs[0]
                    }
        elif self.use_segmentation and not self.load_pts:
            data = {"X": [outs[0], outs[1], outs[2], outs[3]], 
                    "Y": [outs[0], outs[2]]
                    }
        elif self.use_segmentation and self.load_pts:
            data = {"X": [outs[0], outs[1], outs[2], outs[3]], 
                    "Y": [outs[0], outs[2]],
                    "pts": [outs[4], outs[5]]
                    }
        elif not self.use_segmentation and self.load_pts:
            raise ValueError("Please don't do this.")
        return data

    def load_images(self, impath):
        # check if moving image exists
        moving_impath = impath.replace("Fixed", "Moving")
        if not os.path.isfile(moving_impath):
            raise RuntimeError("corresponding moving image does not exist for: {}".format(impath))
        img1 = np.load(impath) #trans, cor, sag
        img2 = np.load(moving_impath)
        logging.debug(f"image sizes: {img1.shape}, {img2.shape}")

        # rescale intensity to 0 to 1
        img1 = img1 / img1.max()
        img2 = img2 / img2.max()

        # to tensor
        d1, h1, w1 = img1.shape
        img1 = torch.from_numpy(img1).float().view(1, d1, h1, w1)
        d2, h2, w2 = img2.shape
        img2 = torch.from_numpy(img2).float().view(1, d2, h2, w2)
        
        outputs = [img1, img2]
        if self.use_segmentation:
            fixed_seg = np.load(impath.replace("Fixed", "Fixed_label"))
            moving_seg = np.load(impath.replace("Fixed", "Moving_label"))
            # to tensor
            fixed_seg = torch.from_numpy(fixed_seg).long()
            moving_seg = torch.from_numpy(moving_seg).long()
            # to onehot
            y = torch.eye(self.num_classes, dtype=torch.float32)
            fixed_seg_onehot = y[fixed_seg].permute(3, 0, 1, 2)[self.classes_to_include]
            moving_seg_onehot = y[moving_seg].permute(3, 0, 1, 2)[self.classes_to_include]

            outputs += [fixed_seg_onehot, moving_seg_onehot]
        
        # crop depth
        start_idx_depth = 0
        if self.max_depth < 100:
            if self.train:
                start_idx_depth = np.random.choice(list(range(0, d1 - self.max_depth)), 
                                         1)[0]
            else:
                start_idx_depth = d1//2 - self.max_depth//2
            end_idx = start_idx_depth + self.max_depth
            outputs = [item[:, start_idx_depth:end_idx, :, :] for item in outputs]
        
        # crop center
        start_idx_inplane = 0
        if self.inplane_size is not None:
            if h1>self.inplane_size: #crop
                start_idx = int((h1 / 2) - (self.inplane_size / 2))
                end_idx = start_idx + self.inplane_size
                outputs = [item[:, :, start_idx:end_idx, start_idx:end_idx] for item in outputs]
                start_idx_inplane = -start_idx
            elif h1<self.inplane_size: #pad
                pad_start = (self.inplane_size - h1) // 2
                pad_end = self.inplane_size - h1 - pad_start
                outputs = [torch.nn.functional.pad(item, (pad_start, pad_end, pad_start, pad_end)) for item in outputs]
                start_idx_inplane = pad_start
        
        if self.load_pts:
            fixed_pts_path = impath.replace("Fixed", "Fixed_points").replace("npy", "json")
            fixed_pts = np.array(json.load(open(fixed_pts_path, "r"))["pts"])

            moving_pts_path = impath.replace("Fixed", "Moving_points").replace("npy", "json")
            moving_pts = np.array(json.load(open(moving_pts_path, "r"))["pts"])

            # adjust for cropping and padding
            fixed_pts = fixed_pts + np.array([start_idx_inplane, start_idx_inplane, -start_idx_depth])
            moving_pts = moving_pts + np.array([start_idx_inplane, start_idx_inplane, -start_idx_depth])

            outputs += [fixed_pts, moving_pts]
 
        return outputs

    def partition(self, indices):
        """
        slice the data and targets for a given list of indices
        """
        self.indices = indices
        self.data = [self.data[i] for i in indices]
        return self      