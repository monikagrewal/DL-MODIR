import os
import cv2
import numpy as np
from problems.modir3d.datasets import (empire10,
                    amc_brachy)
# from datasets import amc_brachy
# from utils import seg_to_im


def get_dataset(name, train=True, **kwargs):
    implemented_classes = ["Empire10", "AMCBrachy"]
    if name not in implemented_classes:
        raise NotImplementedError("class {} not implemented. \
            implemented dataset classes are {}".format(name, implemented_classes))
    elif name == "Empire10":
        data_object = empire10.Empire10(**kwargs)
    elif name == "AMCBrachy":
        data_object = amc_brachy.AMCBrachy(train=train, **kwargs)
    else:
        raise RuntimeError("Something is wrong. \
            You probably added wrong name for the dataset class in implemented_classes variable")

    return data_object


def visualize_seg_mask(image, seg_mask):
    seg_color = seg_to_im(seg_mask)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    alpha = 0.4
    image_combined = alpha * seg_color + (1-alpha) * image_color
    return image_combined


def visualize_pts(image:np.array, pts:list, slice_no:int, color:tuple):
    for x, y, z in pts:
        if int(round(z, 0))==slice_no:
            x, y = int(round(x, 0)), int(round(y, 0))
            image = cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 6)
    return image



if __name__ == '__main__':
    # root = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/MR_sag/MODIR_data_train_split'
    # csv_path = '/export/scratch2/data/grewal/DL-MODIR/data_preparation/meta/mri_dataset_train_pairs.csv'
    # amc_brachy.preprocess_modir_data(root, csv_path)

    # root = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/LUMC_cervical_train'
    # csv_path = '/export/scratch2/data/grewal/DL-MODIR/data_preparation/meta/LUMC_cervical_train_pairs.csv'
    # amc_brachy.preprocess_modir_data(root, csv_path, output_spacing=(1, 1, 1), output_foldername="preprocessed111", output_size="image")

    root = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/LUMC_cervical_test_annotated'
    csv_path = '/export/scratch2/data/grewal/DL-MODIR/data_preparation/meta/LUMC_cervical_test_pairs.csv'
    amc_brachy.preprocess_modir_data(root, csv_path, output_spacing=(1, 1, 1), output_foldername="preprocessed111", load_pts=True, output_size="image")

    # root = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/LUMC_cervical_test_annotated'
    # dataset = amc_brachy.AMCBrachy(root=root)
    # # visualization
    # out_dir = "./sanity_lumc_test"
    # os.makedirs(out_dir, exist_ok=True)
    # for i in range(len(dataset)):
    #     print(i)
    #     data = dataset[i]
    #     img1, img2, img1_seg, img2_seg = data["X"]
    #     img1_pts, img2_pts = data.get("pts", ([], []))
    #     img1 = img1.data.cpu().numpy()
    #     img2 = img2.data.cpu().numpy()
    #     img1_seg = img1_seg.data.cpu().numpy()
    #     img2_seg = img2_seg.data.cpu().numpy()
    #     # convert onehot seg to mask
    #     img1_seg = np.argmax(img1_seg, axis=0)
    #     img2_seg = np.argmax(img2_seg, axis=0)

    #     print("im shapes: ", img1.shape, img2.shape, img1_seg.shape, img2_seg.shape)
    #     print("im values: ", img1.max(), img1.min(), img2.max(), img2.min(), np.unique(img1_seg), np.unique(img2_seg))
    #     nslices = img1.shape[1]
    #     for i_slice in range(nslices):
    #         im1 = img1[0, i_slice, :, :]
    #         im1_seg = img1_seg[i_slice, :, :]
    #         im1_combined = visualize_seg_mask(im1, im1_seg)
    #         im1_combined = visualize_pts(im1_combined, img1_pts, i_slice, (0, 0, 1))

    #         im2 = img2[0, i_slice, :, :]
    #         im2_seg = img2_seg[i_slice, :, :]
    #         im2_combined = visualize_seg_mask(im2, im2_seg)
    #         im1_combined = visualize_pts(im1_combined, img2_pts, i_slice, (0, 1, 0))

    #         im = np.concatenate((im1_combined, im2_combined), axis=1)
    #         im = (im * 255).astype(np.uint8)
    #         cv2.imwrite("{}/{}_{}.jpg".format(out_dir, i, i_slice), im)

    #     if i>10:
    #         break

