import os
import cv2
import numpy as np
from problems.modir3d.datasets import (empire10,
                    amc_brachy)
# from datasets import amc_brachy


def get_dataset(name, train=True, **kwargs):
    implemented_classes = ["Empire10", "AMCBrachy"]
    if name not in implemented_classes:
        raise NotImplementedError("class {} not implemented. \
            implemented dataset classes are {}".format(name, implemented_classes))
    elif name == "Empire10":
        data_object = empire10.Empire10(**kwargs)
    elif name == "AMCBrachy":
        data_object = amc_brachy.AMCBrachy(**kwargs)
    else:
        raise RuntimeError("Something is wrong. \
            You probably added wrong name for the dataset class in implemented_classes variable")

    return data_object     


if __name__ == '__main__':
    # root = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/MR_sag/MODIR_data_train_split'
    # csv_path = '/export/scratch2/data/grewal/DL-MODIR/data_preparation/meta/mri_dataset_train_pairs.csv'
    # amc_brachy.preprocess_modir_data(root, csv_path)

    # root = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/LUMC_cervical_train'
    # csv_path = '/export/scratch2/data/grewal/DL-MODIR/data_preparation/meta/LUMC_cervical_train_pairs.csv'
    # amc_brachy.preprocess_modir_data(root, csv_path, output_spacing=(1, 1, 4), output_size="image")

    root = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/LUMC_cervical_train'
    dataset = amc_brachy.AMCBrachy(root=root)
    # visualization
    out_dir = "./sanity_lumc_cervical"
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(dataset)):
        print(i)
        data = dataset[i]
        img1, img2, img1_seg, img2_seg = data["X"]
        img1 = img1.data.cpu().numpy()
        img2 = img2.data.cpu().numpy()
        img1_seg = img1_seg.data.cpu().numpy()
        img2_seg = img2_seg.data.cpu().numpy()
        # convert onehot seg to mask
        img1_seg = np.argmax(img1_seg, axis=0)
        img2_seg = np.argmax(img2_seg, axis=0)

        # normalize img1, img2
        img1 = img1 / img1.max()
        img2 = img2 / img2.max()

        print("im shapes: ", img1.shape, img2.shape, img1_seg.shape, img2_seg.shape)
        print("im values: ", img1.max(), img1.min(), img2.max(), img2.min(), np.unique(img1_seg), np.unique(img2_seg))
        nslices = img1.shape[1]
        for i_slice in range(nslices):
            im = np.concatenate((img1[0, i_slice, :, :], img2[0, i_slice, :, :]), axis=1)
            im = (im * 255).astype(np.uint8)
            cv2.imwrite("{}/{}_{}.jpg".format(out_dir, i, i_slice), im)

            im_seg = np.concatenate((img1_seg[i_slice, :, :], img2_seg[i_slice, :, :]), axis=1)
            im_seg = ((im_seg / im_seg.max()) * 255).astype(np.uint8)
            cv2.imwrite("{}/seg_{}_{}.jpg".format(out_dir, i, i_slice), im_seg)

        if i>10:
            break

