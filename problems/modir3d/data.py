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
    root = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/MR_sag/MODIR_data_train_split'
    csv_path = '/export/scratch2/data/grewal/DL-MODIR/data_preparation/meta/mri_dataset_train_pairs.csv'
    amc_brachy.preprocess_modir_data(root, csv_path)

    root = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/MR_sag/MODIR_data_train_split'
    dataset = amc_brachy.AMCBrachy(root=root, train=True)
    # visualization
    out_dir = "./sanity_modir"
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(dataset)):
        print(i)
        data = dataset[i]
        img1, img2, img1_seg, img2_seg = data["X"]
        img1 = img1.data.cpu().numpy()
        img2 = img2.data.cpu().numpy()
        img1_seg = img1_seg.data.cpu().numpy()
        img2_seg = img2_seg.data.cpu().numpy()

        print("im after loading: ", img1.shape, img2.shape, img1_seg.shape, img2_seg.shape)
        nslices = img1.shape[1]
        for i_slice in range(nslices):
            im = np.concatenate((img1[0, i_slice, :, :], img2[0, i_slice, :, :]), axis=1)
            im = ((im / im.max()) * 255).astype(np.uint8)
            cv2.imwrite("{}/{}_{}.jpg".format(out_dir, i, i_slice), im)

            im_seg = np.concatenate((img1_seg[0, i_slice, :, :], img2_seg[0, i_slice, :, :]), axis=1)
            im_seg = ((im_seg / im_seg.max()) * 255).astype(np.uint8)
            cv2.imwrite("{}/seg_{}_{}.jpg".format(out_dir, i, i_slice), im_seg)

        # if i>10:
        #     break

