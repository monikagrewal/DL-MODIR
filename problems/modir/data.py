import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import os
import numpy as np, cv2
import skimage
from scipy.spatial import ConvexHull

import pdb


def empty_img(shape, color, grayscale=True):
    img = color * np.ones(shape, dtype=np.float32)
    if not grayscale:
        img = cv2.COLOR_GRAY2RGB(img)
    return img

def random_circle(img, color):
    siz = max(img.shape)
    center_x = torch.randint(siz//4, 3*siz//4, (1,)).item()
    center_y = torch.randint(siz//4, 3*siz//4, (1,)).item()
    radius = torch.randint(siz//8, siz//4, (1,)).item()

    img = cv2.circle(img, (center_x, center_y), radius, color, -1)
    return img

def random_rectangle(img, color):
    siz = max(img.shape)
    x0 = torch.randint(siz//8, siz//2, (1,)).item()
    y0 = torch.randint(siz//8, siz//2, (1,)).item()
    w = torch.randint(siz//4, siz//2, (1,)).item()
    h = torch.randint(siz//4, siz//2, (1,)).item()
    x1 = max(x0 + w, 7*siz//8)
    y1 = max(y0 + h, 7*siz//8)

    img = cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
    return img

def random_ellipse(img, color):
    siz = max(img.shape)
    center_x = torch.randint(siz//4, 3*siz//4, (1,)).item()
    center_y = torch.randint(siz//4, 3*siz//4, (1,)).item()
    radius_x = torch.randint(siz//8, siz//3, (1,)).item()
    radius_y = torch.randint(siz//8, siz//3, (1,)).item()
    angle = torch.randint(0, 90, (1,)).item()

    img = cv2.ellipse(img, (center_x, center_y), (radius_x, radius_y), angle, 0, 360, color, -1)
    return img


class Polygon(object):
    """generates a random polygon of given number of vertices"""
    def __init__(self, vertices=3):
        super(Polygon, self).__init__()
        self.vertices = vertices

    def __call__(self, img, color):
        siz = max(img.shape)
        pts = []
        while len(pts) < self.vertices:
            x0 = torch.randint(siz//8, 7*siz//8, (1,)).item()
            y0 = torch.randint(siz//8, 7*siz//8, (1,)).item()
            if len(pts)>0:
                dist = [np.sqrt((x0 - x)**2 + (y0 - y)**2) for (x, y) in pts]
                if (np.array(dist) >= siz//8).all():
                    pts.append((x0, y0))
            else:
                pts.append((x0, y0))

        pts = np.array(pts).astype(np.int32)
        hull = ConvexHull(pts)
        pts = pts[hull.vertices]

        rr, cc = skimage.draw.polygon(pts[:,0], pts[:,1])
        if len(img.shape)==2:
            img[rr, cc] = color
        else:
            img[rr, cc, :] = (color, color, color)
        return img        


class ShapeDataset(Dataset):
    """Dataset class defining dataloader for 2d shapes"""

    def __init__(self, root="./", n_samples=100, shape=(64, 64), train=True):
        super(ShapeDataset, self).__init__()
        """
        Args:- transform
        """
        self.shape = shape
        self.length = n_samples
        self.train = train
        self.transform = transforms.Compose([
                                transforms.ToTensor()
                                ])
        self.shapes = {0: random_circle,
                    1: random_rectangle,
                    2: random_ellipse,
                    3: Polygon(5),
                    4: Polygon(6)}

        if not self.train:
            self.length = 8
            self.data = []
            for i in range(self.length):
                self.data.append(self.generate_data())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.train:
            data = self.data[idx]
        else:
            data = self.generate_data()
        return data

    def generate_data(self):
        color_bg = torch.randint(50, 250, (1,)).item() / 255.
        color_fg = torch.randint(50, 250, (1,)).item() / 255.
        while abs(color_bg - color_fg) < 0.1:
            color_fg = torch.randint(50, 250, (1,)).item() / 255.

        img1 = empty_img(self.shape, color_bg)
        p1 = torch.randint(0, len(self.shapes), (1,)).item()
        img1 = self.shapes[p1](img1, color_fg)

        img2 = empty_img(self.shape, color_bg)
        p2 = torch.randint(0, len(self.shapes), (1,)).item()
        img2 = self.shapes[p2](img2, color_fg)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        data = {"X": (img1, img2), "Y": img1}
        return data



class MNIST_DIR(MNIST):
    """Dataset class defining dataloader for MNIST"""

    def __init__(self, root="./", n_samples=100, train=True, download=True):
        super(MNIST_DIR, self).__init__(root, train=True, download=download)
        """
        Args:- transform
        """
        self.length = n_samples
        self.train = train
        self.transform = transforms.Compose([
                                transforms.ToTensor()
                                ])
        self.categories = {}
        self.data = self.data.numpy()
        self.targets = self.targets.numpy()
        for i in range(10):
            self.categories[i] = self.data[self.targets==i]
        del self.data, self.targets

        self.data = []
        for i in range(n_samples):
            self.data.append(self.generate_data())
        del self.categories

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def generate_data(self):
        cat_idx = torch.randint(0, 10, (1,)).item()
        nsamples = len(self.categories[cat_idx])
        index1 = torch.randint(0, nsamples, (1,)).item()
        index2 = torch.randint(0, nsamples, (1,)).item()

        img1 = self.categories[cat_idx][index1]
        img2 = self.categories[cat_idx][index2]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        data = {"X": (img1, img2), "Y": img1}
        return data

    def partition(self, indices):
        """
        slice the data and targets for a given list of indices
        """
        self.indices = indices
        self.data = [self.data[i] for i in indices]
        return self


def get_dataset(name, train=True, **kwargs):
    implemented_classes = ["MNIST_DIR"]
    if name not in implemented_classes:
        raise NotImplementedError("class {} not implemented. \
            implemented dataset classes are {}".format(name, implemented_classes))
    elif name == "MNIST_DIR":
        data_object = MNIST_DIR(train=train, **kwargs)
    else:
        raise RuntimeError("Something is wrong. \
            You probably added wrong name for the dataset class in implemented_classes variable")

    return data_object     


if __name__ == '__main__':
    # dataset = ShapeDataset(transform=None, is_training=True)
    # print(len(dataset))

    root = "/export/scratch3/grewal/Data/"
    dataset = MNIST_DIR(root, transform=None, train=True)
    # visualization
    out_dir = "./sanity"
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(dataset)):
        print(i)
        img1, img2 = dataset[i]
        im = np.concatenate((img1, img2), axis=1)
        cv2.imwrite("{}/{}.jpg".format(out_dir, i), (im*255).astype(np.uint8))

