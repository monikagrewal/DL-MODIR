import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class TrigonometricDataset(Dataset):
    def __init__(self, **kwargs):
        n_samples = kwargs.get("n_smaples", 400)
        n_targets = kwargs.get("n_targets", 2)
        cycles = kwargs.get("cycles", 1)
        self.X, self.Y = self.generate_dataset(n_samples=n_samples, n_targets=n_targets, cycles=cycles)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx, :]
        Y = self.Y[idx, :]
        data = {"X": X, "Y": Y}
        return data

    def partition(self, indices):
        """
        slice the data and targets for a given list of indices
        """
        self.indices = indices
        self.X = self.X[indices, :]
        self.Y = self.Y[indices, :]
        return self   
    
    def generate_dataset(self, n_samples=400, n_targets=2, cycles=1):
        X = np.random.uniform(0, cycles * 2 * np.pi, n_samples).astype(np.float32)
        Y = np.zeros((n_samples, n_targets), dtype=np.float32)

        Y[:, 0] = np.cos(X)
        Y[:, 1] = np.sin(X)
        if n_targets==3:
            Y[:, 2] = np.sin(X + 1*np.pi)

        # needs extra dimension so that a NN recognizes these as a batch
        X = X.reshape(-1, 1)
        return X, Y


def get_dataset(name, train=True, **kwargs):
    data_object = TrigonometricDataset(**kwargs)
    return data_object



if __name__ == '__main__':
    pass

