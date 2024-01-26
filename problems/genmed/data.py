import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TrigonometricDataset(Dataset):
    def __init__(self, **kwargs):
        n_samples = kwargs.get("n_samples", 400)
        n_targets = kwargs.get("n_targets", 2)
        n_var = kwargs.get("n_var", n_targets)
        default_weights = [1 for _ in range(n_targets)]
        weights = kwargs.get("weights", default_weights)
        assert n_var >= n_targets
        self.X, self.Y = self.generate_dataset(n_samples=n_samples, \
                                               n_targets=n_targets, \
                                                n_var=n_var, \
                                                weights=weights)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx, :]
        Y = self.Y
        data = {"X": X, "Y": Y}
        return data

    def partition(self, indices):
        """
        slice the data and targets for a given list of indices
        """
        self.indices = indices
        self.X = self.X[indices, :]
        return self   
    
    def generate_dataset(self, n_samples=400, n_targets=2, n_var=2, weights=None):
        if weights is None:
            weights = [1 for _ in range(n_targets)]
        X = torch.ones(n_samples).float()
        Y = [weights[i] * torch.eye(n_var, dtype=torch.float)[i].view(-1) for i in range(n_targets)]

        # needs extra dimension so that a NN recognizes these as a batch
        X = X.reshape(-1, 1)
        return X, Y


def get_dataset(name, train=True, **kwargs):
    data_object = TrigonometricDataset(**kwargs)
    return data_object



if __name__ == '__main__':
    pass

