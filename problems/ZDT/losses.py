import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb

def normalize(x, xl, xu):
    return (x - xl) / (xu - xl)

class Loss(nn.Module):
    """
    ZDT5 from pymoo https://github.com/anyoptimization/pymoo/blob/main/pymoo/problems/multi/zdt.py
    """
    def __init__(self, loss_name_list, m=11, n=5, normalize=True, **kwargs):
        super().__init__()
        self.n_obj = len(loss_name_list)
        self.m = m
        self.n = n
        self.normalize = normalize
        self.n_var = 30 + n * (m - 1)


    def forward(self, x, y):
        _x = [x[:30]]
        for i in range(self.m - 1):
            _x.append(x[30 + i * self.n: 30 + (i + 1) * self.n])

        u = torch.stack([x_i.sum() for x_i in _x])
        v = (2 + u) * (u < self.n) + 1 * (u == self.n)
        g = v[1:].sum()

        f1 = 1 + u[0]
        f2 = g * (1 / f1)

        if self.normalize:
            pass
            # f1 = normalize(f1, 1, 31)
            # f2 = normalize(f2, (self.m-1) * 1/31, (self.m-1))

        return [f1.view(-1), f2.view(-1)]


