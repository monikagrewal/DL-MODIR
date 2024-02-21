import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
import pdb

"""
For information on GenMED problem, refer to https://ir.cwi.nl/pub/21263/bosman.pdf.
"""


class Loss(nn.Module):
    """
    GenMED, scalable upto 3 objectives
    """
    def __init__(self, loss_name_list, mode='convex', **kwargs):
        super().__init__()
        self.n_obj = len(loss_name_list)
        self.mode = mode

    
    def forward(self, x, y):
        y = [item.view(-1).to(x.device) for item in y]
        assert len(x) == len(y[0])
        assert len(y[0]) >= self.n_obj

        f = []
        for i in range(0, self.n_obj):
            if self.mode == 'convex':
                _f = 0.5 * torch.pow(torch.abs(x - y[i]), 2).sum()
            else:
                _f = 0.5 * torch.norm(torch.abs(x - y[i]))
            f.append(_f.view(-1))
        
        logging.debug(f"{x.data}, {[item.item() for item in f]}, {torch.sum(torch.tensor(f))}")
        
        return f
    

# class Loss(nn.Module):
#     """
#     GenMED, scalable upto 3 objectives
#     """
#     def __init__(self, loss_name_list, mode='convex', **kwargs):
#         super().__init__()
#         self.n_obj = len(loss_name_list)
#         self.mode = mode

    
#     def forward(self, x, y):
#         y = [item.view(-1).to(x.device) for item in y]
#         assert len(x) == len(y[0])
#         assert len(y[0]) >= self.n_obj

#         f = []
#         f1 = torch.pow(torch.abs(x - y[0]), 2).sum()
#         f.append(f1.view(-1))
#         f2 = torch.exp( torch.pow(torch.abs(x - y[1]), 2).sum() ) - 1
#         f.append(f2.view(-1))
        
#         logging.debug(f"{x.data}, {[item.item() for item in f]}, {torch.sum(torch.tensor(f))}")
        
#         return f


