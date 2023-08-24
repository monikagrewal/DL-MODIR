import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torchvision import models

from problems.modir3d.models import (mo_voxelmorph,
								    voxelmorph,
										)



def get_network(name, **kwargs):
	implemented_classes = ["VoxelMorph", "MOVoxelMorph"]
	if name not in implemented_classes:
		raise NotImplementedError("class {} not implemented. \
			implemented network classes are {}".format(name, implemented_classes))
	elif name == "VoxelMorph":
		net_object = voxelmorph.VxmDense(**kwargs)
	elif name == "MOVoxelMorph":
		net_object = mo_voxelmorph.VxmDense(**kwargs)
	else:
		raise RuntimeError("Something is wrong. \
			You probably added wrong name for the dataset class in implemented_classes variable")

	return net_object



if __name__ == '__main__':
	device = "cuda:0"
	input1 = torch.rand(1, 1, 64, 64).to(device)
	input2 = torch.rand(1, 1, 64, 64).to(device)
	model = Net().to(device)
	output = model(input1, input2)
	print(output.shape)