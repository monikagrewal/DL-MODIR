import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torchvision import models

from problems.modir3d.models import (unet,
				     				stnet,
								    dirunet,
				     				stunet,
								    voxelmorph,
								    voxelmorph_unet
										)



def get_network(name, **kwargs):
	implemented_classes = ["STNet", "STUNet", "DIRUNet", "VoxelMorph", "VoxelMorphUNet"]
	if name not in implemented_classes:
		raise NotImplementedError("class {} not implemented. \
			implemented network classes are {}".format(name, implemented_classes))
	elif name == "STNet":
		net_object = stnet.STNet(**kwargs)
	elif name == "DIRUNet":
		net_object = dirunet.DIRUNet(**kwargs)
	elif name == "STUNet":
		net_object = stunet.STUNet(**kwargs)
	elif name == "VoxelMorph":
		net_object = voxelmorph.VxmDense(**kwargs)
	elif name == "VoxelMorphUNet":
		net_object = voxelmorph_unet.VxmDense(**kwargs)
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