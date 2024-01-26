import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import logging


class Net(nn.Module):
    """
    It's an optimization type set-up. So the params are directly optimized without inputs
    """
    def __init__(self, n_var=2, target_device="cuda:0", **kwargs):
        super().__init__()
        data = torch.rand(n_var, device=target_device)
        self.params = [torch.nn.Parameter(data)]


    def forward(self, inputs):
        """
        inputs is used to be similar to other neural networks, but is ignored in this case.
        """
        return torch.sigmoid(self.params[0])
    
    def update_device(self, device):
        self.device = device


def get_network(name, **kwargs):
	implemented_classes = ["Net"]
	if name not in implemented_classes:
		raise NotImplementedError("class {} not implemented. \
			implemented network classes are {}".format(name, implemented_classes))
	elif name == "Net":
		net_object = Net(**kwargs)
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