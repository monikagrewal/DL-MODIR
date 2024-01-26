import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import logging


class Net(nn.Module):
    def __init__(self, n_intermediate_layers=2, n_neurons=100, target_device="cuda:0"):
        super().__init__()
        n_features_in = 1
        n_outputs = 1
        
        cur_features = n_features_in
        self.layer_list = nn.ModuleList()
        for _ in range(0, n_intermediate_layers):
                self.layer_list.append(torch.nn.Linear(cur_features, n_neurons))
                cur_features = n_neurons

        self.linear_out = torch.nn.Linear(cur_features, n_outputs)
        self.params = list(self.parameters())

    def forward(self, inputs):
        inputs = inputs.to(self.device)
        outs = inputs
        for layer in self.layer_list:
            outs = layer(outs)
            outs = F.relu(outs)

        model_out = self.linear_out(outs)
        return(model_out)
    
    def update_device(self, device):
        self.device = device
        self.to(self.device)


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