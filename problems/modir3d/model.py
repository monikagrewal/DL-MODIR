import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torchvision import models


def weight_init(m):
	if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
		torch.nn.init.kaiming_normal_(m.weight.data)
		if m.bias is not None:
			m.bias.data.fill_(0.0)
	if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
		m.weight.data.fill_(1.0)
		m.bias.data.fill_(0.0)
	if isinstance(m, nn.Linear):
		torch.nn.init.kaiming_normal_(m.weight.data)
		if m.bias is not None:
			m.bias.data.fill_(0.0)


def conv_block(in_ch=1, out_ch=1, threeD=True, batchnorm=False, drop_rate=0.2):
	if batchnorm:
		if threeD:
			layer = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
									nn.BatchNorm3d(out_ch),
									nn.ReLU(),
									# nn.Dropout3d(drop_rate)
									)
		else:
			layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
									nn.BatchNorm2d(out_ch),
									nn.ReLU(),
									# nn.Dropout2d(drop_rate)
									)
	else:
		if threeD:
			layer = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
									nn.ReLU(),
									# nn.Dropout3d(drop_rate)
									)
		else:
			layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
									nn.ReLU(),
									# nn.Dropout2d(drop_rate)
									)			
	return layer


def deconv_block(in_ch=1, out_ch=1, scale_factor=2, threeD=True, batchnorm=False):
	if batchnorm:
		if threeD:
			layer = nn.Sequential(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor),
									nn.BatchNorm3d(out_ch),
									nn.ReLU()
									)
		else:
			layer = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor),
									nn.BatchNorm2d(out_ch),
									nn.ReLU()
									)
	else:
		if threeD:
			layer = nn.Sequential(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor),
									nn.ReLU()
									)
		else:
			layer = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor),
									nn.ReLU()
									)

	return layer


def Unet_DoubleConvBlock(in_ch=1, out_ch=1, threeD=True, batchnorm=False):
	layer = nn.Sequential(conv_block(in_ch=in_ch, out_ch=out_ch, threeD=threeD, batchnorm=batchnorm),
							conv_block(in_ch=out_ch, out_ch=out_ch, threeD=threeD, batchnorm=batchnorm)
							)
	return layer


class UNet(nn.Module):
	"""
	Implementation of U-Net
	"""
	def __init__(self, depth=4, width=64, growth_rate=2, in_channels=1, out_channels=2, threeD=False, batchnorm=False):
		super(UNet, self).__init__()
		self.depth = depth
		self.out_channels = [width*(growth_rate**i) for i in range(self.depth+1)]

		# Downsampling Path Layers
		self.downblocks = nn.ModuleList()
		current_in_channels = in_channels
		for i in range(self.depth+1):
			self.downblocks.append(Unet_DoubleConvBlock(current_in_channels, self.out_channels[i], threeD=threeD, batchnorm=batchnorm))
			current_in_channels = self.out_channels[i]

		self.feature_channels = current_in_channels + self.out_channels[i-1]
		# Upsampling Path Layers
		self.deconvblocks = nn.ModuleList()
		self.upblocks = nn.ModuleList()
		for i in range(self.depth):
			self.deconvblocks.append(deconv_block(current_in_channels, self.out_channels[-2 - i], threeD=threeD, batchnorm=batchnorm))
			self.upblocks.append(Unet_DoubleConvBlock(current_in_channels, self.out_channels[-2 - i], threeD=threeD, batchnorm=batchnorm))
			current_in_channels = self.out_channels[-2 - i]

		if threeD:
			self.last_layer = nn.Conv3d(current_in_channels, out_channels, kernel_size=1)
			self.downsample = nn.MaxPool3d(2)
		else:
			self.last_layer = nn.Conv2d(current_in_channels, out_channels, kernel_size=1)
			self.downsample = nn.MaxPool2d(2)			

		# Initialization
		self.apply(weight_init)


	def forward(self, x):
		# Downsampling Path
		out = x
		down_features_list = list()
		for i in range(self.depth):
			out = self.downblocks[i](out)
			down_features_list.append(out)
			out = self.downsample(out)

		# bottleneck
		out = self.downblocks[-1](out)
		features = [down_features_list[-1], out]

		# Upsampling Path
		for i in range(self.depth):
			out = self.deconvblocks[i](out)
			down_features = down_features_list[-1 - i]

			# pad slice and image dimensions if necessary
			down_shape = torch.tensor(down_features.shape)
			out_shape = torch.tensor(out.shape)
			shape_diff = down_shape - out_shape
			pad_list = [padding for diff in reversed(shape_diff.numpy()) for padding in [diff,0]]
			if max(pad_list) == 1:
				out = F.pad(out, pad_list)

			out = torch.cat([down_features, out], dim=1)
			out = self.upblocks[i](out)

		out = self.last_layer(out)	

		return out, features


class Net(nn.Module):
	"""
	NN class for Deformable Image Registration
	"""
	def __init__(self, device, depth=2, width=8, threeD=False, batchnorm=False):
		super().__init__()
		self.device = device
		self.threeD = threeD
		if threeD:
			self.DVFgenerator = UNet(depth=depth, width=width, in_channels=2, out_channels=3, threeD=threeD, batchnorm=batchnorm)
		else:
			self.DVFgenerator = UNet(depth=depth, width=width, in_channels=2, out_channels=2, threeD=threeD, batchnorm=batchnorm)
		self.apply(weight_init)

		# define parameters
		self.params = list(self.parameters())


	def forward(self, inputs):
		"""
		input1: target image
		input2: source image
		output: transformed source image
		"""
		device = self.device
		input1, input2 = inputs[0].to(device), inputs[1].to(device)
		input = torch.cat([input1, input2], dim=1)
		dvf, _ = self.DVFgenerator(input)
		if not self.threeD:
			dvf = dvf.permute(0,2,3,1)

			b, h, w, _ = dvf.shape
			dtype = dvf.dtype
			grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
			grid = torch.stack((grid_x, grid_y)).permute(1,2,0).view(1, h, w, 2)
			grid = torch.repeat_interleave(grid, b, dim=0)
			grid = grid.to(device=device, dtype=dtype)

			grid = grid + dvf
			output = F.grid_sample(input2, grid, mode="bilinear")
		else:
			dvf = dvf.permute(0,2,3,4,1)

			b, d, h, w, _ = dvf.shape
			dtype = dvf.dtype
			grid_z, grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, d), torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
			grid = torch.stack((grid_x, grid_y, grid_z)).permute(1,2,3,0).view(1, d, h, w, 3)
			grid = torch.repeat_interleave(grid, b, dim=0)
			grid = grid.to(device=device, dtype=dtype)

			grid = grid + dvf
			output = F.grid_sample(input2, grid, mode="bilinear")			
		return output, dvf


	def predict(self, input1, input2):
		output, _ = self.forward(input1, input2)
		return output


	def update_device(self, device):
		self.device = device
		self.to(self.device)


def get_network(name, target_device="cuda:0", **kwargs):
	implemented_classes = ["Net"]
	if name not in implemented_classes:
		raise NotImplementedError("class {} not implemented. \
			implemented network classes are {}".format(name, implemented_classes))
	elif name == "Net":
		net_object = Net(device=target_device, **kwargs)
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