import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

from problems.modir3d.models import unet


class DIRUNet(nn.Module):
	"""
	NN class for Deformable Image Registration
	"""
	def __init__(self, depth=2, width=8, threeD=True, batchnorm=True, **kwargs):
		super().__init__()
		self.threeD = threeD
		if threeD:
			self.DVFgenerator = unet.UNet(depth=depth, width=width, in_channels=2, out_channels=3, threeD=threeD, batchnorm=batchnorm)
		else:
			self.DVFgenerator = unet.UNet(depth=depth, width=width, in_channels=2, out_channels=2, threeD=threeD, batchnorm=batchnorm)

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
		dvf = self.DVFgenerator(input)
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



if __name__ == '__main__':
	pass