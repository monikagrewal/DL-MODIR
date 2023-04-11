import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.distributions.normal import Normal

from problems.modir3d.models.unet import (conv_block,
                                          deconv_block,
                                          Unet_DoubleConvBlock,
                                          weight_init)

from . import vm_layers
                                          



class STUNet(nn.Module):
    """
    Implementation of U-Net with Spatial Transformer layers
    """

    def __init__(
        self,
        depth=4,
        width=64,
        growth_rate=2,
        in_channels=2,
        out_channels=1,
        threeD=True,
        use_segmentation=False,
        multi_resolution=False,
        **kwargs
    ):
        super().__init__()
        self.depth = depth
        self.out_channels = [width * (growth_rate ** i) for i in range(self.depth + 1)]
        self.threeD = threeD
        self.use_segmentation = use_segmentation
        self.multi_resolution = multi_resolution
        logging.debug(f"use_segmentation: {use_segmentation}, multi_resolution: {multi_resolution}")

        # Downsampling Path Layers
        self.downblocks = nn.ModuleList()
        current_in_channels = in_channels
        for i in range(self.depth + 1):
            self.downblocks.append(
                Unet_DoubleConvBlock(
                    current_in_channels, self.out_channels[i], threeD=threeD
                )
            )
            current_in_channels = self.out_channels[i]

        # Upsampling Path Layers
        self.deconvblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.dvfblocks = nn.ModuleList()
        for i in range(self.depth):
            self.deconvblocks.append(
                deconv_block(
                    current_in_channels, self.out_channels[-2 - i], threeD=threeD
                )
            )
            self.upblocks.append(
                Unet_DoubleConvBlock(
                    current_in_channels, self.out_channels[-2 - i], threeD=threeD
                )
            )
            if self.threeD:
                self.dvfblocks.append(
                    nn.Conv3d(self.out_channels[-2 - i], 3, kernel_size=3)
                    # nn.Sequential(
                    #     conv_block(in_ch=current_in_channels, out_ch=self.out_channels[-2 - i], threeD=threeD),
                    #     conv_block(in_ch=self.out_channels[-2 - i], out_ch=self.out_channels[-2 - i], threeD=threeD),
                    #     nn.Conv3d(self.out_channels[-2 - i], 3, kernel_size=1)
                    # )
                )
            else:
                self.dvfblocks.append(
                    nn.Conv2d(self.out_channels[-2 - i], 2, kernel_size=3)
                    # nn.Sequential(
                    #     conv_block(in_ch=current_in_channels, out_ch=self.out_channels[-2 - i], threeD=threeD),
                    #     conv_block(in_ch=self.out_channels[-2 - i], out_ch=self.out_channels[-2 - i], threeD=threeD),
                    #     nn.Conv2d(self.out_channels[-2 - i], 2, kernel_size=1)
                    # )
                )

            current_in_channels = self.out_channels[-2 - i]

        if threeD:
            self.last_layer = nn.Conv3d(
                current_in_channels, 3, kernel_size=1
            )
            self.downsample = nn.MaxPool3d(2)
        else:
            self.last_layer = nn.Conv2d(
                current_in_channels, 2, kernel_size=1
            )
            self.downsample = nn.MaxPool2d(2)

        # Initialization
        self.apply(weight_init)
        self.params = list(self.parameters())

        # init with small weights and bias
        """
        calculation of sigma: max deformation in pixels = 16,
        in precited dvf h or w (192) pixels --> 2, 16 pixels --> 2/192 * 16
        max deformation = 3 * sigma = 2/192 * 16,
        sigma = (2/192 * 16) / 3 = 0.05
        """
        # sigma = 0.05
        # self.last_layer.weight = nn.Parameter(Normal(0, sigma).sample(self.last_layer.weight.shape))
        # self.last_layer.bias = nn.Parameter(torch.zeros(self.last_layer.bias.shape))
        # for i in range(len(self.dvfblocks)):
        #     self.dvfblocks[i].weight = nn.Parameter(Normal(0, sigma).sample(self.dvfblocks[i].weight.shape))
        #     self.dvfblocks[i].bias = nn.Parameter(torch.zeros(self.dvfblocks[i].bias.shape))

        # configure transformer
        # this saves time in computing the grid in every forward propagation,
        # but has a prolem during variable size inference
        inshape = (32, 192, 192)
        self.transformer = vm_layers.SpatialTransformer(inshape)
    
    def resampler(self, input, dvf, mode="bilinear"):
        device = input.device
        dtype = dvf.dtype
        if not self.threeD:
            dvf = dvf.permute(0,2,3,1)

            b, h, w, _ = dvf.shape
            dtype = dvf.dtype
            grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
            grid = torch.stack((grid_x, grid_y)).permute(1,2,0).view(1, h, w, 2)
            grid = torch.repeat_interleave(grid, b, dim=0)
            grid = grid.to(device=device, dtype=dtype)

            grid = grid + dvf
            # copied from voxelmorph code, who is not sure why the channels need to be reversed
            grid = grid[..., [1, 0]]
            output = F.grid_sample(input, grid, mode=mode, align_corners=True)
        else:
            dvf = dvf.permute(0,2,3,4,1)
    
            b, d, h, w, _ = dvf.shape
            grid_z, grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, d), torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
            grid = torch.stack((grid_x, grid_y, grid_z)).permute(1,2,3,0).view(1, d, h, w, 3)
            grid = torch.repeat_interleave(grid, b, dim=0)
            grid = grid.to(device=device, dtype=dtype)
    
            grid = grid + dvf
            output = F.grid_sample(input, grid, mode=mode, align_corners=True)
        return output, dvf

    def forward(self, inputs):
        device = self.device
        fixed, moving = inputs[0].to(device), inputs[1].to(device)
        shp = list(fixed.shape)[2:]

        # Downsampling Path and bottleneck
        down_features_list = list()
        out = torch.cat([fixed, moving], dim=1)
        for i in range(self.depth):
            out = self.downblocks[i](out)
            down_features_list.append(out)
            out = self.downsample(out)
        
        out = self.downblocks[-1](out)

        # Upsampling Path
        dvf_list = []
        for i in range(self.depth):
            out = self.deconvblocks[i](out)

            # ST
            down_features = down_features_list[-1 - i]
            out_combo = torch.cat([down_features, out], dim=1)
            out = self.upblocks[i](out_combo)
            
            dvf = self.dvfblocks[i](out)
            dvf_list.append(dvf)

        if self.multi_resolution:
            dvf_full_list = [F.interpolate(dvf, shp, align_corners=True, mode="trilinear") for
                            dvf in dvf_list[2:]]
            final_dvf = (1/len(dvf_full_list)) * torch.stack(dvf_full_list, dim=0).sum(dim=0)
        else:
            final_dvf = self.last_layer(out)
        
        # warp image with dvf
        out_moving = self.transformer(moving, final_dvf)

        if self.use_segmentation:
            fixed_seg = inputs[2].to(device)
            moving_seg = inputs[3].to(device)
            moving_seg_warped = self.transformer(moving_seg, final_dvf)
            return out_moving, final_dvf, fixed_seg, moving_seg_warped
        else:
            return out_moving, final_dvf

    def inference(self, x):
        out, _ = self.forward(x)
        return out

    def update_device(self, device):
        self.device = device
        self.to(self.device)


if __name__ == "__main__":
    pass
