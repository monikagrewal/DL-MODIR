import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

from problems.modir3d.models.unet import (conv_block,
                                          deconv_block,
                                          Unet_DoubleConvBlock,
                                          weight_init)
                                          

class STNet(nn.Module):
    """
    Implementation of U-Net with Spatial Transformer layers
    """

    def __init__(
        self,
        depth=4,
        width=64,
        growth_rate=2,
        in_channels=1,
        threeD=True,
        **kwargs
    ):
        super().__init__()
        self.depth = depth
        self.out_channels = [width * (growth_rate ** i) for i in range(self.depth + 1)]
        self.threeD = threeD
        if threeD:
            self.num_dvf_channels = 3
        else:
            self.num_dvf_channels = 2

        # Downsampling Path Layers
        self.downblocks = nn.ModuleList()
        self.dvfblocks = nn.ModuleList()
        current_in_channels = in_channels
        for i in range(self.depth):
            self.downblocks.append(
                Unet_DoubleConvBlock(
                    current_in_channels, self.out_channels[i], threeD=threeD
                )
            )
            current_in_channels = self.out_channels[i]

            if self.threeD:
                self.dvfblocks.append(
                    nn.Sequential(
                        conv_block(in_ch=2*current_in_channels, out_ch=current_in_channels, threeD=threeD),
                        conv_block(in_ch=current_in_channels, out_ch=current_in_channels//2, threeD=threeD),
                        nn.Conv3d(current_in_channels//2, self.num_dvf_channels, kernel_size=1)
                    )
                )
            else:
                self.dvfblocks.append(
                    nn.Sequential(
                        conv_block(in_ch=2*current_in_channels, out_ch=current_in_channels, threeD=threeD),
                        conv_block(in_ch=current_in_channels, out_ch=current_in_channels//2, threeD=threeD),
                        nn.Conv2d(current_in_channels//2, self.num_dvf_channels, kernel_size=1)
                    )
                )

        if threeD:
            self.downsample = nn.MaxPool3d(2)
        else:
            self.downsample = nn.MaxPool2d(2)

        # Initialization
        self.apply(weight_init)
        self.params = list(self.parameters())
    
    def resampler(self, input, dvf):
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
            output = F.grid_sample(input, grid, mode="bilinear")
        else:
            dvf = dvf.permute(0,2,3,4,1)
    
            b, d, h, w, _ = dvf.shape
            grid_z, grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, d), torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
            grid = torch.stack((grid_x, grid_y, grid_z)).permute(1,2,3,0).view(1, d, h, w, 3)
            grid = torch.repeat_interleave(grid, b, dim=0)
            grid = grid.to(device=device, dtype=dtype)
    
            grid = grid + dvf
            output = F.grid_sample(input, grid, mode="bilinear", align_corners=True)
        return output, dvf

    def forward(self, inputs):
        device = self.device
        fixed, moving = inputs[0].to(device), inputs[1].to(device)
        shp = list(fixed.shape)[2:]
        # Downsampling Path and bottleneck
        fixed_features_list = list()
        out = fixed
        for i in range(self.depth):
            out = self.downblocks[i](out)
            fixed_features_list.append(out)
            out = self.downsample(out)

        moving_features_list = list()
        out = moving
        for i in range(self.depth):
            out = self.downblocks[i](out)
            moving_features_list.append(out)
            out = self.downsample(out)

        # ST
        for i in range(self.depth):
            fixed_features = fixed_features_list[i]
            moving_features = moving_features_list[i]

            out_combo = torch.cat([fixed_features, moving_features], dim=1)
            if i==0:
                final_dvf = self.dvfblocks[i](out_combo)
            else:
                dvf = self.dvfblocks[i](out_combo)
                dvf = F.interpolate(dvf, shp, align_corners=True, mode="trilinear")
                final_dvf += dvf
        
        out_moving, final_dvf = self.resampler(moving, dvf)
        return out_moving, final_dvf

    def inference(self, x):
        out, _ = self.forward(x)
        return out

    def update_device(self, device):
        self.device = device
        self.to(self.device)


if __name__ == "__main__":
    pass
