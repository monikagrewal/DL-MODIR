import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
import logging

from . import vm_layers
# from .vm_modelio import LoadableModel, store_config_args

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


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


def spatial_transformer(src, flow, mode='bilinear'):
    """
    N-D Spatial Transformer
    """

    # create sampling grid
    size = src.shape[2:]
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors, indexing="ij")
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor).to(src.device)

    # new locations
    new_locs = grid + flow
    shape = flow.shape[2:]

    # need to normalize grid values to [-1, 1] for resampler
    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

    # move channels dim to last position
    # also not sure why, but the channels need to be reversed
    if len(shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
    elif len(shape) == 3:
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

    return nnf.grid_sample(src, new_locs, align_corners=True, mode=mode)


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 K=5,
                 convblock_type="original",
                 use_segmentation=True):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
            K: number of reg_decoders
        """

        super().__init__()
        self.K = K

        # select Original or CustomConvBlock
        if convblock_type=="original":
            ConvBlock = OriginalConvBlock
        elif convblock_type=="unet":
            ConvBlock = CustomConvBlock
        else:
            raise ValueError("conv block = {convblock_type} unknown.")

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        
        # configure reg decoder (up-sampling path)
        self.reg_decoders_list = nn.ModuleList()
        self.reg_remaining_list = nn.ModuleList()
        self.flow_layer_list = nn.ModuleList()
        Conv = getattr(nn, 'Conv%dd' % ndims)
        decoder_nfs = np.flip(encoder_nfs)
        for k in range(self.K):
            decoder = nn.ModuleList()
            prev_nf_reg = prev_nf
            for level in range(self.nb_levels - 1):
                convs = nn.ModuleList()
                for conv in range(nb_conv_per_level):
                    nf = dec_nf[level * nb_conv_per_level + conv]
                    convs.append(ConvBlock(ndims, prev_nf_reg, nf))
                    prev_nf_reg = nf
                decoder.append(convs)
                if not half_res or level < (self.nb_levels - 2):
                    prev_nf_reg += decoder_nfs[level]

            self.reg_decoders_list.append(decoder)

            # now we take care of any remaining convolutions
            reg_remaining = nn.ModuleList()
            for num, nf in enumerate(final_convs):
                reg_remaining.append(ConvBlock(ndims, prev_nf_reg, nf))
                prev_nf_reg = nf
            self.reg_remaining_list.append(reg_remaining)

            # configure unet to flow field layer
            flow_layer = Conv(prev_nf_reg, ndims, kernel_size=3, padding=1)
            # init flow layer with small weights and bias
            sigma = 1  #originally it is 1e-5
            flow_layer.weight = nn.Parameter(Normal(0, sigma).sample(flow_layer.weight.shape))
            flow_layer.bias = nn.Parameter(torch.zeros(flow_layer.bias.shape))
            self.flow_layer_list.append(flow_layer)
        
        if convblock_type=="unet":
            self.encoder.apply(weight_init)
            self.reg_decoders_list.apply(weight_init)
            self.reg_remaining_list.apply(weight_init)
        
        self.use_segmentation = use_segmentation


    def forward(self, x, y, xs=None, ys=None):
        # encoder forward pass
        if self.use_segmentation and xs is not None and ys is not None:
            x = torch.cat([x, y, xs, ys], dim=1)
        else:
            x = torch.cat([x, y], dim=1)
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)
        x_encoder = x

        # reg decoder forward pass with upsampling and concatenation
        flow_fields = []
        for k in range(self.K):
            i = -1
            x = x_encoder
            for level, convs in enumerate(self.reg_decoders_list[k]):
                for conv in convs:
                    x = conv(x)
                if not self.half_res or level < (self.nb_levels - 2):
                    x = self.upsampling[level](x)
                    x = torch.cat([x, x_history[i]], dim=1)
                    i -= 1

            # remaining convs at full resolution
            for conv in self.reg_remaining_list[k]:
                x = conv(x)
            
            # flow layer
            dvf = self.flow_layer_list[k](x)
            flow_fields.append(dvf)

        return flow_fields


class VxmDense(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self,
                 inshape=(32, 192, 192),
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 K=1,
                 use_segmentation=True,
                 seg_classes=5,
                 convblock_type="original",
                 **kwargs):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
            K: number of heads for multi objective implementation.
            use_segmentation: Whether or not to use segmentation masks as input
            seg_classes: Number of classes in segmentation mask
        """
        super().__init__()

        self.use_segmentation = use_segmentation #False has no meaning
        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        if use_segmentation:
            infeats = src_feats + trg_feats + 2 * seg_classes
        else:
            infeats = src_feats + trg_feats
        self.unet_model = Unet(
            inshape,
            infeats=infeats,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            K=K,
            convblock_type=convblock_type,
            use_segmentation=use_segmentation
        )
        self.K = K

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = vm_layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = vm_layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = vm_layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        self.transformer = spatial_transformer

        self.params = list(self.parameters())

    def forward(self, inputs, registration=False):
        '''
        Parameters:
            inputs = [target, source, target_seg, source_seg]
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''
        device = self.device
        inputs = [item.to(device) for item in inputs]
        
        # concatenate inputs and propagate unet
        flow_fields = self.unet_model(*inputs)

        outputs_list = []
        for flow_field in flow_fields:
            # resize flow for integration
            pos_flow = flow_field
            if self.resize:
                pos_flow = self.resize(pos_flow)

            preint_flow = pos_flow
            logging.debug(f"DVF: max = {preint_flow.max()}, min = {preint_flow.min()}")

            # negate flow for bidirectional model
            neg_flow = -pos_flow if self.bidir else None

            # integrate to produce diffeomorphic warp
            if self.integrate:
                pos_flow = self.integrate(pos_flow)
                neg_flow = self.integrate(neg_flow) if self.bidir else None

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                    neg_flow = self.fullsize(neg_flow) if self.bidir else None

            # warp image with flow field
            y_source = self.transformer(inputs[1], pos_flow)
            y_target = self.transformer(inputs[0], neg_flow) if self.bidir else None

            # return non-integrated flow field if training
            if not registration:
                outputs = [y_source, y_target, preint_flow] if self.bidir else [y_source, preint_flow]
            else:
                outputs = [y_source, pos_flow]
            
            if len(inputs)==4:  #seg masks provided
                source_seg_warped = self.transformer(inputs[3], pos_flow)   
                outputs += [source_seg_warped]
            
            outputs_list.append(outputs)

        if self.K==1:
            return outputs_list[0]
        else:
            return outputs_list


    def inference(self, x, pts=None):
        outs_list = self.forward(x)
        if pts is not None:
            pts = pts.float().to(self.device)
            if self.K==1:
                flow = outs_list[1]
                outs_list.append(vm_layers.transform_points(pts, flow))
            else:
                for i in range(len(outs_list)):
                    flow = outs_list[i][1]
                    outs_list[i].append(vm_layers.transform_points(pts, flow))
        return outs_list

    def update_device(self, device):
        self.device = device
        self.to(self.device)


class OriginalConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class CustomConvBlock(nn.Module):
    """
    Specific convolutional block followed by batchnorm and relu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        BatchNorm = getattr(nn, 'BatchNorm%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.bn = BatchNorm(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.main(x)
        out = self.bn(out)
        out = self.activation(out)
        return out