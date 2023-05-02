# most of the code comes from https://github.com/jaxony/unet-pytorch/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from simple_einet.einet import EinetConfig, Einet
from simple_einet.ccleinet import CCLEinet
from simple_einet.distributions import RatNormal, CCRatNormal

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.pooling:
            x = self.pool(x)
        return x


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, x):
        """ Forward pass
        Arguments:
            x: upconv'd tensor from the decoder pathway
        """
        x = self.upconv(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x



class AutoSPN(nn.Module):

    def __init__(
        self,
        in_channels=3,
        # num_classes=1
        depth=5,
        start_filts=64,
        up_mode='transpose',
        spn_depth=3,
        spn_S=5,
        spn_I=5,
        spn_R=5,
        spn_leaf_kwargs={"min_sigma": 1e-2, "max_sigma": 2.0},
        spn_leaf_type=RatNormal
    ):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.

        Note:
            AutoSPN uses lazy init for spn so send one full sample/ batch through
            before using this for crops...
        """
        super(AutoSPN, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        # self.num_classes = num_classes

        self.spn = None # lazy init
        # self.flatten = nn.Flatten(start_dim=2) # flatten height and width and channels remain
        self.flatten = nn.Flatten(start_dim=1) # flatten channels, height and width
        self.unflatten = None # lazy init
        self.down_convs = []
        self.up_convs = []

        self.spn_depth = spn_depth
        self.spn_S = spn_S
        self.spn_I = spn_I
        self.spn_R = spn_R
        self.spn_leaf_kwargs = spn_leaf_kwargs
        self.spn_leaf_type = spn_leaf_type

        # create the encoder
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # decoder
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, 1)

        # add the list of modules to current module
        self.down_convs = nn.Sequential(*self.down_convs)
        self.up_convs = nn.Sequential(*self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def lazy_init(self, unflattend_input_size):
        config = EinetConfig(
            num_features=unflattend_input_size[-3] * unflattend_input_size[-2] * unflattend_input_size[-1], #unflattend_input_size[-2] * unflattend_input_size[-1],
            num_channels=1, #unflattend_input_size[1],
            depth=self.spn_depth,
            num_sums=self.spn_S,
            num_leaves=self.spn_I,
            num_repetitions=self.spn_R,
            num_classes=10, # change later
            leaf_kwargs=self.spn_leaf_kwargs,
            leaf_type=self.spn_leaf_type,
            dropout=0.0,
            cross_product=True,
            log_weights=False,
        )
        self.spn = Einet(config)
        # self.unflatten = nn.Unflatten(-1, unflattend_input_size[-2:])
        self.unflatten = nn.Unflatten(-1, unflattend_input_size[-3:])
        self.latent_shape = unflattend_input_size

    def fix_missing_in_last_dim(self, x): # this can be done much better!
        device = self.spn._sampling_root.weights.device
        needed = self.latent_shape[-1] - x.shape[-1]
        nans = torch.ones(x.shape[0], *self.latent_shape[1:-1], needed) * torch.nan
        nans = nans.to(device)
        return torch.cat((x, nans), -1)


    def forward(self, x, y=None):
        latents = self.down_convs(x)
        if self.unflatten is None: # init unflatten and spn
            self.lazy_init(latents.shape)
        flattend_latents = self.flatten(latents)
        ll = self.spn(flattend_latents)
        return ll

    def sample(self, num_samples=None, mpe_at_leaves=False, evidence=None, training=False):
        marginalized_scopes = None
        # with torch.no_grad():
        #     if evidence is not None:
        #         evidence = self.down_convs(evidence)
        #         evidence = self.fix_missing_in_last_dim(evidence)
        #         evidence = self.flatten(evidence)
        #         marginalized_scopes = evidence.isnan().nonzero(as_tuple=True)[-1]
        #         evidence[..., marginalized_scopes] = torch.zeros(1)
        #     latents = self.spn.sample(
        #         evidence=evidence,
        #         marginalized_scopes=marginalized_scopes,
        #         num_samples=num_samples,
        #         mpe_at_leaves=mpe_at_leaves
        #     )
        #     unflattend_latents = self.unflatten(latents)

        if training:
            unflattend_latents = self.down_convs(evidence)
        else:
            flattend_latents = self.spn.sample(
                num_samples=num_samples,
                mpe_at_leaves=mpe_at_leaves
            )
            flattend_latents = flattend_latents.squeeze(1) # remove channel dim for unflatten layer
            unflattend_latents = self.unflatten(flattend_latents)
        decoded = self.up_convs(unflattend_latents)
        return self.conv_final(decoded)




class AutoCCLSPN(nn.Module):

    def __init__(
        self,
        in_channels=3,
        # num_classes=1
        depth=5,
        start_filts=64,
        up_mode='transpose',
        spn_depth=3,
        spn_S=5,
        spn_I=5,
        spn_R=5,
        spn_leaf_kwargs={"min_sigma": 1e-2, "max_sigma": 2.0},
        spn_leaf_type=CCRatNormal
    ):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.

        Note:
            AutoCCLSPN uses lazy init for spn so send one full sample/ batch through
            before using this for crops...
        """
        super(AutoCCLSPN, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        # self.num_classes = num_classes

        self.spn = None # lazy init
        # self.flatten = nn.Flatten(start_dim=2) # flatten height and width and channels remain
        self.flatten = nn.Flatten(start_dim=1) # flatten channels, height and width
        self.unflatten = None # lazy init
        self.down_convs = []
        self.up_convs = []

        self.spn_depth = spn_depth
        self.spn_S = spn_S
        self.spn_I = spn_I
        self.spn_R = spn_R
        self.spn_leaf_kwargs = spn_leaf_kwargs
        self.spn_leaf_type = spn_leaf_type

        # create the encoder
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # decoder
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, 1)

        # add the list of modules to current module
        self.down_convs = nn.Sequential(*self.down_convs)
        self.up_convs = nn.Sequential(*self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def lazy_init(self, unflattend_input_size):
        config = EinetConfig(
            num_features=unflattend_input_size[-3] * unflattend_input_size[-2] * unflattend_input_size[-1], #unflattend_input_size[-2] * unflattend_input_size[-1],
            num_channels=1, #unflattend_input_size[1],
            depth=self.spn_depth,
            num_sums=self.spn_S,
            num_leaves=self.spn_I,
            num_repetitions=self.spn_R,
            num_classes=10, # change later
            leaf_kwargs=self.spn_leaf_kwargs,
            leaf_type=self.spn_leaf_type,
            dropout=0.0,
            cross_product=True,
            log_weights=False,
        )
        self.spn = CCLEinet(config)
        # self.unflatten = nn.Unflatten(-1, unflattend_input_size[-2:])
        self.unflatten = nn.Unflatten(-1, unflattend_input_size[-3:])
        self.latent_shape = unflattend_input_size

    def fix_missing_in_last_dim(self, x): # this can be done much better!
        device = self.spn._sampling_root.weights.device
        needed = self.latent_shape[-1] - x.shape[-1]
        nans = torch.ones(x.shape[0], *self.latent_shape[1:-1], needed) * torch.nan
        nans = nans.to(device)
        return torch.cat((x, nans), -1)

    def forward(self, x, y=None):
        with torch.no_grad():
            latents = self.down_convs(x)
        if self.unflatten is None: # init unflatten and spn
            self.lazy_init(latents.shape)
        flattend_latents = self.flatten(latents)
        ll = self.spn(flattend_latents, y)
        return ll

    def sample(self, num_samples=None, mpe_at_leaves=False, evidence=None, training=False, class_index=None):
        marginalized_scopes = None
        if training:
            # with torch.no_grad():
            unflattend_latents = self.down_convs(evidence)
        else:
            flattend_latents = self.spn.sample(
                num_samples=num_samples,
                mpe_at_leaves=mpe_at_leaves,
                class_index=class_index
            )
            flattend_latents = flattend_latents.squeeze(1) # remove channel dim for unflatten layer
            unflattend_latents = self.unflatten(flattend_latents)
        decoded = self.up_convs(unflattend_latents)
        return self.conv_final(decoded)




if __name__ == "__main__":
    """
    testing
    """
    model = AutoSPN(in_channels=3, depth=5)
    x = Variable(torch.FloatTensor(np.random.random((1, 3, 320, 320))))
    ll = model(x)
    ll_sum = ll.sum()
    ll_sum.backward()
    print(ll_sum.item())

    x_crop = x[...,160:]
    sample = model.sample(evidence=x)
    mae = torch.abs(sample - x).sum()
    mae.backward()
    print(mae.item())