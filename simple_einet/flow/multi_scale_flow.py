import torch
from torch import nn

from simple_einet.flow.coupling_layer import CouplingLayer, GatedConvNet
from simple_einet.flow.dequantization import VariationalDequantization
from simple_einet.flow.masking import create_checkerboard_mask, create_channel_mask
import numpy as np


class SqueezeFlow(nn.Module):

    def forward(self, z, ldj, reverse=False):
        B, C, H, W = z.shape
        if not reverse:
            # Forward direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, C, H//2, 2, W//2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4)
            z = z.reshape(B, 4*C, H//2, W//2)
        else:
            # Reverse direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, C//4, 2, 2, H, W)
            z = z.permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(B, C // 4, H * 2, W * 2)

        return z, ldj


class SplitFlow(nn.Module):

    def __init__(self):
        super().__init__()
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z, z_split = z.chunk(2, dim=1)
            ldj += self.prior.log_prob(z_split).sum(dim=[1, 2, 3])
        else:
            z_split = self.prior.sample(
                sample_shape=z.shape).cuda()  # TODO handle device
            z = torch.cat([z, z_split], dim=1)
            ldj -= self.prior.log_prob(z_split).sum(dim=[1, 2, 3])
        return z, ldj


def create_flows(image_shape):
    c_in = image_shape.channels
    flow_layers = []

    # add squeezeFlow until hight and width is 2
    n = int(np.log2(image_shape.height)) - 2

    flow_layers = [SqueezeFlow() for _ in range(n)]

    vardeq_layers = [
        CouplingLayer(
            network=GatedConvNet(
                c_in=2*c_in*4**n,
                c_out=2,
                c_hidden=32
            ),
            mask=create_checkerboard_mask(
                h=4,
                w=4,
                invert=(i % 2 == 1)
            ),
            c_in=1*c_in*4**n
        ) for i in range(4)
    ]

    flow_layers += [VariationalDequantization(vardeq_layers)]

    flow_layers += [
        CouplingLayer(
            network=GatedConvNet(
                c_in=1*c_in*4**n,
                c_out=2,
                c_hidden=32
            ),
            mask=create_checkerboard_mask(
                h=4,
                w=4,
                invert=(i % 2 == 1)
            ),
            c_in=1*c_in*4**n
        ) for i in range(10)
    ]

    # flow_layers += [SqueezeFlow()]
    # for i in range(2):
    #     flow_layers += [CouplingLayer(network=GatedConvNet(c_in=4 * c_in, c_hidden=48),
    #                                   mask=create_channel_mask(
    #                                       c_in=4 * c_in, invert=(i % 2 == 1)),
    #                                   c_in=4 * c_in)]
    # flow_layers += [SplitFlow(),
    #                 SqueezeFlow()]
    # for i in range(4):
    #     flow_layers += [CouplingLayer(network=GatedConvNet(c_in=8 * c_in, c_hidden=64),
    #                                   mask=create_channel_mask(
    #                                       c_in=8 * c_in, invert=(i % 2 == 1)),
    #                                   c_in=8 * c_in)]

    return flow_layers


def backup_create_flows(image_shape):
    c_in = image_shape.channels
    flow_layers = []

    vardeq_layers = [
        CouplingLayer(
            network=GatedConvNet(
                c_in=2 * c_in,
                c_out=2,
                c_hidden=16
            ),
            mask=create_checkerboard_mask(
                h=image_shape.height,
                w=image_shape.width,
                invert=(i % 2 == 1)
            ),
            c_in=1 * c_in
        ) for i in range(4)
    ]

    flow_layers += [VariationalDequantization(vardeq_layers)]

    flow_layers += [
        CouplingLayer(
            network=GatedConvNet(
                c_in=1 * c_in,
                c_hidden=32
            ),
            mask=create_checkerboard_mask(
                h=image_shape.height,
                w=image_shape.width,
                invert=(i % 2 == 1)
            ),
            c_in=1 * c_in
        ) for i in range(4)
    ]

    flow_layers += [SqueezeFlow()]
    for i in range(2):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=4 * c_in, c_hidden=48),
                                      mask=create_channel_mask(
                                          c_in=4 * c_in, invert=(i % 2 == 1)),
                                      c_in=4 * c_in)]
    flow_layers += [SplitFlow(),
                    SqueezeFlow()]
    for i in range(4):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=8 * c_in, c_hidden=64),
                                      mask=create_channel_mask(
                                          c_in=8 * c_in, invert=(i % 2 == 1)),
                                      c_in=8 * c_in)]

    return flow_layers
