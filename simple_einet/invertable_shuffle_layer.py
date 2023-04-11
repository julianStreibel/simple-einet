from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from simple_einet.layers import AbstractLayer

class InvertableFeatureShuffleLayer(nn.Module):
    """ This layer suffles in the feature dim. It is expected that the feature dim is the last in x. """
    def __init__(self):
        super().__init__()
        self.idx = None

    def forward(self, x):
        if self.idx is None:
            self.idx = torch.randperm(x.size(-1))
        return x[..., self.idx]

    def inv(self, x):
        res = torch.zeros_like(x)
        res[..., self.idx] = x
        return res

