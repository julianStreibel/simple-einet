from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from simple_einet.layers import AbstractLayer

class InvertableShuffleLayer(nn.Module):
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


class ShufflePerRepetitionLayer(nn.Module):
    """ 
    This layer suffles in the feature dim and creates different permutations for everey repetition.
    It is expected that the feature dim is the last in x and x without the repetition dimension.
    """
    def __init__(self, R):
        super().__init__()
        self.idx = None
        self.R = R

    def forward(self, x):
        if self.idx is None:
            # create different permutations for every repetition
            self.idx = torch.randn(x.size(-1), self.R).argsort(dim=1).to(x.device)
        x = x.unsqueeze(-1).expand(-1, -1, -1, self.R).gather(-1, self.idx.expand(*x.shape[:2], -1, -1))
        x = x.unsqueeze(3)
        return x

    def inv(self, x):
        raise NotImplementedError("Inverse shuffelling is not implemented jet.")
