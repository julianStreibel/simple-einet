from torch import nn
from torch.distributions.categorical import Categorical
import torch

class CustomCategorical(nn.Module):
    """ Implementation of a categorical leave used for a class distribution """

    def __init__(
        self,
        num_classes,
        num_repetitions
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_repetitions = num_repetitions

        self.p = nn.Parameter(torch.randn(1, num_repetitions, num_classes))


    def forward(self, y):

        p = nn.functional.softmax(self.p, dim=1)
        dist = Categorical(p)

        return dist.log_prob(y)
