from torch import nn
from torch.distributions.categorical import Categorical
import torch

from simple_einet.utils import SamplingContext


class CustomCategorical(nn.Module):
    """ Implementation of a categorical leave used for a class distribution """

    def __init__(
        self,
        num_classes,
        num_leaves,
        num_repetitions
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_leaves = num_leaves
        self.num_repetitions = num_repetitions

        # self.p = nn.Parameter(torch.randn(num_leaves, num_repetitions, num_classes))
        self.p = nn.Parameter(torch.randn(
            1, num_leaves, num_repetitions, num_classes))

    def forward(self, y, batch_size=None):

        if y is None:
            return torch.zeros(batch_size, self.num_leaves, self.num_repetitions).cuda()
        p = nn.functional.softmax(self.p, dim=3)
        dist = Categorical(p)
        y = y.reshape(-1, 1, 1)  # reshape for broadcast
        return dist.log_prob(y)

    def sample(self, context: SamplingContext = None):
        p = nn.functional.softmax(self.p, dim=3)

        if context.is_mpe or context.mpe_at_leaves:
            return p[context.indices_repetition].argmax(dim=3)

        dist = Categorical(p[context.indices_repetition])
        return dist.sample()

    def extra_repr(self):
        return "num_classes={}, num_leaves={}, num_repetitions={}".format(
            self.num_classes, self.num_leaves, self.num_repetitions
        )
