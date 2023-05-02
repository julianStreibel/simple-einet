from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from simple_einet.layers import AbstractLayer
from simple_einet.type_checks import check_valid
from simple_einet.utils import SamplingContext, index_one_hot, diff_sample_one_hot


class MixingEinsumLayer(AbstractLayer):
    def __init__(
        self,
        num_features: int,
        num_sums_in: int,
        num_sums_out: int,
        num_mixtures_in: int,
        num_mixtures_out: int,
        num_repetitions: int,
        dropout: float = 0.0,
    ):
        super().__init__(num_features, num_repetitions=num_repetitions)

        self.num_sums_in = check_valid(num_sums_in, int, 1)
        self.num_sums_out = check_valid(num_sums_out, int, 1)
        self.num_mixtures_in = check_valid(num_mixtures_in, int, 1)
        self.num_mixtures_out = check_valid(num_mixtures_out, int, 1)
        self.num_repetitions = check_valid(num_repetitions, int, 1)
        cardinality = 2  # Fixed to binary graphs for now
        self.cardinality = check_valid(cardinality, int, 2, num_features + 1)
        self.num_features_out = np.ceil(
            self.num_features / self.cardinality).astype(int)

        mixing_weights = torch.randn(
            self.num_features_out,
            self.num_sums_out,
            self.num_mixtures_out,
            self.num_repetitions,
            self.num_mixtures_in,
        )
        self.mixing_weights = nn.Parameter(mixing_weights)

        einsum_weights = torch.randn(
            self.num_features_out,
            self.num_sums_out,
            self.num_mixtures_in,
            self.num_repetitions,
            self.num_sums_in,
            self.num_sums_in,
        )

        self.einsum_weights = nn.Parameter(einsum_weights)

        # Dropout
        self.dropout = check_valid(
            dropout, expected_type=float, lower_bound=0.0, upper_bound=1.0)
        self._bernoulli_dist = torch.distributions.Bernoulli(
            probs=self.dropout)

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache_left = None
        self._input_cache_right = None
        self._input_cache = None

        self.out_shape = f"(N, {self.num_features_out}, {self.num_sums_out}, {self.num_mixtures_out}, {self.num_repetitions})"

    def forward(self, x):
        # Apply dropout: remove random sum component
        if self.dropout > 0.0 and self.training:
            # dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            # x[dropout_indices] = np.NINF
            pass

        # Dimensions
        # N: batch size
        # F: Features
        # S: Sums
        # M: Mixtures
        # R: Repetitions
        N, D, S, M, R = x.size()

        D_out = D // 2

        # Get left and right partition probs
        left = x[:, 0::2]
        right = x[:, 1::2]

        # Prepare for LogEinsumExp trick (see paper for details)
        left_max = torch.max(left, dim=2, keepdim=True)[0]  # TODO: dim=4?!
        left_prob = torch.exp(left - left_max)
        right_max = torch.max(right, dim=2, keepdim=True)[0]
        right_prob = torch.exp(right - right_max)

        # Project weights into valid space
        einsum_weights = self.einsum_weights
        einsum_weights = einsum_weights.view(
            D_out, self.num_sums_out, self.num_mixtures_in, self.num_repetitions, -1)
        # TODO: Is this normalizing the weights for each sum or also over all sums?
        einsum_weights = F.softmax(einsum_weights, dim=-1)
        einsum_weights = einsum_weights.view(self.einsum_weights.shape)

        # Einsum operation for sum(product(x))
        # n: batch, i: left-channels, j: right-channels, d:features, o: output-channels, m: mixtures, r: repetitions
        prob = torch.einsum("ndimr,ndjmr,domrij->ndomr",
                            left_prob, right_prob, einsum_weights)

        # LogEinsumExp trick, re-add the max
        prob = torch.log(prob) + left_max + right_max

        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache_left = left
            self._input_cache_right = right
            self._input_cache = prob.clone()

        probs_max = torch.max(prob, dim=3, keepdim=True)[0]
        probs = torch.exp(prob - probs_max)

        mixing_weights = self.mixing_weights
        mixing_weights = F.softmax(mixing_weights, dim=-1)
        # n: batch, d: features, s: sums,  i: mixtures in, o: mixtures out, r: repetitions
        out = torch.einsum("ndsir,dsori->ndsor", probs, mixing_weights)
        lls = torch.log(out) + probs_max

        return lls

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> Union[SamplingContext, torch.Tensor]:
        """
        Sample from this layer.
        Args:
            num_samples: Number of samples.
            context: Sampling context.

        Returns:
            torch.Tensor: Generated samples.
        """
        pass

    def extra_repr(self):
        return "num_features={}, num_sums_in={}, num_sums_out={}, num_mixtures_in={}, num_mixtures_out={}, num_repetitions={}, out_shape={}, " "einsum_weights_shape={} mixing_weights_shape={}".format(
            self.num_features,
            self.num_sums_in,
            self.num_sums_out,
            self.num_mixtures_in,
            self.num_mixtures_out,
            self.num_repetitions,
            self.out_shape,
            self.einsum_weights.shape,
            self.mixing_weights.shape,
        )
