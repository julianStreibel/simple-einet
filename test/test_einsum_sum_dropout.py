import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


def test_sum_dropout():

    num_features = 4
    num_sums = 5
    num_reps = 4

    x = torch.rand(
        24,
        num_features,
        num_sums,
        num_reps
    )

    N, D, C, R = x.size()
    D_out = D // 2

    ws = torch.randn(
        num_features // 2,
        num_sums,
        num_reps,
        num_sums,
        num_sums,
    )

    og_weights = nn.Parameter(ws)

    sum_dropout = 0.9999
    sum_bernoulli_dist = torch.distributions.Bernoulli(
        probs=sum_dropout)

    # Get left and right partition probs
    left = x[:, 0::2]
    right = x[:, 1::2]

    # Prepare for LogEinsumExp trick (see paper for details)
    left_max = torch.max(left, dim=2, keepdim=True)[0]
    left_prob = torch.exp(left - left_max)
    right_max = torch.max(right, dim=2, keepdim=True)[0]
    right_prob = torch.exp(right - right_max)

    # Project weights into valid space
    weights = og_weights
    weights = weights.view(D_out, num_sums, num_reps, -1)

    #########

    # get dropout indices
    dropout_indices = sum_bernoulli_dist.sample(weights.shape).bool()

    # check for sums where all components drop
    indices_all_dropout = dropout_indices.all(
        dim=-1, keepdim=True).expand(weights.shape)

    replacement = ~torch.nn.functional.one_hot(torch.rand_like(
        weights).argmax(dim=-1), num_sums * num_sums).bool()
    dropout_indices = torch.where(
        indices_all_dropout, replacement, dropout_indices)

    ninf = torch.zeros_like(weights)
    ninf[dropout_indices] = np.NINF

    weights = weights + ninf

    #########

    weights = F.softmax(weights, dim=-1)
    weights = weights.view(og_weights.shape)

    # Einsum operation for sum(product(x))
    # n: batch, i: left-channels, j: right-channels, d:features, o: output-channels, r: repetitions
    prob = torch.einsum("ndir,ndjr,dorij->ndor",
                        left_prob, right_prob, weights)

    t_prob = left_prob * right_prob  # * weights

    # LogEinsumExp trick, re-add the max
    prob = torch.log(prob) + left_max + right_max

    prob.mean().backward()

    breakpoint()


test_sum_dropout()
