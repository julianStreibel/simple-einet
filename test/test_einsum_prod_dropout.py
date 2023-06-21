import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


def test_prod_dropout():
    """
    The idea is to dropout a hole product node.
    For that we need to set all component weights of this product to 0 = softmax(ninf).
    1. Sampling product nodes to dropout
        1.1. We have num_sums * num_sums products in every repetition
        1.2. Take binomial b(n=num_sums*num_sums, p=dropout_rate) and sample how many products to dropout nd
        1.3. Take product indices, shuffle them and take first nd indices to dropout and do this for every rep
    2. Dropout
        2.1. We create a vecotr ninf with shape num_sums, num_reps, num_sums * num_sums (leaving out the feautre dim for now)
        2.2. In every dim of out_sum (0) we take the same indices of dim 2 that we fill with ninf but different per reps (dim 1)
        2.3. Add the ninf vector to the weights setting some component weights to ninf

    """

    num_features = 4
    num_sums = 3
    num_reps = 2

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

    sum_dropout = 0.4
    sum_bernoulli_dist = torch.distributions.binomial.Binomial(
        num_sums**2 - 1, sum_dropout)
    n_dropout = sum_bernoulli_dist.sample().int().item()

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

    dropout_idxs = weights[:, :1].argsort(
    )[..., :n_dropout]  # select idx to dropout
    ninf = torch.zeros_like(weights[:, :1])  # dropout weights
    ninf.scatter_(-1, dropout_idxs, np.NINF)  # set dropout weights
    ninf = ninf.expand(-1, num_sums, -1, -1)  # broadcast in sum_out dim
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

    print(weights)


test_prod_dropout()