import torch
import numpy as np


def test_mixing():

    # this represents a sum layer output
    # shape: features, sums_reps, repetitions
    # here: 4, 2, 3
    output_einsum_layer = torch.tensor([
        [
            [
                [1, 2, 1],
                [1, 1, 1],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
            ],
        ],
        [
            [
                [1, 2, 1],
                [1, 1, 1],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
            ],
        ]
    ]).float()

    # this represents the mixing weights for repetitions
    # shape: features, sums_reps, repetitions (i for in), repetitions (o for out)
    # here: 4, 2, 3, 3
    rep_mixing_weights = torch.tensor(
        [
            [
                [
                    [0.2, 0.2, 0.6],
                    [0.6, 0.2, 0.2],
                    [0.2, 0.6, 0.2],
                ],
                [
                    [0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2],
                    [0.6, 0.6, 0.6],
                ],
            ],
            [
                [
                    [0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2],
                    [0.6, 0.6, 0.6],
                ],
                [
                    [0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2],
                    [0.6, 0.6, 0.6],
                ],
            ],
            [
                [
                    [0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2],
                    [0.6, 0.6, 0.6],
                ],
                [
                    [0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2],
                    [0.6, 0.6, 0.6],
                ],
            ],
            [
                [
                    [0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2],
                    [0.6, 0.6, 0.6],
                ],
                [
                    [0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2],
                    [0.6, 0.6, 0.6],
                ],
            ],
        ]
    )

    rep_mixing_weights = torch.nn.functional.softmax(rep_mixing_weights, -2)
    # sum out repetitions_in
    res = torch.einsum("fsio,bfsi->bfso",
                       rep_mixing_weights, output_einsum_layer)
    print(res)


def test_sum_dropout():

    num_features_out = 4
    num_sums_out = 2
    num_mixtures_out = 3
    num_repetitions = 5
    num_mixtures_in = 4

    sum_dropout = 0.1
    sum_bernoulli_dist = torch.distributions.Bernoulli(
        probs=sum_dropout)

    probs = torch.randn(
        12,  # batch
        num_features_out,
        num_sums_out,
        num_mixtures_in,
        num_repetitions,
    )

    rep_mixing_weights = torch.randn(
        num_features_out,
        num_sums_out,
        num_mixtures_out,
        num_repetitions,
        num_mixtures_in,
    )

    og_mixing_weights = torch.nn.Parameter(rep_mixing_weights)

    dropout_indices = sum_bernoulli_dist.sample(og_mixing_weights.shape).bool()
    ninf = torch.zeros_like(og_mixing_weights)

    indices_all_dropout = dropout_indices.all(
        dim=-1, keepdim=True).expand(-1, -1, -1, -1, num_mixtures_in)
    replacement = torch.nn.functional.one_hot(
        torch.rand_like(rep_mixing_weights).argmax(dim=-1), num_mixtures_in).bool()
    dropout_indices = torch.where(
        indices_all_dropout, replacement, dropout_indices)

    ninf[dropout_indices] = np.NINF

    rep_mixing_weights = og_mixing_weights + ninf
    # rep_mixing_weights[dropout_indices] = np.NINF

    # problem: all components get set to -inf of one sum
    # solution: when all component probs are 0 set one to 1
    # 1. check for all nan in last dim and find indices
    # 2.

    rep_mixing_weights = torch.nn.functional.softmax(rep_mixing_weights, -1)

    # rep_mixing_weights[rep_mixing_weights.isnan()] = 0.0

    probs_max = torch.max(probs, dim=3, keepdim=True)[0]
    probs = torch.exp(probs - probs_max)

    out = torch.einsum("ndsir,dsori->ndsor", probs, rep_mixing_weights)

    out = torch.log(out) + probs_max
    out.mean().backward()

    breakpoint()


test_sum_dropout()
