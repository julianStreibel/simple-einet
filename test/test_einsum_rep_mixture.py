import torch


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


def test_sampling()


mixing_layer =
