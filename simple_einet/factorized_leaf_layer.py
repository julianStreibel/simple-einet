from typing import List
import torch
import numpy as np

from simple_einet.utils import SamplingContext, index_one_hot
from simple_einet.layers import AbstractLayer
from simple_einet.distributions import AbstractLeaf
from simple_einet.distributions.categorical import CustomCategorical
from simple_einet.permutation_layer import PermutationLayer


class FactorizedLeaf(AbstractLayer):
    """
    A 'meta'-leaf layer that combines multiple scopes of a base-leaf layer via naive factorization.

    Attributes:
        base_leaf: Base leaf layer that contains the actual leaf distribution.
        in_features: Number of input features/RVs.
        out_features: Number of output features/RVs. This determines the factorization group size (round(in_features / out_features))
        scopes: One-hot mapping from which in_features correspond to which out_features.

    """

    def __init__(
        self,
        num_features: int,
        num_features_out: int,
        num_repetitions,
        base_leaf: AbstractLeaf,
        learn_permutations=False,
        independent_colors=True,
        shuffle_features=True,
        num_mixes=0,
        tau=1
    ):
        """
        Args:
            in_features (int): Number of input features/RVs.
            out_features (int): Number of output features/RVs.
            num_repetitions (int): Number of repetitions.
            base_leaf (Leaf): Base leaf distribution object.
        """

        super().__init__(num_features, num_repetitions=num_repetitions)

        self.base_leaf = base_leaf
        self.num_features_out = num_features_out
        self.num_mixes = num_mixes
        self.learn_permutations = learn_permutations
        self.tau = tau
        self.independent_colors = independent_colors
        if not independent_colors:
            self.num_features *= self.base_leaf.num_channels

        # Size of the factorized groups of RVs
        cardinality = int(np.round(self.num_features / self.num_features_out))

        # Construct mapping of scopes from in_features -> out_features
        scopes = torch.zeros(
            self.num_features, self.num_features_out, max(1, num_mixes), self.num_repetitions)
        for r in range(self.num_repetitions):
            if shuffle_features:
                idxs = torch.randperm(n=self.num_features)
            else:
                idxs = torch.arange(self.num_features)
            for o in range(self.num_features_out):
                low = o * cardinality
                high = (o + 1) * cardinality
                if o == self.num_features_out - 1:
                    high = self.num_features
                scopes[idxs[low:high], o, :, r] = 1

        self.register_buffer("scopes", scopes)

        if self.learn_permutations:
            self.permutation_layer = PermutationLayer(
                self.num_features, self.num_repetitions)

    def forward(self, x: torch.Tensor, marginalized_scopes: List[int], tau=None):
        # Forward through base leaf
        x = self.base_leaf(x, marginalized_scopes)

        if self.independent_colors:
            # Factorize input channels
            x = x.sum(dim=1)  # removing channel dim
            x = x.view(
                x.shape[0],  # b
                x.shape[1],  # f
                x.shape[2],  # s
                max(1, self.num_mixes),  # m
                self.num_repetitions  # r
            )
        else:
            # create mixture dim and flattend channels in features
            x = x.view(
                x.shape[0],
                x.shape[1] * x.shape[2],
                x.shape[3],
                max(1, self.num_mixes),
                self.num_repetitions
            )

        x = x.view(
            x.shape[0],  # b
            x.shape[1],  # f
            x.shape[2],  # s
            max(1, self.num_mixes),  # m
            self.num_repetitions  # r
        )

        if self.learn_permutations:
            t = tau if tau is not None else self.tau
            x = self.permutation_layer(x, t)

        # Merge scopes by naive factorization
        x = torch.einsum("bicmr,iomr->bocmr", x, self.scopes)

        if self.num_mixes == 0:
            x = x.squeeze(3)
            assert x.shape == (
                x.shape[0],
                self.num_features_out,
                self.base_leaf.num_leaves,
                self.num_repetitions,
            )
        else:
            assert x.shape == (
                x.shape[0],
                self.num_features_out,
                self.base_leaf.num_leaves,
                self.num_mixes,
                self.num_repetitions,
            )
        return x

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        # Save original indices_out and set context indices_out to none, such that the out_channel
        # are not filtered in the base_leaf sampling procedure
        indices_out = context.indices_out
        context.indices_out = None
        samples = self.base_leaf.sample(context=context)
        num_samples = samples.shape[0]

        # Check that shapes match as expected
        if samples.dim() == 4:
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.num_channels,
                self.num_features,
                self.base_leaf.num_leaves,
            )
        elif samples.dim() == 5:
            assert self.num_features == samples.shape[1]
            assert hasattr(self.base_leaf, "cardinality")
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.out_features,
                self.base_leaf.cardinality,
                self.base_leaf.num_leaves,
            )

        if not context.is_differentiable:

            # TODO: fix this for more mixture dims
            scopes = self.scopes
            if self.scopes.shape[2] == 1:
                scopes = scopes.squeeze(2)

            scopes = scopes[...,
                            context.indices_repetition].permute(2, 0, 1)
            rnge_in = torch.arange(self.num_features_out,
                                   device=samples.device)
            scopes = (scopes * rnge_in).sum(-1).long()
            indices_in_gather = indices_out.gather(dim=1, index=scopes)
            indices_in_gather = indices_in_gather.view(num_samples, 1, -1, 1)

            indices_in_gather = indices_in_gather.expand(
                -1, samples.shape[1], -1, -1)
            samples = samples.gather(dim=-1, index=indices_in_gather)
            samples.squeeze_(-1)  # Remove num_leaves dimension
        else:
            # i_rep = context.indices_repetition.argmax(-1).long()
            # i_out = indices_out.argmax(-1).long()
            # scopes_orig = self.scopes[..., i_rep].permute(2, 0, 1)
            # rnge_in = torch.arange(self.num_features_out, device=samples.device)
            # scopes_orig = (scopes_orig * rnge_in).sum(-1).long()
            # indices_in_gather_orig = i_out.gather(dim=1, index=scopes_orig)
            # indices_in_gather_orig = indices_in_gather_orig.view(num_samples, 1, -1, 1)
            #
            # indices_in_gather_orig = indices_in_gather_orig.expand(-1, samples.shape[1], -1, -1)
            # samples_orig = samples.gather(dim=-1, index=indices_in_gather_orig)
            # samples_orig.squeeze_(-1)  # Remove num_leaves dimension

            scopes = self.scopes.unsqueeze(0)  # make space for batch dim
            r_idx = context.indices_repetition.view(
                context.num_samples, 1, 1, -1)
            scopes = index_one_hot(scopes, index=r_idx, dim=-1)

            indices_in = index_one_hot(indices_out.unsqueeze(
                1), index=scopes.unsqueeze(-1), dim=2)

            indices_in = indices_in.unsqueeze(1)  # make space for channel dim
            samples = index_one_hot(samples, index=indices_in, dim=-1)

            # assert (samples - samples_orig).sum() < 1e-4

        return samples

    def extra_repr(self):
        return f"num_features={self.num_features}, num_features_out={self.num_features_out}"


class CCLFactorizedLeaf(FactorizedLeaf):
    """
    A 'meta'-leaf layer that combines multiple scopes of a base-leaf layer via naive factorization and supports class conditional leaves.

    Attributes:
        base_leaf: Base leaf layer that contains the actual leaf distribution.
        in_features: Number of input features/RVs.
        out_features: Number of output features/RVs. This determines the factorization group size (round(in_features / out_features))
        scopes: One-hot mapping from which in_features correspond to which out_features.

    """

    def __init__(
        self,
        num_mixes=0,
        independent_colors=True,
        learn_permutations=False,
        tau=1,
        **args,
    ):
        """
        Args:
            in_features (int): Number of input features/RVs.
            out_features (int): Number of output features/RVs.
            num_repetitions (int): Number of repetitions.
            base_leaf (Leaf): Base leaf distribution object.
        """

        super().__init__(**args)
        self.independent_colors = independent_colors
        if not independent_colors:
            self.num_features *= self.base_leaf.num_channels
        # Size of the factorized groups of RVs
        cardinality = int(np.round(self.num_features / self.num_features_out))
        self.num_mixes = num_mixes
        self.learn_permutations = learn_permutations
        self.tau = tau

        # Construct mapping of scopes from in_features -> out_features
        scopes = torch.zeros(
            self.num_features, self.num_features_out, max(1, num_mixes), self.num_repetitions)
        for r in range(self.num_repetitions):
            idxs = torch.randperm(n=self.num_features)
            for o in range(self.num_features_out):
                low = o * cardinality
                high = (o + 1) * cardinality
                if o == self.num_features_out - 1:
                    high = self.num_features
                scopes[idxs[low:high], o, :, r] = 1

        self.register_buffer("scopes", scopes)

        if self.learn_permutations:
            self.permutation_layer = PermutationLayer(
                self.num_features, self.num_repetitions)

    def forward(self, x: torch.Tensor, y: torch.Tensor, marginalized_scopes: List[int]):
        # Forward through base leaf
        x = self.base_leaf(x, y, marginalized_scopes)

        if self.independent_colors:
            # Factorize input channels
            x = x.sum(dim=1)  # removing channel dim
            x = x.view(
                x.shape[0],  # b
                x.shape[1],  # f
                x.shape[2],  # s
                max(1, self.num_mixes),  # m
                self.num_repetitions  # r
            )
        else:
            # create mixture dim and push channels in feature dim
            x = x.view(
                x.shape[0],
                x.shape[1] * x.shape[2],
                x.shape[3],
                max(1, self.num_mixes),
                self.num_repetitions
            )

        if self.learn_permutations:
            x = self.permutation_layer(x, self.tau)

        # Merge scopes by naive factorization
        x = torch.einsum("bicmr,iomr->bocmr", x, self.scopes)

        if self.num_mixes == 0:
            x = x.squeeze(3)
            assert x.shape == (
                x.shape[0],
                self.num_features_out,
                self.base_leaf.num_leaves,
                self.num_repetitions,
            )
        else:
            assert x.shape == (
                x.shape[0],
                self.num_features_out,
                self.base_leaf.num_leaves,
                self.num_mixes,
                self.num_repetitions,
            )
        return x

    def sample(self, y, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        # Save original indices_out and set context indices_out to none, such that the out_channel
        # are not filtered in the base_leaf sampling procedure
        indices_out = context.indices_out
        context.indices_out = None
        samples = self.base_leaf.sample(y, context=context)
        num_samples = samples.shape[0]

        # Check that shapes match as expected
        if samples.dim() == 4:
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.num_channels,
                self.num_features,
                self.base_leaf.num_leaves,
            )

        elif samples.dim() == 5:
            raise NotImplementedError
            assert self.num_features == samples.shape[1]
            assert hasattr(self.base_leaf, "cardinality")
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.out_features,
                self.base_leaf.cardinality,
                self.base_leaf.num_leaves,
            )

        # create mixture dim
        samples = samples.view(
            context.num_samples,
            self.base_leaf.num_channels,
            self.num_features,
            -1
        )

        if not context.is_differentiable:
            # bring batch size to the front
            scopes = self.scopes[...,
                                 context.indices_repetition].permute(3, 0, 1, 2)
            scopes = scopes.squeeze(-1)
            rnge_in = torch.arange(self.num_features_out,
                                   device=samples.device)
            scopes = (scopes * rnge_in).sum(-1).long()
            indices_in_gather = indices_out.gather(dim=1, index=scopes)
            indices_in_gather = indices_in_gather.view(num_samples, 1, -1, 1)

            indices_in_gather = indices_in_gather.expand(
                -1, samples.shape[1], -1, -1)
            samples = samples.gather(dim=-1, index=indices_in_gather)
            samples.squeeze_(-1)  # Remove num_leaves dimension
        else:
            raise NotImplementedError

        return samples

    def extra_repr(self):
        return f"num_features={self.num_features}, num_features_out={self.num_features_out}, num_mixtures={self.num_mixes}, num_repetitions={self.num_repetitions}"


class ClassFactorizedLeaf(AbstractLayer):
    """
    A 'meta'-leaf layer that combines multiple scopes of a base-leaf layer via naive factorization.

    Attributes:
        base_leaf: Base leaf layer that contains the actual leaf distribution.
        in_features: Number of input features/RVs.
        out_features: Number of output features/RVs. This determines the factorization group size (round(in_features / out_features))
        scopes: One-hot mapping from which in_features correspond to which out_features.

    """

    def __init__(
        self,
        num_features: int,
        num_features_out: int,
        num_repetitions: int,
        num_classes: int,
        base_leaf: AbstractLeaf,
        learn_permutations=False,
        independent_colors=True,
        shuffle_features=True,
        num_mixes=0,
        tau=1
    ):
        """
        Args:
            in_features (int): Number of input features/RVs.
            out_features (int): Number of output features/RVs.
            num_repetitions (int): Number of repetitions.
            base_leaf (Leaf): Base leaf distribution object.
        """

        super().__init__(num_features, num_repetitions=num_repetitions)

        self.base_leaf = base_leaf
        self.num_features_out = num_features_out
        self.num_mixes = num_mixes
        self.learn_permutations = learn_permutations
        self.tau = tau
        self.independent_colors = independent_colors
        self.num_classes = num_classes
        if not independent_colors:
            self.num_features *= self.base_leaf.num_channels

        # Size of the factorized groups of RVs
        cardinality = int(np.round(self.num_features / self.num_features_out))

        # Construct mapping of scopes from in_features -> out_features
        scopes = torch.zeros(
            self.num_features, self.num_features_out, max(1, num_mixes), self.num_repetitions, self.num_classes)
        for r in range(self.num_repetitions):
            if shuffle_features:
                idxs = torch.randperm(n=self.num_features)
            else:
                idxs = torch.arange(self.num_features)
            for o in range(self.num_features_out):
                low = o * cardinality
                high = (o + 1) * cardinality
                if o == self.num_features_out - 1:
                    high = self.num_features
                scopes[idxs[low:high], o, :, r] = 1

        self.register_buffer("scopes", scopes)

        if self.learn_permutations:
            self.permutation_layer = PermutationLayer(
                self.num_features, self.num_repetitions)

    def forward(self, x: torch.Tensor, marginalized_scopes: List[int], tau=None):
        # Forward through base leaf
        x = self.base_leaf(x, marginalized_scopes)

        if self.independent_colors:
            # Factorize input channels
            x = x.sum(dim=1)  # removing channel dim
            x = x.view(
                x.shape[0],  # b
                x.shape[1],  # f
                x.shape[2],  # s
                max(1, self.num_mixes),
                self.num_repetitions,
                self.num_classes
            )
        else:
            # create mixture dim and flattend channels in features
            x = x.view(
                x.shape[0],
                x.shape[1] * x.shape[2],
                x.shape[3],
                max(1, self.num_mixes),
                self.num_repetitions,
                self.num_classes
            )

        if self.learn_permutations:
            t = tau if tau is not None else self.tau
            x = self.permutation_layer(x, t)

        # Merge scopes by naive factorization
        x = torch.einsum("bifmrc,iomrc->bofmrc", x, self.scopes)

        assert x.shape == (
            x.shape[0],
            self.num_features_out,
            self.base_leaf.num_leaves,
            max(1, self.num_mixes),
            self.num_repetitions,
            self.num_classes
        )
        return x

    def sample(self, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        raise NotImplementedError
        # Save original indices_out and set context indices_out to none, such that the out_channel
        # are not filtered in the base_leaf sampling procedure
        indices_out = context.indices_out
        context.indices_out = None
        samples = self.base_leaf.sample(context=context)
        num_samples = samples.shape[0]

        # Check that shapes match as expected
        if samples.dim() == 4:
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.num_channels,
                self.num_features,
                self.base_leaf.num_leaves,
            )
        elif samples.dim() == 5:
            assert self.num_features == samples.shape[1]
            assert hasattr(self.base_leaf, "cardinality")
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.out_features,
                self.base_leaf.cardinality,
                self.base_leaf.num_leaves,
            )

        if not context.is_differentiable:

            # TODO: fix this for more mixture dims
            scopes = self.scopes
            if self.scopes.shape[2] == 1:
                scopes = scopes.squeeze(2)

            scopes = scopes[...,
                            context.indices_repetition].permute(2, 0, 1)
            rnge_in = torch.arange(self.num_features_out,
                                   device=samples.device)
            scopes = (scopes * rnge_in).sum(-1).long()
            indices_in_gather = indices_out.gather(dim=1, index=scopes)
            indices_in_gather = indices_in_gather.view(num_samples, 1, -1, 1)

            indices_in_gather = indices_in_gather.expand(
                -1, samples.shape[1], -1, -1)
            samples = samples.gather(dim=-1, index=indices_in_gather)
            samples.squeeze_(-1)  # Remove num_leaves dimension
        else:
            # i_rep = context.indices_repetition.argmax(-1).long()
            # i_out = indices_out.argmax(-1).long()
            # scopes_orig = self.scopes[..., i_rep].permute(2, 0, 1)
            # rnge_in = torch.arange(self.num_features_out, device=samples.device)
            # scopes_orig = (scopes_orig * rnge_in).sum(-1).long()
            # indices_in_gather_orig = i_out.gather(dim=1, index=scopes_orig)
            # indices_in_gather_orig = indices_in_gather_orig.view(num_samples, 1, -1, 1)
            #
            # indices_in_gather_orig = indices_in_gather_orig.expand(-1, samples.shape[1], -1, -1)
            # samples_orig = samples.gather(dim=-1, index=indices_in_gather_orig)
            # samples_orig.squeeze_(-1)  # Remove num_leaves dimension

            scopes = self.scopes.unsqueeze(0)  # make space for batch dim
            r_idx = context.indices_repetition.view(
                context.num_samples, 1, 1, -1)
            scopes = index_one_hot(scopes, index=r_idx, dim=-1)

            indices_in = index_one_hot(indices_out.unsqueeze(
                1), index=scopes.unsqueeze(-1), dim=2)

            indices_in = indices_in.unsqueeze(1)  # make space for channel dim
            samples = index_one_hot(samples, index=indices_in, dim=-1)

            # assert (samples - samples_orig).sum() < 1e-4

        return samples

    def extra_repr(self):
        return f"num_features={self.num_features}, num_features_out={self.num_features_out}, num_classes={self.num_classes}"


class FactorizedLeafPlusCategorical(FactorizedLeaf):
    """
    A 'meta'-leaf layer that combines multiple scopes of a base-leaf layer via naive factorization and supports class conditional leaves.

    Attributes:
        base_leaf: Base leaf layer that contains the actual leaf distribution.
        in_features: Number of input features/RVs.
        out_features: Number of output features/RVs. This determines the factorization group size (round(in_features / out_features))
        scopes: One-hot mapping from which in_features correspond to which out_features.

    """

    def __init__(
        self,
        num_mixes=0,
        independent_colors=True,
        learn_permutations=False,
        tau=1,
        num_classes=1,
        **args,
    ):
        """
        Args:
            in_features (int): Number of input features/RVs.
            out_features (int): Number of output features/RVs.
            num_repetitions (int): Number of repetitions.
            base_leaf (Leaf): Base leaf distribution object.
        """

        super().__init__(**args)
        self.independent_colors = independent_colors
        self.num_features += 1  # add class feature
        if not independent_colors:
            self.num_features *= self.base_leaf.num_channels
        # Size of the factorized groups of RVs
        cardinality = int(np.round(self.num_features / self.num_features_out))
        self.num_mixes = num_mixes
        self.learn_permutations = learn_permutations
        self.tau = tau
        self.num_classes = num_classes

        # Construct mapping of scopes from in_features -> out_features
        scopes = torch.zeros(
            self.num_features, self.num_features_out, max(1, num_mixes), self.num_repetitions)
        for r in range(self.num_repetitions):
            idxs = torch.randperm(n=self.num_features)
            for o in range(self.num_features_out):
                low = o * cardinality
                high = (o + 1) * cardinality
                if o == self.num_features_out - 1:
                    high = self.num_features
                scopes[idxs[low:high], o, :, r] = 1

        self.register_buffer("scopes", scopes)

        if self.learn_permutations:
            self.permutation_layer = PermutationLayer(
                self.num_features + 1, self.num_repetitions)

        self.categorical = CustomCategorical(
            self.num_classes,
            self.base_leaf.num_leaves,
            self.base_leaf.num_repetitions
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, marginalized_scopes: List[int], tau=None):
        # Forward through base leaf
        x = self.base_leaf(x, marginalized_scopes)
        ll_y = self.categorical(y, batch_size=x.shape[0])

        # add mixture to ll_y
        ll_y = ll_y.view(
            ll_y.shape[0],  # batch
            1,  # feature
            ll_y.shape[1],  # sums/leaves
            max(1, self.num_mixes),
            self.num_repetitions,  # repetitions

        )

        if self.independent_colors:
            # Factorize input channels
            x = x.sum(dim=1)  # removing channel dim
            x = x.view(
                x.shape[0],  # b
                x.shape[1],  # f
                x.shape[2],  # s
                max(1, self.num_mixes),  # m
                self.num_repetitions  # r
            )
        else:
            # create mixture dim and push channels in feature dim
            x = x.view(
                x.shape[0],
                x.shape[1] * x.shape[2],
                x.shape[3],
                max(1, self.num_mixes),
                self.num_repetitions
            )

        # add p(y) into x
        x = torch.cat((x, ll_y), dim=1)

        if self.learn_permutations:
            t = tau if tau is not None else self.tau
            x = self.permutation_layer(x, t)

        # Merge scopes by naive factorization
        x = torch.einsum("bicmr,iomr->bocmr", x, self.scopes)

        if self.num_mixes == 0:
            x = x.squeeze(3)
            assert x.shape == (
                x.shape[0],
                self.num_features_out,
                self.base_leaf.num_leaves,
                self.num_repetitions,
            )
        else:
            assert x.shape == (
                x.shape[0],
                self.num_features_out,
                self.base_leaf.num_leaves,
                self.num_mixes,
                self.num_repetitions,
            )
        return x

    def sample(self, y, num_samples: int = None, context: SamplingContext = None) -> torch.Tensor:
        # Save original indices_out and set context indices_out to none, such that the out_channel
        # are not filtered in the base_leaf sampling procedure
        indices_out = context.indices_out
        context.indices_out = None
        samples = self.base_leaf.sample(y, context=context)
        num_samples = samples.shape[0]

        # Check that shapes match as expected
        if samples.dim() == 4:
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.num_channels,
                self.num_features,
                self.base_leaf.num_leaves,
            )

        elif samples.dim() == 5:
            raise NotImplementedError
            assert self.num_features == samples.shape[1]
            assert hasattr(self.base_leaf, "cardinality")
            assert samples.shape == (
                context.num_samples,
                self.base_leaf.out_features,
                self.base_leaf.cardinality,
                self.base_leaf.num_leaves,
            )

        # create mixture dim
        samples = samples.view(
            context.num_samples,
            self.base_leaf.num_channels,
            self.num_features,
            -1
        )

        if not context.is_differentiable:
            # bring batch size to the front
            scopes = self.scopes[...,
                                 context.indices_repetition].permute(3, 0, 1, 2)
            scopes = scopes.squeeze(-1)
            rnge_in = torch.arange(self.num_features_out,
                                   device=samples.device)
            scopes = (scopes * rnge_in).sum(-1).long()
            indices_in_gather = indices_out.gather(dim=1, index=scopes)
            indices_in_gather = indices_in_gather.view(num_samples, 1, -1, 1)

            indices_in_gather = indices_in_gather.expand(
                -1, samples.shape[1], -1, -1)
            samples = samples.gather(dim=-1, index=indices_in_gather)
            samples.squeeze_(-1)  # Remove num_leaves dimension
        else:
            raise NotImplementedError

        return samples

    def extra_repr(self):
        return f"num_features={self.num_features}, num_features_out={self.num_features_out}, num_mixtures={self.num_mixes}, num_repetitions={self.num_repetitions}"
