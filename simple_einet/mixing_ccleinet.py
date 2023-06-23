from dataclasses import dataclass
from typing import List, Sequence, Type

import numpy as np
import torch
from torch import nn

from simple_einet.distributions import CustomCategorical
from simple_einet.mixing_einsum_layer import MixingEinsumLayer
from simple_einet.einsum_layer import EinsumLayer, EinsumMixingLayer
from simple_einet.residual_einsum_layer import ResidualEinsumLayer
from simple_einet.utils import SamplingContext, provide_evidence
from simple_einet.einet import EinetConfig, Einet
from simple_einet.factorized_leaf_layer import CCLFactorizedLeaf


class MixingCCLEinet(Einet):
    """
    Einet with class conditional data leaves and mixing dimension.
    This einet factorizes using the chainrule in its first product node. P(X, Y) = P(X|Y) * P(Y)
    The class conditional data distribution is modelled with class conditinal leaves.
    This SPN is not decomposable in its first product node and for the data marginal needs to eliminate Y.
    """

    def __init__(self, config: EinetConfig):
        """
        Create a RatSpn with class conditional data leaves based on a configuration object.

        Args:
            config (RatSpnConfig): RatSpn configuration object.
        """
        self.num_classes = config.num_classes
        if config.leaf_kwargs is not None:
            config.leaf_kwargs = {
                **config.leaf_kwargs,
                "num_classes": self.num_classes}
        else:
            config.leaf_kwargs = {
                "num_classes": self.num_classes}

        # setting number of classes to one because we only need one class conditional data dist.
        config.num_classes = 1

        # using build from parent but with different distribution creation
        super().__init__(config)

    def _build_input_distribution(self, num_features_out: int):
        """Construct the input distribution layer."""
        # Cardinality is the size of the region in the last partitions
        num_reps = self.config.num_repetitions * max(1, self.config.num_mixes)
        base_leaf = self.config.leaf_type(
            num_features=self.config.num_features,
            num_channels=self.config.num_channels,
            num_leaves=self.config.num_leaves,
            num_repetitions=num_reps,
            **self.config.leaf_kwargs,
        )

        # if num_mixes > 0 CCLFactorizedLeaf pushes repetitions in mixture dim
        return CCLFactorizedLeaf(
            num_mixes=self.config.num_mixes,
            num_features=base_leaf.out_features,
            num_features_out=num_features_out,
            num_repetitions=self.config.num_repetitions,
            base_leaf=base_leaf,
            independent_colors=True,
            learn_permutations=self.config.learn_permutations,
            tau=self.config.sinkhorn_tau
        )

    def _forward_layers(self, x):
        """
        Forward pass through the inner sum and product layers.

        Args:
            x: Input.

        Returns:
            torch.Tensor: Output of the last layer before the root layer.
        """
        # Forward to inner product and sum layers
        for layer in self.einsum_layers:
            x = layer(x)
        return x

    def _build(self):
        """Construct the internal architecture of the RatSpn with class conditional leaves."""
        # Build the SPN bottom up:
        # Definition from RAT Paper
        # Leaf Region:      Create I leaf nodes
        # Root Region:      Create C sum nodes
        # Internal Region:  Create S sum nodes
        # Partition:        Cross products of all child-regions

        einsum_layers = []

        # building layers top down (starting at the root)
        for i in np.arange(start=1, stop=self.config.depth + 1):

            if i < self.config.depth:
                _num_sums_in = self.config.num_sums
            else:
                _num_sums_in = self.config.num_leaves

            if i > 1:
                _num_mixes_out = self.config.num_mixes
            else:
                _num_mixes_out = 1

            in_features = 2 ** i

            layer = ResidualEinsumLayer(
                num_features=in_features,
                num_sums_in=_num_sums_in,
                num_sums_out=self.config.num_sums,
                num_mixtures_in=self.config.num_mixes,
                num_mixtures_out=_num_mixes_out,
                num_repetitions=self.config.num_repetitions,
                dropout=self.config.dropout,
                sum_dropout=self.config.sum_dropout
            )

            einsum_layers.append(layer)

        # Construct leaf
        self.leaf = self._build_input_distribution(
            num_features_out=einsum_layers[-1].num_features)

        # List layers in a bottom-to-top fashion
        self.einsum_layers: Sequence[EinsumLayer] = nn.ModuleList(
            reversed(einsum_layers))

        # Construct the categorical class distribution with repetitions
        self._class_dist = CustomCategorical(
            self.num_classes,
            self.config.num_sums,
            self.config.num_repetitions
        )

        self._joint_layer = EinsumLayer(
            num_features=2,
            num_sums_in=self.config.num_sums,  # take sums out from last layer
            num_sums_out=1,
            num_repetitions=self.config.num_repetitions,
            dropout=self.config.dropout,
        )

        # If model has multiple reptitions, add repetition mixing layer
        if self.config.num_repetitions > 1:
            self.mixing = EinsumMixingLayer(
                num_features=1,
                num_sums_in=self.config.num_repetitions,
                num_sums_out=1,
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, marginalization_mask: torch.Tensor = None, return_conditional=False) -> torch.Tensor:
        """
        Inference pass for the Class Conditinal Leave Einet model.

        Args:
          x (torch.Tensor): Input data of shape [N, C, D], where C is the number of input channels (useful for images) and D is the number of features/random variables (H*W for images).
          y (torch.Tensor): Input data of shape [N] if given returning only the class conditional likelihood/joint of the given class
          marginalized_scope: torch.Tensor:  (Default value = None)

        Returns:
            Log-likelihood tensor of the input:
            - p(X, y) = p(X | y) * p(y) if in train
            - [p(X, y=i) for i in range(C)] if in eval
        """

        # Add channel dimension if not present
        if x.dim() == 2:  # [N, D]
            x = x.unsqueeze(1)

        if x.dim() == 4:  # [N, C, H, W]
            x = x.view(x.shape[0], self.config.num_channels, -1)

        assert x.dim() == 3
        assert x.shape[1] == self.config.num_channels

        if y is not None:  # if in train -> do one upwards pass
            x = self._one_class_forward(
                x,
                y=y,
                marginalization_mask=marginalization_mask,
                return_conditional=return_conditional
            )
        else:  # if y is None -> do C upwards passes TODO: make this as a batch with artificial classes
            res = list()
            x_copy = x.clone()
            for i in range(self.num_classes):
                y = (torch.ones(x.shape[0]) * i).to(torch.int64).to(x.device)
                res.append(self._one_class_forward(
                    x_copy,
                    y,
                    marginalization_mask=marginalization_mask,
                    return_conditional=return_conditional))
            x = torch.stack(res).T.squeeze(0)

        return x

    def _one_class_forward(self, x: torch.Tensor, y: torch.Tensor, marginalization_mask: torch.Tensor = None, return_conditional=False) -> torch.Tensor:
        """
        Inference pass with one class. p(X, Y=y)
        """

        # check for right shape of y
        # y = y.reshape(1, -1)

        class_conditional_log_likelihood = self.leaf(
            x, y, marginalization_mask)

        # Pass through intermediate layers
        class_conditional_log_likelihood = self._forward_layers(
            class_conditional_log_likelihood)

        # squeeze mixing dim
        class_conditional_log_likelihood = class_conditional_log_likelihood.squeeze(
            3)

        if return_conditional:
            # set class conditionals two ones (zero in log space)
            class_log_likelihood = torch.zeros_like(
                class_conditional_log_likelihood)
        else:
            # add channel dim to class probs
            class_log_likelihood = self._class_dist(
                y).unsqueeze(1)

        # stack class_conditional_log_likelihood and class_log_likelihood
        # in feature dimension, add the and pass thorugh the joint einsum layer
        joint_layer_input = torch.stack(
            (class_conditional_log_likelihood, class_log_likelihood), dim=1).squeeze(2)

        joint_log_likelihood = self._joint_layer(joint_layer_input)

        # If model has multiple reptitions, perform repetition mixing
        if self.config.num_repetitions > 1:
            # Mix repetitions
            joint_log_likelihood = self.mixing(joint_log_likelihood)
        else:
            # Remove repetition index
            joint_log_likelihood = joint_log_likelihood.squeeze(-1)

        batch_size, features, channels = joint_log_likelihood.size()
        assert features == 1  # number of features should be 1 at this point
        assert channels == self.config.num_classes

        # Remove feature dimension
        joint_log_likelihood = joint_log_likelihood.squeeze(1)

        # Final shape check
        assert joint_log_likelihood.shape == (
            batch_size, 1)

        return joint_log_likelihood

    def sample(
        self,
        num_samples: int = None,
        class_index=None,
        evidence: torch.Tensor = None,
        is_mpe: bool = False,
        mpe_at_leaves: bool = False,
        temperature_leaves: float = 1.0,
        temperature_sums: float = 1.0,
        marginalized_scopes: List[int] = None,
    ):
        """
        Draw samples from the ccleinet.
        The ccleinet represents p(x, y) = p(x | y) * p(y)
        First sample y from p(y) and then sample x from p(x | y=y)

        1. sample einsum mixing layer (head) for one repetition if rep > 1
        2. sample joint repetition for indices of p(y) and p(x | y)
        3. sample y from p(y)
        4. sample layers for indices in lower layer
        5. sample leaves of p(x| y=y) for x

        """
        assert class_index is None or evidence is None, "Cannot provide both, evidence and class indices."
        assert (
            num_samples is None or evidence is None
        ), "Cannot provide both, number of samples to generate (num_samples) and evidence."

        # Check if evidence contains nans
        if evidence is not None:
            # Set n to the number of samples in the evidence
            num_samples = evidence.shape[0]
        elif num_samples is None:
            num_samples = 1

        with provide_evidence(self, evidence, marginalized_scopes):
            # Create new sampling context
            ctx = SamplingContext(
                num_samples=num_samples,
                # mixing has only one rep.
                indices_repetition=torch.zeros(
                    (num_samples, 1), dtype=int, device=self._joint_layer.weights.device),
                # mixing only returns one output
                indices_out=torch.zeros(
                    (num_samples, 1), dtype=int, device=self._joint_layer.weights.device),
                is_mpe=is_mpe,
                mpe_at_leaves=mpe_at_leaves,
                temperature_leaves=temperature_leaves,
                temperature_sums=temperature_sums,
                num_repetitions=self.config.num_repetitions,
                evidence=evidence)

            if self.config.num_repetitions > 1:
                ctx = self.mixing.sample(context=ctx)
                # Obtain repetition indices sampled from mixture layer
            ctx.indices_repetition = ctx.indices_out.view(num_samples)
            # Set indices_out to zero tensor because there is only one sum returned from the joint layer
            ctx.indices_out = torch.zeros_like(ctx.indices_out)

            ctx = self._joint_layer.sample(
                num_samples=ctx.num_samples, context=ctx)
            # indices_out is has two features one for class and one for data distributions
            # both have only on sum out so the inidices out needs to be adjusted here
            # this is because the input for the joint is stacked
            # this should allways be zero tensor
            ctx.indices_out = ctx.indices_out[:, 0::2]

            # TODO: is it ok to continue with ctx when y is given??????
            # If class is given, use it as base index
            if class_index is not None:
                if isinstance(class_index, list):
                    y = torch.tensor(
                        class_index, device=self._joint_layer.weights.device).to(torch.int64)
                    num_samples = y.shape[0]
                else:
                    y = torch.empty(
                        size=num_sample, device=self._joint_layer.weights.device)
                    y.fill_(class_index).to(torch.int64)
            else:
                y = self._class_dist.sample(context=ctx).to(torch.int64)

            # create mixing dim in ctx.

            # Sample inner layers in reverse order (starting from topmost)
            for layer in reversed(self.einsum_layers):
                ctx = layer.sample(num_samples=ctx.num_samples, context=ctx)

            # Sample leaf
            samples = self.leaf.sample(y, context=ctx)

            if evidence is not None:
                # First make a copy such that the original object is not changed
                evidence = evidence.clone()
                shape_evidence = evidence.shape
                evidence = evidence.view_as(samples)
                evidence[:, :, marginalized_scopes] = samples[:,
                                                              :, marginalized_scopes]
                evidence = evidence.view(shape_evidence)
                return evidence
            else:
                return samples

    def mpe(
        self,
        evidence: torch.Tensor = None,
        marginalized_scopes: List[int] = None,
    ) -> torch.Tensor:
        """
        Perform MPE given some evidence.

        Args:
            evidence: Input evidence. Must contain some NaN values.
        Returns:
            torch.Tensor: Clone of input tensor with NaNs replaced by MPE estimates.
        """
        return self.sample(evidence=evidence, is_mpe=True, marginalized_scopes=marginalized_scopes)
