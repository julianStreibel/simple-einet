from collections import defaultdict
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Type

import numpy as np
import torch
from fast_pytorch_kmeans import KMeans
from torch import nn

from simple_einet.distributions import AbstractLeaf, RatNormal, truncated_normal_, CustomCategorical
from simple_einet.mixing_class_einsum_layer import MixingClassEinsumLayer
from simple_einet.einsum_layer import ClassEinsumMixingLayer, EinsumLayer, EinsumMixingLayer
from simple_einet.factorized_leaf_layer import ClassFactorizedLeaf
from simple_einet.layers import Sum
from simple_einet.type_checks import check_valid
from simple_einet.utils import SamplingContext, provide_evidence
from simple_einet.einet import EinetConfig

logger = logging.getLogger(__name__)


class MixingClassEinet(nn.Module):

    def __init__(self, config: EinetConfig):
        """
        Create a RatSpn based on a configuration object.

        Args:
            config (RatSpnConfig): RatSpn configuration object.
        """
        super().__init__()
        config.assert_valid()
        self.config = config

        # Construct the architecture
        self._build()

        # Initialize weights
        self._init_weights()

    def forward(self, x: torch.Tensor, marginalization_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Inference pass for the Einet model.

        Args:
          x (torch.Tensor): Input data of shape [N, C, D], where C is the number of input channels (useful for images) and D is the number of features/random variables (H*W for images).
          marginalized_scope: torch.Tensor:  (Default value = None)

        Returns:
            Log-likelihood tensor of the input: p(X) or p(X | C) if number of classes > 1.
        """
        # Add channel dimension if not present
        if x.dim() == 2:  # [N, D]
            x = x.unsqueeze(1)

        if x.dim() == 4:  # [N, C, H, W]
            x = x.view(x.shape[0], self.config.num_channels, -1)

        assert x.dim() == 3
        assert x.shape[1] == self.config.num_channels

        # repeat for the classes
        x = (
            x
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, -1,
                    self.config.num_leaves,
                    self.config.num_mixes,
                    self.config.num_classes
                    )
        )
        # Apply leaf distributions (replace marginalization indicators with 0.0 first)
        x = self.leaf(x, marginalization_mask)

        # remove mixing dim
        x = x.squeeze(3)

        # Pass through intermediate layers
        x = self._forward_layers(x)

        # Merge results from the different repetitions into the channel dimension
        batch_size, features, sums, mixtures, repetitions, num_classes = x.size()
        x = x.reshape(batch_size, features, sums,
                      mixtures * repetitions, num_classes)
        assert features == 1  # number of features should be 1 at this point
        assert sums == 1
        # If model has multiple reptitions, perform repetition mixing
        if self.config.num_repetitions > 1:
            # Mix repetitions
            x = self.mixing(x)
        else:
            # Remove repetition index
            x = x.squeeze(-1)

        # Remove feature, mixing, rep and sum dimension
        x = x.squeeze(1).squeeze(1).squeeze(1).squeeze(1)

        # Final shape check
        assert x.shape == (batch_size, self.config.num_classes)

        return x

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
        """Construct the internal architecture of the RatSpn."""
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
                _num_sums_out = self.config.num_sums
            else:
                _num_sums_out = 1

            if i < self.config.depth:
                _num_sums_in = self.config.num_sums
            else:
                _num_sums_in = self.config.num_leaves

            if i > 1:
                _num_sums_out = self.config.num_sums
                _num_mixes_out = self.config.num_mixes
            else:
                _num_sums_out = 1
                _num_mixes_out = 1

            in_features = 2**i

            layer = MixingClassEinsumLayer(
                num_features=in_features,
                num_sums_in=_num_sums_in,
                num_sums_out=_num_sums_out,
                num_repetitions=self.config.num_repetitions,
                num_mixtures_in=self.config.num_mixes,
                num_mixtures_out=_num_mixes_out,
                num_hidden_mixtures=self.config.num_hidden_mixtures,
                mixing_depth=self.config.mixing_depth,
                num_classes=self.config.num_classes,
                dropout=self.config.dropout,
            )

            einsum_layers.append(layer)

        # Construct leaf
        self.leaf = self._build_input_distribution(
            num_features_out=einsum_layers[-1].num_features)

        # List layers in a bottom-to-top fashion
        self.einsum_layers: Sequence[EinsumLayer] = nn.ModuleList(
            reversed(einsum_layers))

        # If model has multiple reptitions, add repetition mixing layer
        if self.config.num_repetitions > 1:
            self.mixing = ClassEinsumMixingLayer(
                num_features=1,
                num_sums_in=self.config.num_repetitions,
                num_sums_out=1,
                num_classes=self.config.num_classes
            )

        # Construct sampling root with weights according to priors for sampling
        self._sampling_root = Sum(
            num_sums_in=self.config.num_classes,
            num_features=1,
            num_sums_out=1,
            num_repetitions=1,
        )
        self._sampling_root.weights = nn.Parameter(
            torch.ones(size=(1, self.config.num_classes, 1, 1)) *
            torch.tensor(1 / self.config.num_classes),
            requires_grad=False,
        )

    def _build_input_distribution(self, num_features_out: int):
        """Construct the input distribution layer."""
        # Cardinality is the size of the region in the last partitions
        num_reps = self.config.num_repetitions * max(1, self.config.num_mixes)
        base_leaf = self.config.leaf_type(
            num_features=self.config.num_features,
            num_channels=self.config.num_channels,
            num_leaves=self.config.num_leaves,
            num_repetitions=num_reps,
            conditional=False,
            num_classes=self.config.num_classes,
            **self.config.leaf_kwargs,
        )

        return ClassFactorizedLeaf(
            num_features=base_leaf.out_features,
            num_features_out=num_features_out,
            num_repetitions=self.config.num_repetitions,
            num_classes=self.config.num_classes,
            num_mixes=self.config.num_mixes,
            base_leaf=base_leaf,
        )

    @ property
    def __device(self):
        """Small hack to obtain the current device."""
        return self._sampling_root.weights.device

    def _init_weights(self):
        """Initiale the weights. Calls `_init_weights` on all modules that have this method."""
        for module in self.modules():
            # if hasattr(module, "_init_weights") and module != self:
            #     module._init_weights()
            if isinstance(module, EinsumMixingLayer):
                truncated_normal_(module.weights, std=0.5)
            elif isinstance(module, EinsumLayer):
                truncated_normal_(module.weights, std=0.5)
            elif isinstance(module, Sum):
                truncated_normal_(module.weights, std=0.5)
            elif isinstance(module, RatNormal):
                truncated_normal_(module.stds, std=0.1)

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
        raise NotImplementedError
        """
        Sample from the distribution represented by this SPN.

        Possible valid inputs:

        - `num_samples`: Generates `num_samples` samples.
        - `num_samples` and `class_index (int)`: Generates `num_samples` samples from P(X | C = class_index).
        - `class_index (List[int])`: Generates `len(class_index)` samples. Each index `c_i` in `class_index` is mapped
            to a sample from P(X | C = c_i)
        - `evidence`: If evidence is given, samples conditionally and fill NaN values.

        Args:
            num_samples: Number of samples to generate.
            class_index: Class index. Can be either an int in combination with a value for `num_samples` which will result in `num_samples`
                samples from P(X | C = class_index). Or can be a list of ints which will map each index `c_i` in the
                list to a sample from P(X | C = c_i).
            evidence: Evidence that can be provided to condition the samples. If evidence is given, `num_samples` and
                `class_index` must be `None`. Evidence must contain NaN values which will be imputed according to the
                distribution represented by the SPN. The result will contain the evidence and replace all NaNs with the
                sampled values.
            is_mpe: Flag to perform max sampling (MPE).
            mpe_at_leaves: Flag to perform mpe only at leaves.
            temperature_leaves: Variance scaling for leaf distribution samples.
            temperature_sums: Variance scaling for sum node categorical sampling.

        Returns:
            torch.Tensor: Samples generated according to the distribution specified by the SPN.

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
            # If class is given, use it as base index
            if class_index is not None:
                if isinstance(class_index, list):
                    indices = torch.tensor(
                        class_index, device=self.__device).view(-1, 1)
                    num_samples = indices.shape[0]
                else:
                    indices = torch.empty(
                        size=(num_samples, 1), device=self.__device)
                    indices.fill_(class_index)

                # Create new sampling context
                ctx = SamplingContext(
                    num_samples=num_samples,
                    indices_out=indices,
                    indices_repetition=None,
                    is_mpe=is_mpe,
                    mpe_at_leaves=mpe_at_leaves,
                    temperature_leaves=temperature_leaves,
                    temperature_sums=temperature_sums,
                    num_repetitions=self.config.num_repetitions,
                    evidence=evidence,
                )
            else:
                # Start sampling one of the C root nodes TODO: check what happens if C=1
                ctx = SamplingContext(
                    num_samples=num_samples,
                    is_mpe=is_mpe,
                    mpe_at_leaves=mpe_at_leaves,
                    temperature_leaves=temperature_leaves,
                    temperature_sums=temperature_sums,
                    num_repetitions=self.config.num_repetitions,
                    evidence=evidence,
                )
                ctx = self._sampling_root.sample(context=ctx)

            # Save parent indices that were sampled from the sampling root
            if self.config.num_repetitions > 1:
                indices_out_pre_root = ctx.indices_out

                ctx.indices_repetition = torch.zeros(
                    num_samples, dtype=int, device=self.__device)
                ctx = self.mixing.sample(context=ctx)

                # Obtain repetition indices
                ctx.indices_repetition = ctx.indices_out.view(num_samples)
                ctx.indices_out = indices_out_pre_root

            # Sample inner layers in reverse order (starting from topmost)
            for layer in reversed(self.einsum_layers):
                ctx = layer.sample(num_samples=ctx.num_samples, context=ctx)

            # Sample leaf
            samples = self.leaf.sample(context=ctx)

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

    def sample_differentiable(
        self,
        num_samples: int = None,
        class_index=None,
        evidence: torch.Tensor = None,
        is_mpe: bool = False,
        temperature_leaves: float = 1.0,
        temperature_sums: float = 1.0,
        marginalized_scopes: List[int] = None,
        hard=False,
        tau=1.0,
        mpe_at_leaves=False,
    ) -> torch.Tensor:
        """
        Sample from the distribution represented by this SPN.

        Possible valid inputs:

        - `num_samples`: Generates `num_samples` samples.
        - `num_samples` and `class_index (int)`: Generates `num_samples` samples from P(X | C = class_index).
        - `class_index (List[int])`: Generates `len(class_index)` samples. Each index `c_i` in `class_index` is mapped
            to a sample from P(X | C = c_i)
        - `evidence`: If evidence is given, samples conditionally and fill NaN values.

        Args:
            num_samples: Number of samples to generate.
            class_index: Class index. Can be either an int in combination with a value for `num_samples` which will result in `num_samples`
                samples from P(X | C = class_index). Or can be a list of ints which will map each index `c_i` in the
                list to a sample from P(X | C = c_i).
            evidence: Evidence that can be provided to condition the samples. If evidence is given, `num_samples` and
                `class_index` must be `None`. Evidence must contain NaN values which will be imputed according to the
                distribution represented by the SPN. The result will contain the evidence and replace all NaNs with the
                sampled values.
            is_mpe: Flag to perform max sampling (MPE).
            temperature_leaves: Variance scaling for leaf distribution samples.
            temperature_sums: Variance scaling for sum node categorical sampling.
            marginalized_scopes: List of scope indices to be marginalized.
            hard: Flag to perform hard sampling.
            tau: Temperature for categorical sampling.

        Returns:
            torch.Tensor: Samples generated according to the distribution specified by the SPN.

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

        with provide_evidence(self, evidence, marginalized_scopes, requires_grad=True):
            # If class is given, use it as base index
            # Start sampling one of the C root nodes TODO: check what happens if C=1
            ctx = SamplingContext(
                num_samples=num_samples,
                is_mpe=is_mpe,
                temperature_leaves=temperature_leaves,
                temperature_sums=temperature_sums,
                num_repetitions=self.config.num_repetitions,
                evidence=evidence,
                is_differentiable=True,
                hard=hard,
                tau=tau,
                mpe_at_leaves=mpe_at_leaves,
            )
            # ctx = self._sampling_root.sample(context=ctx)
            ctx.indices_out = torch.ones(
                num_samples,
                1,
                1,
                dtype=torch.float,
                device=self.__device,
                requires_grad=True,
            )

            # Save parent indices that were sampled from the sampling root
            if self.config.num_repetitions > 1:
                indices_out_pre_root = ctx.indices_out

                ctx.indices_repetition = torch.ones(
                    num_samples, 1, dtype=torch.float, device=self.__device)
                ctx = self.mixing.sample(context=ctx)

                # Obtain repetition indices
                ctx.indices_repetition = ctx.indices_out.view(
                    num_samples, self.config.num_repetitions)
                ctx.indices_out = indices_out_pre_root
            else:
                ctx.indices_repetition = torch.ones(
                    num_samples,
                    1,
                    dtype=torch.float,
                    device=self.__device,
                    requires_grad=True,
                )

            # Sample inner layers in reverse order (starting from topmost)
            for layer in reversed(self.einsum_layers):
                ctx = layer.sample(num_samples=ctx.num_samples, context=ctx)

            # Sample leaf
            samples = self.leaf.sample(context=ctx)

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

    def extra_repr(self) -> str:
        return f"{self.config}"


class EinetMixture(nn.Module):
    def __init__(self, n_components: int, einet_config: EinetConfig):
        super().__init__()
        self.n_components = check_valid(
            n_components, expected_type=int, lower_bound=1)
        self.config = einet_config

        einets = []

        for i in range(n_components):
            einets.append(Einet(einet_config))

        self.einets: Sequence[Einet] = nn.ModuleList(einets)
        self._kmeans = KMeans(n_clusters=self.n_components,
                              mode="euclidean", verbose=1)
        self.mixture_weights = nn.Parameter(
            torch.empty(n_components), requires_grad=False)
        self.centroids = nn.Parameter(torch.empty(
            n_components, einet_config.num_features), requires_grad=False)

    @ torch.no_grad()
    def initialize(self, data: torch.Tensor):
        data = data.float()  # input has to be [n, d]
        self._kmeans.fit(data.view(data.shape[0], -1))

        self.mixture_weights.data = self._kmeans.num_points_in_clusters / \
            self._kmeans.num_points_in_clusters.sum()
        self.centroids.data = self._kmeans.centroids

    def _predict_cluster(self, x, marginalized_scopes: List[int] = None):
        x = x.view(x.shape[0], -1)  # input needs to be [n, d]
        if marginalized_scopes is not None:
            keep_idx = list(sorted(
                [i for i in range(self.config.num_features) if i not in marginalized_scopes]))
            centroids = self.centroids[:, keep_idx]
            x = x[:, keep_idx]
        else:
            centroids = self.centroids
        return self._kmeans.max_sim(a=x.float(), b=centroids)[1]

    def _separate_data_by_cluster(self, x: torch.Tensor, marginalized_scope: List[int]):
        cluster_idxs = self._predict_cluster(x, marginalized_scope).tolist()

        separated_data = defaultdict(list)
        separated_idxs = defaultdict(list)
        for data_idx, cluster_idx in enumerate(cluster_idxs):
            separated_data[cluster_idx].append(x[data_idx])
            separated_idxs[cluster_idx].append(data_idx)

        return separated_idxs, separated_data

    def forward(self, x, marginalized_scope: torch.Tensor = None):
        assert self._kmeans is not None, "EinetMixture has not been initialized yet."

        separated_idxs, separated_data = self._separate_data_by_cluster(
            x, marginalized_scope)

        lls_result = []
        data_idxs_all = []
        for cluster_idx, data_list in separated_data.items():
            data_tensor = torch.stack(data_list, dim=0)
            lls = self.einets[cluster_idx](data_tensor)

            data_idxs = separated_idxs[cluster_idx]
            for data_idx, ll in zip(data_idxs, lls):
                lls_result.append(ll)
                data_idxs_all.append(data_idx)

        # Sort results into original order as observed in the batch
        L = [(data_idxs_all[i], i) for i in range(len(data_idxs_all))]
        L.sort()
        _, permutation = zip(*L)
        permutation = torch.tensor(permutation, device=x.device).view(-1)
        lls_result = torch.stack(lls_result)
        lls_sorted = lls_result[permutation]

        return lls_sorted

    def sample(
        self,
        num_samples: int = None,
        num_samples_per_cluster: int = None,
        class_index=None,
        evidence: torch.Tensor = None,
        is_mpe: bool = False,
        temperature_leaves: float = 1.0,
        temperature_sums: float = 1.0,
        marginalized_scopes: List[int] = None,
    ):
        assert num_samples is None or num_samples_per_cluster is None
        if num_samples is None and num_samples_per_cluster is not None:
            num_samples = num_samples_per_cluster * self.n_components

        # Check if evidence contains nans
        if evidence is not None:
            # Set n to the number of samples in the evidence
            num_samples = evidence.shape[0]
        elif num_samples is None:
            num_samples = 1

        if is_mpe:
            # Take cluster idx with largest weights
            cluster_idxs = [self.mixture_weights.argmax().item()]
        else:
            if num_samples_per_cluster is not None:
                cluster_idxs = torch.arange(self.n_components).repeat_interleave(
                    num_samples_per_cluster).tolist()
            else:
                # Sample from categorical over weights
                cluster_idxs = (
                    torch.distributions.Categorical(
                        probs=self.mixture_weights).sample((num_samples,)).tolist()
                )

        if evidence is None:
            # Sample without evidence
            separated_idxs = defaultdict(int)
            for cluster_idx in cluster_idxs:
                separated_idxs[cluster_idx] += 1

            samples_all = []
            for cluster_idx, num_samples_cluster in separated_idxs.items():
                samples = self.einets[cluster_idx].sample(
                    num_samples_cluster,
                    class_index=class_index,
                    evidence=evidence,
                    is_mpe=is_mpe,
                    temperature_leaves=temperature_leaves,
                    temperature_sums=temperature_sums,
                    marginalized_scopes=marginalized_scopes,
                )
                samples_all.append(samples)

            samples = torch.cat(samples_all, dim=0)
        else:
            # Sample with evidence
            separated_idxs, separated_data = self._separate_data_by_cluster(
                evidence, marginalized_scopes)

            samples_all = []
            evidence_idxs_all = []
            for cluster_idx, evidence_pre_cluster in separated_data.items():
                evidence_per_cluster = torch.stack(evidence_pre_cluster, dim=0)
                samples = self.einets[cluster_idx].sample(
                    evidence=evidence_per_cluster,
                    is_mpe=is_mpe,
                    temperature_leaves=temperature_leaves,
                    temperature_sums=temperature_sums,
                    marginalized_scopes=marginalized_scopes,
                )

                evidence_idxs = separated_idxs[cluster_idx]
                for evidence_idx, sample in zip(evidence_idxs, samples):
                    samples_all.append(sample)
                    evidence_idxs_all.append(evidence_idx)

            # Sort results into original order as observed in the batch
            L = [(evidence_idxs_all[i], i)
                 for i in range(len(evidence_idxs_all))]
            L.sort()
            _, permutation = zip(*L)
            permutation = torch.tensor(
                permutation, device=evidence.device).view(-1)
            samples_all = torch.stack(samples_all)
            samples_sorted = samples_all[permutation]
            samples = samples_sorted

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
