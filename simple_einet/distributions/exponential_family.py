import torch
from torch.nn.functional import multilabel_soft_margin_loss

# from utils import one_hot
from typing import Dict, List


class ExponentialFamilyArray(torch.nn.Module):
    """
    ExponentialFamilyArray computes log-densities of exponential families in parallel. ExponentialFamilyArray is
    abstract and needs to be derived, in order to implement a concrete exponential family.

    The main use of ExponentialFamilyArray is to compute the densities for FactorizedLeafLayer, which computes products
    of densities over single RVs. All densities over single RVs are computed in parallel via ExponentialFamilyArray.

    Note that when we talk about single RVs, these can in fact be multi-dimensional. A natural use-case is RGB image
    data: it is natural to consider pixels as single RVs, which are, however, 3-dimensional vectors each.

    Although ExponentialFamilyArray is not derived from class Layer, it implements a similar interface. It is intended
    that ExponentialFamilyArray is a helper class for FactorizedLeafLayer, which just forwards calls to the Layer
    interface.

    Best to think of ExponentialFamilyArray as an array of log-densities, of shape array_shape, parallel for each RV.
    When evaluated, it returns a tensor of shape (batch_size, num_features, *array_shape) -- for each sample in the batch and
    each RV, it evaluates an array of array_shape densities, each with their own parameters. Here, num_features is the number
    of random variables, i.e. the size of the set (boldface) X in the paper.

    The boolean use_em indicates if we want to use the on-board EM algorithm (alternatives would be SGD, Adam,...).

    After the ExponentialFamilyArray has been generated, we need to initialize it. There are several options for
    initialization (see also method initialize(...) below):
        'default': use the default initializer (to be written in derived classes).
        Tensor: provide a custom initialization.

    In order to implement a concrete exponential family, we need to derive this class and implement

        sufficient_statistics(self, x)
        log_normalizer(self, theta)
        log_h(self, x)

        expectation_to_natural(self, phi)
        default_initializer(self)
        project_params(self, params)
        reparam(self, params)
        _sample(self, *args, **kwargs)

    Please see docstrings of these functions below, for further details.
    """

    def __init__(self, num_features, num_channels, num_leaves, num_repetitions, num_stats, use_em=True):
        """
        :param num_features: number of random variables (int)
        :param num_channels: dimensionality of random variables (int)
        :param array_shape: shape of log-probability tensor, (tuple of ints)
                            log-probability tensor will be of shape (batch_size, num_features,) + array_shape
        :param num_stats: number of sufficient statistics of exponential family (int)
        :param use_em: use internal EM algorithm? (bool)
        """
        super(ExponentialFamilyArray, self).__init__()

        self.num_features = num_features
        self.num_channels = num_channels
        self.num_leaves = num_leaves
        self.num_repetitions = num_repetitions
        self.num_stats = num_stats
        self.params_shape = (num_features, num_leaves,
                             num_repetitions, num_stats)

        self.params = None
        self.ll = None
        self.suff_stats = None

        self._use_em = use_em
        self._p_acc = None
        self._stats_acc = None
        self._online_em_frequency = None
        self._online_em_stepsize = None
        self._online_em_counter = 0

        self.params = torch.nn.Parameter(
            self.default_initializer())  # theta (not eta on wikipedia)
        self.params.leaf = self

    # --------------------------------------------------------------------------------
    # The following functions need to be implemented to specify an exponential family.

    def sufficient_statistics(self, x):
        """
        The sufficient statistics function for the implemented exponential family (called T(x) in the paper).

        :param x: observed data (Tensor).
                  If self.num_channels == 1, this can be either of shape (batch_size, self.num_features, 1) or
                  (batch_size, self.num_features).
                  If self.num_channels > 1, this must be of shape (batch_size, self.num_features, self.num_channels).
        :return: sufficient statistics of the implemented exponential family (Tensor).
                 Must be of shape (batch_size, self.num_features, self.num_channels, self.num_stats)
        """
        raise NotImplementedError

    def log_normalizer(self, theta):
        """
        Log-normalizer of the implemented exponential family (called A(theta) in the paper).

        :param theta: natural parameters (Tensor). Must be of shape (self.num_features, *self.array_shape, self.num_stats).
        :return: log-normalizer (Tensor). Must be of shape (self.num_features, *self.array_shape).
        """
        raise NotImplementedError

    def log_h(self, x):
        """
        The log of the base measure (called h(x) in the paper).

        :param x: observed data (Tensor).
                  If self.num_channels == 1, this can be either of shape (batch_size, self.num_features, 1) or
                  (batch_size, self.num_features).
                  If self.num_channels > 1, this must be of shape (batch_size, self.num_features, self.num_channels).
        :return: log(h) of the implemented exponential family (Tensor).
                 Can either be a scalar or must be of shape (batch_size, self.num_features)
        """
        raise NotImplementedError

    def expectation_to_natural(self, phi):
        """
        Conversion from expectations parameters phi to natural parameters theta, for the implemented exponential
        family.

        :param phi: expectation parameters (Tensor). Must be of shape (self.num_features, *self.array_shape, self.num_stats).
        :return: natural parameters theta (Tensor). Same shape as phi.
        """
        raise NotImplementedError

    def default_initializer(self):
        """
        Default initializer for params.

        :return: initial parameters for the implemented exponential family (Tensor).
                 Must be of shape (self.num_features, *self.array_shape, self.num_stats)
        """
        raise NotImplementedError

    def project_params(self):
        """
        Project onto parameters' constraint set.

        Exponential families are usually defined on a constrained domain, e.g. the second parameter of a Gaussian needs
        to be non-negative. The EM algorithm takes the parameters sometimes out of their domain. This function projects
        them back onto their domain.
        """
        raise NotImplementedError

    def reparam(self, params):
        """
        Re-parameterize parameters, in order that they stay in their constrained domain.

        When we are not using the EM, we need to transform unconstrained (real-valued) parameters to the constrained set
        of the expectation parameter. This function should return such a function (i.e. the return value should not be
        a projection, but a function which does the projection).

        :param params: un-constrained parameters.
        :return: re-parametrized parameters.
        """
        raise NotImplementedError

    def _sample(self, num_samples, params, differentiable=False, **kwargs):
        """
        Helper function for sampling the exponential family.

        :param num_samples: number of samples to be produced
        :param params: expectation parameters (phi) of the exponential family, of shape
                       (self.num_sigma, *self.array_shape, self.num_stats)
        :param kwargs: keyword arguments
               Depending on the implementation, kwargs can also contain further arguments.
        :return: i.i.d. samples of the exponential family (Tensor).
                 Should be of shape (num_samples, self.num_sigma, self.num_channels, *self.array_shape)
        """
        raise NotImplementedError

    def _argmax(self, params, differentiable=False, **kwargs):
        """
        Helper function for getting the argmax of the exponential family.

        :param params: expectation parameters (phi) of the exponential family, of shape
                       (self.num_sigma, *self.array_shape, self.num_stats)
        :param kwargs: keyword arguments
               Depending on the implementation, kwargs can also contain further arguments.
        :return: argmax of the exponential family (Tensor).
                 Should be of shape (self.num_sigma, self.num_channels, *self.array_shape)
        """
        raise NotImplementedError

    # --------------------------------------------------------------------------------

    def forward(self, x, marginalized_scopes=None):
        """
        Evaluates the exponential family, in log-domain. For a single log-density we would compute
            log_h(X) + <params, T(X)> + A(params)
        Here, we do this in parallel and compute an array of log-densities of shape array_shape, for each sample in the
        batch and each RV.

        :param x: input data (Tensor).
                  If self.num_channels == 1, this can be either of shape (batch_size, self.num_features, 1) or
                  (batch_size, self.num_features).
                  If self.num_channels > 1, this must be of shape (batch_size, self.num_features, self.num_channels).
        :return: log-densities of implemented exponential family (Tensor).
                 Will be of shape (batch_size, self.num_features, *self.array_shape)
        """
        x = x.permute(0, 2, 1)

        if self._use_em:
            with torch.no_grad():
                phi = self.expectation_to_natural(self.params)
        else:
            raise NotImplementedError("use normal leaves without em")

        self.suff_stats = self.sufficient_statistics(x)

        # reshape for broadcasting
        # suff_stats: (batch_size, features, leaves, reps, channels, suff_stats)
        self.suff_stats = (
            self.suff_stats
            .unsqueeze(2)  # num leaves
            .unsqueeze(2)  # num reps
            .expand(
                -1, -1, self.num_leaves, -1, -1, -1
            )
        )

        # log_normalizer: (self.num_features, *self.array_shape)
        log_normalizer = self.log_normalizer(phi)
        log_normalizer = log_normalizer.unsqueeze(
            0)

        # log_h: scalar, or (batch_size, self.num_features)
        log_h = self.log_h(x)
        if len(log_h.shape) > 0:
            # reshape for broadcasting
            log_h = log_h.reshape(
                log_h.shape[0:2] + (1, 1))

        # compute the exponential family tensor
        # (batch_size, self.num_features, *self.array_shape)

        self.ll = (
            log_h + (phi.unsqueeze(0) * self.suff_stats).sum(-1) -
            log_normalizer
        )

        if marginalized_scopes is not None:
            self.ll.data[:, marginalized_scopes] = 0.0

        if self._use_em:
            # EM needs the gradient with respect to self.ll
            self.ll.requires_grad_()

        outputs = self.ll
        # move channel dim
        outputs = outputs.permute(0, 4, 1, 2, 3)
        return outputs

    def sample(self, num_samples=1, differentiable=False, **kwargs):
        if self._use_em:
            params = self.params
        else:
            with get_context(differentiable):
                params = self.reparam(self.params)
        return self._sample(
            num_samples=num_samples,
            params=params,
            differentiable=differentiable,
            **kwargs
        )

    def argmax(self, differentiable=False, **kwargs):
        if self._use_em:
            params = self.params
        else:
            with get_context(differentiable):
                params = self.reparam(self.params)
        return self._argmax(params=params, differentiable=differentiable, **kwargs)

    def em_set_hyperparams(self, online_em_frequency, online_em_stepsize, purge=True):
        """Set new setting for online EM."""
        if purge:
            self.em_purge()
            self._online_em_counter = 0
        self._online_em_frequency = online_em_frequency
        self._online_em_stepsize = online_em_stepsize

    def em_purge(self):
        """Discard em statistics."""
        if self.ll is not None and self.ll.grad is not None:
            self.ll.grad.zero_()
        self._p_acc = None
        self._stats_acc = None

    def em_process_batch(self):
        """
        Accumulate EM statistics of current batch. This should typically be called via EinsumNetwork.em_process_batch().
        """
        if not self._use_em:
            raise AssertionError(
                "em_process_batch called while _use_em==False.")
        if self.params is None:
            return

        with torch.no_grad():
            p = self.ll.grad
            weighted_stats = (p.unsqueeze(-1) * self.suff_stats).sum(0)
            p = p.sum(0)

            if self._p_acc is None:
                self._p_acc = torch.zeros_like(p)
            self._p_acc += p

            if self._stats_acc is None:
                self._stats_acc = torch.zeros_like(weighted_stats)
            self._stats_acc += weighted_stats

            self.ll.grad.zero_()

            if self._online_em_frequency is not None:
                self._online_em_counter += 1
                if self._online_em_counter == self._online_em_frequency:
                    self.em_update()
                    self._online_em_counter = 0

    def em_update(self, stepsize: int, marginalized_scopes: torch.Tensor = None):
        """
        Do an EM update step.

        Note that scopes that were marginalized during the forward pass are indicated by marginalized_scopes.
        All parameters that correspond to these scopes will be ignored for the update step.

        Args:
            stepsize: Learning rate.
            marginalized_scopes: Scopes which were marginalized during the forward pass.
        """
        if not self._use_em:
            raise AssertionError("em_update called while _use_em==False.")

        with torch.no_grad():

            if marginalized_scopes is not None:
                # Store params data
                params_tmp = self.params.data[marginalized_scopes]

            p = self.ll.grad
            weighted_stats = (p.unsqueeze(-1) * self.suff_stats).sum(0)
            p = p.sum(0)

            self.ll.grad.zero_()

            self.params.data = (1.0 - stepsize) * self.params + stepsize * (
                weighted_stats / (p.unsqueeze(-1) + 1e-12)
            )

            if marginalized_scopes is not None:
                # Restore params where marginalization happened
                self.params.data[marginalized_scopes] = params_tmp

            # Project params
            self.params.data = self.project_params(self.params.data)

    def set_marginalization_idx(self, idx):
        """Set indicices of marginalized variables."""
        self.marginalization_idx = idx

    def get_marginalization_idx(self):
        """Set indicices of marginalized variables."""
        return self.marginalization_idx


def shift_last_axis_to(x, i):
    """This takes the last axis of tensor x and inserts it at position i"""
    num_axes = len(x.shape)
    return x.permute(tuple(range(i)) + (num_axes - 1,) + tuple(range(i, num_axes - 1)))


class NormalArray(ExponentialFamilyArray):
    """
    Implementation of Normal distribution.
    Notes:
        params are theta = mu, sigma

    """

    def __init__(
        self,
        num_features,
        num_channels,
        num_leaves,
        num_repetitions,
        min_sigma=0.0001,
        max_sigma=10.0,
        use_em=True,
    ):
        super(NormalArray, self).__init__(
            num_features, num_channels, num_leaves, num_repetitions, 2, use_em=use_em
        )
        self.log_2pi = torch.tensor(1.8378770664093453)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def default_initializer(self):
        phi = torch.rand(self.num_features, self.num_leaves,
                         self.num_repetitions, self.num_channels, 2)

        return phi

    def project_params(self):
        """
        projecting std between self.min_sigma and self.max_sigma
        theta = (mu; sigma)
        """
        self.params[..., 1] = torch.clamp(
            self.params[..., 1], self.min_sigma, self.max_sigma)

    def reparam(self, params):
        raise NotImplementedError("reparam not implemented in NormalArray")

    def sufficient_statistics(self, x):
        return torch.stack((x, x**2), -1)

    def expectation_to_natural(self, theta):
        """
        theta -> eta
        """
        var = theta[..., 1]**2
        eta1 = theta[..., 0] / var
        eta2 = -1.0 / (2.0 * var)
        return torch.stack((eta1, eta2), -1)

    def log_normalizer(self, phi):
        """ A(eta) """
        log_normalizer = - \
            phi[..., 0] ** 2 / (4 * phi[..., 1]) + 0.5 * \
            torch.log((1/(2.0 * phi[..., 1])).abs())
        return log_normalizer

    def log_h(self, x):
        return -0.5 * self.log_2pi

    def _sample(self, num_samples, params, differentiable=False, std_correction=1.0):
        with get_context(differentiable):
            mu = params[..., 0]
            std = params[..., 1]
            shape = (num_samples,) + mu.shape
            samples = mu.unsqueeze(0) + std_correction * std.unsqueeze(0) * torch.randn(
                shape, dtype=mu.dtype, device=mu.device
            )
        return shift_last_axis_to(samples, 2)  # TODO: check this

    def _argmax(self, params, differentiable=False, **kwargs):
        with get_context(differentiable):
            mu = params[..., 0]
        return shift_last_axis_to(mu, 1)  # TODO: check this


# everything down from here needs to be refactored to have the channels in an own dim


class BinomialArray(ExponentialFamilyArray):
    """Implementation of Binomial distribution."""

    def __init__(
        self, num_features, num_channels, num_leaves, num_repetitions, total_count, use_em=True
    ):
        self.total_count = torch.tensor(float(total_count))
        super(BinomialArray, self).__init__(
            num_features, num_channels, num_leaves, num_repetitions, num_channels, use_em=use_em
        )

    def default_initializer(self):
        phi = (
            0.01
            + 0.98 * torch.rand(self.num_features, self.num_leaves,
                                self.num_repetitions, self.num_channels)
        ) * self.total_count
        return phi

    def project_params(self, phi):
        return torch.clamp(phi, 0.0, self.total_count)

    def reparam(self, params):
        return torch.sigmoid(params * 0.1) * float(self.total_count)

    def sufficient_statistics(self, x):
        if len(x.shape) == 2:
            stats = x.unsqueeze(-1)
        elif len(x.shape) == 3:
            stats = x
        else:
            raise AssertionError("Input must be 2 or 3 dimensional tensor.")
        return stats

    def expectation_to_natural(self, phi):
        theta = torch.clamp(phi / self.total_count, 1e-6, 1.0 - 1e-6)
        theta = torch.log(theta) - torch.log(1.0 - theta)
        return theta

    def log_normalizer(self, theta):
        return torch.sum(self.total_count * torch.nn.functional.softplus(theta), -1)

    def log_h(self, x):
        if self.total_count == 1:
            return torch.zeros([], device=x.device)
        else:
            log_h = (
                torch.lgamma(self.total_count + 1.0)
                - torch.lgamma(x + 1.0)
                - torch.lgamma(self.total_count + 1.0 - x)
            )
            if len(x.shape) == 3:
                log_h = log_h.sum(-1)
            return log_h

    def _sample(
        self,
        num_samples,
        params,
        differentiable=False,
        dtype=torch.float32,
        memory_efficient_binomial_sampling=True,
    ):
        with get_context(differentiable):
            params = params / self.total_count
            if memory_efficient_binomial_sampling:
                samples = torch.zeros(
                    (num_samples,) + params.shape, dtype=dtype, device=params.device
                )
                for n in range(int(self.total_count)):
                    rand = torch.rand(
                        (num_samples,) + params.shape, device=params.device
                    )
                    samples += (rand < params).type(dtype)
            else:
                rand = torch.rand(
                    (num_samples,) + params.shape + (int(self.total_count),), device=params.device
                )
                samples = torch.sum(
                    rand < params.unsqueeze(-1), -1).type(dtype)
        return shift_last_axis_to(samples, 2)

    def _argmax(self, params, differentiable=False, dtype=torch.float32):
        with get_context(differentiable):
            params = params / self.total_count
            mode = torch.clamp(torch.floor((self.total_count + 1.0) * params), 0.0, self.total_count).type(
                dtype
            )
        return shift_last_axis_to(mode, 1)


class CategoricalArray(ExponentialFamilyArray):
    """Implementation of Categorical distribution."""

    def __init__(self, num_features, num_channels, num_leaves, num_repetitions, K, use_em=True):
        super(CategoricalArray, self).__init__(
            num_features, num_channels, num_leaves, num_repetitions, num_channels * K, use_em=use_em
        )
        self.K = K

    def default_initializer(self):
        phi = 0.01 + 0.98 * torch.rand(
            self.num_features, self.num_leaves, self.num_repetitions, self.num_channels * self.K
        )
        return phi

    def project_params(self, phi):
        """Note that this is not actually l2-projection. For simplicity, we simply renormalize."""
        phi = phi.reshape(
            self.num_features, self.num_leaves, self.num_repetitions, self.num_channels, self.K
        )
        phi = torch.clamp(phi, min=1e-12)
        phi = phi / torch.sum(phi, -1, keepdim=True)
        return phi.reshape(
            self.num_features, self.num_leaves, self.num_repetitions, self.num_channels * self.K
        )

    def reparam(self, params):
        return torch.nn.functional.softmax(params, -1)

    def sufficient_statistics(self, x):
        if len(x.shape) == 2:
            stats = one_hot(x.long(), self.K)
        elif len(x.shape) == 3:
            stats = one_hot(x.long(), self.K).reshape(
                -1, self.num_features, self.num_channels * self.K
            )
        else:
            raise AssertionError("Input must be 2 or 3 dimensional tensor.")
        return stats

    def expectation_to_natural(self, phi):
        theta = torch.clamp(phi, 1e-12, 1.0)
        theta = theta.reshape(
            self.num_features, self.num_leaves, self.num_repetitions, self.num_channels, self.K
        )
        theta /= theta.sum(-1, keepdim=True)
        theta = theta.reshape(
            self.num_features, self.num_leaves, self.num_repetitions, self.num_channels * self.K
        )
        theta = torch.log(theta)
        return theta

    def log_normalizer(self, theta):
        return 0.0

    def log_h(self, x):
        return torch.zeros([], device=x.device)

    def _sample(self, num_samples, params, differentiable=False, dtype=torch.float32):
        with get_context(differentiable):
            dist = params.reshape(
                self.num_sigma, self.num_leaves, self.num_repetitions, self.num_channels, self.K
            )
            cum_sum = torch.cumsum(dist[..., 0:-1], -1)
            rand = torch.rand(
                (num_samples,) + cum_sum.shape[0:-1] + (1,), device=cum_sum.device
            )
            samples = torch.sum(rand > cum_sum, -1).type(dtype)
            return shift_last_axis_to(samples, 2)

    def _argmax(self, params, differentiable=False, dtype=torch.float32):
        with get_context(differentiable):
            dist = params.reshape(
                self.num_sigma, self.num_leaves, self.num_repetitions, self.num_channels, self.K
            )
            mode = torch.argmax(dist, -1).type(dtype)
            return shift_last_axis_to(mode, 1)
