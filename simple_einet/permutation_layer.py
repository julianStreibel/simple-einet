import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
import numpy as np


class PermutationLayer(torch.nn.Module):
    def __init__(self, f, r, iterations=20):
        super().__init__()
        self.iterations = iterations
        self.perm_seed = torch.nn.Parameter(torch.rand(f, f, r))
        self.r = r
        self.matching_perm = None
        self.matching_up_to_date = False

    def forward(self, X, tau=1):
        if self.training:
            p = self.gumbel_sinkhorn(
                self.perm_seed.permute(2, 0, 1), tau, self.iterations)
            self.matching_up_to_date = False
        else:
            if self.matching_up_to_date:
                p = self.matching_perm
            else:
                detached_perm_seed = self.perm_seed.detach().cpu()
                p = torch.stack([
                    self.matching(detached_perm_seed[:, :, r_i])
                    for r_i in range(self.r)
                ])
                self.matching_perm = p
                self.matching_up_to_date = True

        # torch.einsum("nm,mb->nb", p, X.T).T  # p.matmul(X.T).T
        X = X.permute(4, 1, 0, 2, 3)
        X = torch.einsum("rfm,rfblc->rmblc", p, X)
        X = X.permute(2, 1, 3, 4, 0)
        return X

    def inv(self, X, tau=1):
        if self.training:
            p = self.gumbel_sinkhorn(self.perm_seed, tau, self.iterations)
        else:
            detached_perm_seed = self.perm_seed.detach().cpu()
            p = torch.stack([
                self.matching(detached_perm_seed[:, :, r_i])
                for r_i in range(self.r)
            ])

        X = X.permute(1, 0, 2, 3, 4)
        X = torch.einsum("mf,fblcr->mblcr", p.T, X)
        X = X.permute(1, 0, 2, 3, 4)
        return X

    def get_perm(self):
        detached_perm_seed = self.perm_seed.detach()
        p = torch.stack([
            self.matching(detached_perm_seed[:, :, r_i])
            for r_i in range(self.r)
        ])
        return p

    def matching(self, alpha):
        # Negate the probability matrix to serve as cost matrix. This function
        # yields two lists, the row and colum indices for all entries in the
        # permutation matrix we should set to 1.
        row, col = linear_sum_assignment(-alpha)

        # Create the permutation matrix.
        permutation_matrix = coo_matrix(
            (np.ones_like(row), (row, col))).toarray()
        return torch.from_numpy(permutation_matrix).type(torch.FloatTensor).to(self.perm_seed.device)

    def log_sinkhorn(self, log_alpha, n_iter):
        """Performs incomplete Sinkhorn normalization to log_alpha.
        By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
        with positive entries can be turned into a doubly-stochastic matrix
        (i.e. its rows and columns add up to one) via the successive row and column
        normalization.

        [1] Sinkhorn, Richard and Knopp, Paul.
        Concerning nonnegative matrices and doubly stochastic
        matrices. Pacific Journal of Mathematics, 1967
        Args:
        log_alpha: 2D tensor (a matrix of shape [N, N])
            or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        n_iters: number of sinkhorn iterations (in practice, as little as 20
            iterations are needed to achieve decent convergence for N~100)
        Returns:
        A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
            converted to 3D tensors with batch_size equals to 1)
        """
        for _ in range(n_iter):
            log_alpha = log_alpha - \
                torch.logsumexp(log_alpha, -1, keepdim=True)
            log_alpha = log_alpha - \
                torch.logsumexp(log_alpha, -2, keepdim=True)
        return log_alpha.exp()

    def sample_gumbel(self, shape, device='cpu', eps=1e-20):
        """Samples arbitrary-shaped standard gumbel variables.
        Args:
        shape: list of integers
        eps: float, for numerical stability
        Returns:
        A sample of standard Gumbel random variables
        """
        u = torch.rand(shape, device=device)
        return - torch.log(-torch.log(u + eps) + eps)

    def gumbel_sinkhorn(self, log_alpha, tau, n_iter):
        """ Sample a permutation matrix from the Gumbel-Sinkhorn distribution
        with parameters given by log_alpha and temperature tau.

        Args:
        log_alpha: Logarithm of assignment probabilities. In our case this is
            of dimensionality [num_pieces, num_pieces].
        tau: Temperature parameter, the lower the value for tau the more closely
            we follow a categorical sampling.
        """
        # Sample Gumbel noise.
        gumbel_noise = self.sample_gumbel(
            log_alpha.shape, device=log_alpha.device)

        # Apply the Sinkhorn operator!
        sampled_perm_mat = self.log_sinkhorn(
            (log_alpha + gumbel_noise)/tau, n_iter)
        return sampled_perm_mat
