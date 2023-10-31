from functools import partial

import numpy as np
import torch
from survae.distributions import Distribution
from torch import nn

from ddflow.utils import hard_sigmoid


class DependentDistribution(Distribution):
    def log_prob(self, x):
        return self.log_prob_independent(x)

    def log_prob_independent(self, x):
        scale = 1
        var = scale**2
        log_scale = np.log(scale)
        return (
            -((x - self.loc) ** 2) / (2 * var) - log_scale - np.log(np.sqrt(2 * np.pi))
        ).sum(1)

    def log_prob_mini_batch(self, x, sample_indices):
        raise NotImplementedError()

    def sample(self, num_samples):
        eps = torch.randn(num_samples, self.p, device=self.loc.device)
        return self.loc + eps


class CovMixtureMVN(DependentDistribution):
    def __init__(
        self,
        loc,
        relationship_matrix,
        spectral_decomp=None,
        default_lam=0.5,
        opt_params=True,
        lam_activation="sigmoid",
    ):
        super().__init__()
        assert relationship_matrix is None or spectral_decomp is None
        self._setup_lam_activation(lam_activation)

        self.opt_params = opt_params
        self.p = loc.shape[-1]
        self.register_buffer("loc", loc)

        lam = torch.tensor(default_lam)
        if opt_params:
            self.lam = nn.Parameter(lam)
        else:
            self.register_buffer("lam", lam)
        self._setup_relationship_matrix(relationship_matrix, spectral_decomp)

        self.const_term = -self.n * self.p * torch.log(torch.tensor(2 * torch.pi)) / 2

    def rotate(self, z):
        return self.Q.T @ (z - self.loc)

    def log_prob_lam3(self, outer):
        """alternative implementation for `log_prob_lam` -- unused atm"""
        lam = self.get_lam()

        const_term = self.const_term

        det_term = -0.5 * self.p * torch.log(lam + (1 - lam) * self.eigenvals).sum()

        diag = (lam + (1 - lam) * self.eigenvals) ** -1

        trace2 = (outer.diag() * diag).sum()
        trace_term = -0.5 * trace2
        log_p = (const_term + det_term + trace_term) / self.n
        return log_p

    def log_prob_lam2(self, rotated_value):
        """alternative implementation for `log_prob_lam` -- unused atm"""
        lam = self.get_lam()

        # const term: np/2 log(2pi)
        const_term = self.const_term

        # det term: sum log(lam + (1-lam) EV)
        det_term = -0.5 * self.p * torch.log(lam + (1 - lam) * self.eigenvals).sum()

        # trace-term
        diag = (lam + (1 - lam) * self.eigenvals) ** -1

        outer = rotated_value @ rotated_value.T
        trace2 = (outer.diag() * diag).sum()
        trace_term = -0.5 * trace2
        log_p = (const_term + det_term + trace_term) / self.n
        return log_p

    def log_prob_lam(self, rotated_value):
        """full-batch loss for lambda only;
        caution, only meant for optimizing lambda alone! Otherwise, relationship between log-det-jac and base-log-likelihood might be off
        assumes that the input has already been centered and rotated by self.Q, i.e.:
            rotated_value = self.Q.T @ (value - self.loc)
        --> way more efficient this way, otherwise we'd need to recompute that matrix product everytime
        """
        lam = self.get_lam()

        # const term: np/2 log(2pi)
        const_term = self.const_term

        # det term: sum log(lam + (1-lam) EV)
        det_term = -0.5 * self.p * torch.log(lam + (1 - lam) * self.eigenvals).sum()

        # trace-term
        diag = (lam + (1 - lam) * self.eigenvals) ** -1

        # naive implementation:
        # trace1 = torch.trace(rotated_value.T @ (diag[:, None] * rotated_value))
        # more efficient implementation; better even with einsum, but torch implementation of einsum is super slow
        trace2 = (rotated_value.T * (diag[:, None] * rotated_value).T).sum()

        trace_term = -0.5 * trace2
        log_p = (const_term + det_term + trace_term) / self.n
        return log_p

    def log_prob_mini_batch(self, value, sample_indices):
        """use fixed lam only!"""
        bs = len(sample_indices)

        # const term: p/2 log(2pi)
        const_term = self.const_term / self.n

        # det term: sum log(lam + (1-lam) EV)
        det_term = -0.5 * self.p * self.get_cached_log_det() / self.n

        # trace term:
        A = self.get_cached_precision_matrix()
        diff = value - self.loc
        diag_A = A[sample_indices, sample_indices]
        diag_term = diag_A * (diff**2).sum(1)

        uiuj = diff @ diff.T
        prod = A[sample_indices, :][:, sample_indices] * uiuj
        prod[torch.arange(bs), torch.arange(bs)] = 0
        off_diag = prod.sum(1) * (self.n - 1) / (bs - 1)

        trace_term = -0.5 * (diag_term + off_diag)
        log_p = const_term + det_term + trace_term
        return log_p

    def get_cached_precision_matrix(self):
        if not hasattr(self, "_cached_precision"):
            self.cache_precision_matrix()
        return self._cached_precision

    def get_cached_log_det(self):
        if not hasattr(self, "_log_det"):
            self.cache_precision_matrix()
        return self._log_det

    @torch.no_grad()
    def cache_precision_matrix(self):
        lam = self.get_lam()
        diag = (lam + (1 - lam) * self.eigenvals) ** -1
        self._cached_precision = self.Q @ (diag * self.Q).T
        self._log_det = torch.log(lam + (1 - lam) * self.eigenvals).sum()

    def get_spectral_decomp(self):
        return (self.eigenvals, self.Q)

    def _setup_relationship_matrix(self, relationship_matrix, spectral_decomp):
        """currently implemented with torch.linalg.eigh on float"""
        if relationship_matrix is None:
            print("using stored spectral decomposition...")
            eigenvals, Q = spectral_decomp
        else:
            print("performing spectral decomposition ...")
            eigh = torch.linalg.eigh(torch.from_numpy(relationship_matrix).float())
            eigenvals, Q = eigh.eigenvalues, eigh.eigenvectors

            print("finished spectral decomposition")
        self.register_buffer("eigenvals", eigenvals)
        self.register_buffer("Q", Q)
        self.n = len(self.eigenvals)

    def get_lam(self):
        return self.lambda_activation(self.lam)

    def _setup_lam_activation(self, act):
        if act == "sigmoid":
            self.lambda_activation = torch.sigmoid
        # e.g. 'hardsigmoid-3.0'
        elif act.startswith("hardsigmoid"):
            bound = float(act.split("-")[1])
            self.lambda_activation = partial(hard_sigmoid, bound=bound, eps=0)


class EquiDependentUnitMVN(DependentDistribution):
    def __init__(
        self,
        loc,
        ind_blocks,
        default_rho=0.5,
        opt_params=True,
        rho_activation="sigmoid",
    ):
        """
        default_rho can either be single scalar, or can be list/array of same length as ind_blocks, specifying one rho per block
        """
        super().__init__()
        self._setup_rho_activation(rho_activation)

        self.opt_params = opt_params
        self.p = loc.shape[-1]
        self.register_buffer("loc", loc)

        # ind_blocks = [[1,2], [3], [4, 5, 6], ...]
        self.N = len(ind_blocks)
        self.register_buffer("dims", torch.tensor([len(t) for t in ind_blocks]))
        self.n = sum(self.dims)
        if isinstance(default_rho, torch.Tensor) and default_rho.numel() == 1:
            default_rho = default_rho.item()
        if isinstance(default_rho, float):
            rhos = torch.tensor([default_rho for _ in ind_blocks])
        else:
            rhos = torch.tensor(default_rho).float()
        if opt_params:
            self.rhos = nn.Parameter(rhos)
        else:
            self.register_buffer("rhos", rhos)

        self.index_lookup = dict(
            (elm, block_ind)
            for block_ind, block in enumerate(ind_blocks)
            for elm in block
        )
        self.const_term = -self.n * self.p * torch.log(torch.tensor(2 * torch.pi)) / 2

    def get_rhos(self):
        return self.rho_activation(self.rhos)

    def _setup_rho_activation(self, act):
        if act == "sigmoid":
            self.rho_activation = torch.sigmoid
        # e.g. 'hardsigmoid-3.0'
        elif act.startswith("hardsigmoid"):
            bound = float(act.split("-")[1])
            self.rho_activation = partial(hard_sigmoid, bound=bound)

    def log_prob_mini_batch(self, value, sample_indices, overwrite_bs=None):
        # TODO: inefficient, especially the loop below, but doesn't seem to be a bottleneck, so not optimized for now
        if overwrite_bs:
            bs = overwrite_bs
        else:
            bs = len(sample_indices)
        sub_bs = len(sample_indices)
        p12 = (bs - 1) / (self.n - 1)

        const_term = self.const_term / self.n

        block_inds = torch.tensor(
            [
                self.index_lookup[ind.item() if isinstance(ind, torch.Tensor) else ind]
                for ind in sample_indices
            ]
        )
        batch_rhos = self.rho_activation(self.rhos[block_inds])
        batch_ns = self.dims[block_inds]
        batch_deltas = (1 - batch_rhos) * (1 + (batch_ns - 1) * batch_rhos)

        det_term = (
            -0.5
            * self.p
            * (
                torch.log(1 + (batch_ns - 1) * batch_rhos)
                + (batch_ns - 1) * torch.log(1 - batch_rhos)
            )
            / batch_ns
        )

        diff = value - self.loc

        diag_A = torch.ones(sub_bs, device=value.device)
        diag_A[batch_ns > 1] = ((1 + (batch_ns - 2) * batch_rhos) / batch_deltas)[
            batch_ns > 1
        ]
        diag = diag_A * (diff**2).sum(1)

        uniques, indices, inverse, counts = np.unique(
            block_inds, return_index=True, return_counts=True, return_inverse=True
        )
        dupl_blocks = uniques[counts > 1]
        dupl_rhos = self.rho_activation(self.rhos[dupl_blocks])
        dupl_ns = self.dims[dupl_blocks]
        dupl_deltas = (1 - dupl_rhos) * (1 + (dupl_ns - 1) * dupl_rhos)

        off_diag = torch.zeros(sub_bs, device=value.device)
        for i, (b, rho, delta) in enumerate(zip(dupl_blocks, dupl_rhos, dupl_deltas)):
            sub_block = block_inds == b
            c = sub_block.sum()
            aij = -rho / delta
            uiuj = diff[sub_block] @ diff[sub_block].T
            prod = aij * uiuj
            prod[torch.arange(c), torch.arange(c)] = 0
            off_diag[sub_block] = prod.sum(1) / p12

        trace_term = -0.5 * (diag + off_diag)

        log_p = const_term + det_term + trace_term

        return log_p
