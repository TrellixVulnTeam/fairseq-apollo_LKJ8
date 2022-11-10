import math
from typing import Dict, Optional, Tuple

import logging
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from fairseq.incremental_decoding_utils import with_incremental_state

logger = logging.getLogger(__name__)

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


@with_incremental_state
class S4(nn.Module):
    def __init__(
        self,
        embed_dim,
        ndim=16,
        bidirectional=False,
        disc='bilinear'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.bidirectional = bidirectional
        self.dt_max = 0.1
        self.dt_min = 0.001
        self.disc = disc
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.log_dt = nn.Parameter(Tensor(kernel_dim, 1))
        self.log_A_real = nn.Parameter(Tensor(kernel_dim, ndim))
        self.A_imaginary = nn.Parameter(Tensor(kernel_dim, ndim))
        self.B = nn.Parameter(Tensor(kernel_dim, ndim, 2))
        self.C = nn.Parameter(Tensor(kernel_dim, ndim, 2))
        self.D = nn.Parameter(Tensor(embed_dim))

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def _transition(self):
        q = np.arange(2 * self.ndim, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)

        return A, B

    def _rank_correction(self, N, rank=1, dtype=torch.float):
        """ Return low-rank matrix L such that A + L is normal """
        assert rank >= 1
        P = torch.sqrt(.5 + torch.arange(N, dtype=dtype)).unsqueeze(0)  # (1 N)
        d = P.size(0)
        if rank > d:
            P = torch.cat([P, torch.zeros(rank - d, N, dtype=dtype)], dim=0)  # (rank N)
        return P

    def nplr(self, N, rank=1, dtype=torch.float16, diagonalize_precision=True):
        assert dtype == torch.float or dtype == torch.double
        cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

        A, B = self._transition()
        A = torch.as_tensor(A, dtype=dtype)  # (N, N)
        B = torch.as_tensor(B, dtype=dtype)[:, 0]  # (N,)

        P = self._rank_correction(N, rank=rank, dtype=dtype)  # (r N)
        AP = A + torch.sum(P.unsqueeze(-2) * P.unsqueeze(-1), dim=-3)

        # We require AP to be nearly skew-symmetric
        _A = AP + AP.transpose(-1, -2)

        err = torch.sum((_A - _A[0, 0] * torch.eye(N)) ** 2) / N
        if err > 1e-5:
            print("WARNING: HiPPO matrix not skew symmetric", err)

        # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
        # Imaginary part can use eigh instead of eig
        w_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

        # Diagonalize in double precision
        if diagonalize_precision: AP = AP.to(torch.double)
        # w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
        w_im, V = torch.linalg.eigh(AP * -1j)  # (..., N) (..., N, N)
        if diagonalize_precision: w_im, V = w_im.to(cdtype), V.to(cdtype)
        w = w_re + 1j * w_im
        # Check: V w V^{-1} = A
        # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))

        # Only keep half of each conjugate pair
        _, idx = torch.sort(w.imag)
        w_sorted = w[idx]
        V_sorted = V[:, idx]

        # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
        # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
        V = V_sorted[:, :N // 2]
        w = w_sorted[:N // 2]
        assert w[-2].abs() > 1e-4, "Only 1 zero eigenvalue allowed in diagonal part of A"
        if w[-1].abs() < 1e-4:
            V[:, -1] = 0.
            V[0, -1] = 2 ** -0.5
            V[1, -1] = 2 ** -0.5 * 1j

        _AP = V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2)

        err = torch.sum((2 * _AP.real - AP) ** 2) / N
        if err > 1e-5:
            print("Warning: Diagonalization of A matrix not numerically precise - error", err)

        V_inv = V.conj().transpose(-1, -2)

        # C = initial_C(measure, N, dtype=dtype)
        B = torch.einsum('ij,j->i', V_inv, B.to(V))  # V^* B
        # C = contract('ij, j -> i', V_inv, C.to(V)) # V^* C
        P = torch.einsum('ij,...j->...i', V_inv, P.to(V))  # V^* P

        # return w, P, B, C, V
        return w, P, B, V

    def reset_parameters(self):
        log_dt = torch.randn_like(self.log_dt) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        with torch.no_grad():
            self.log_dt.copy_(log_dt)

        w, P, B, V = self.nplr(self.ndim * 2, rank=1)

        C = _c2r(torch.randn(self.ndim, dtype=torch.cfloat))
        with torch.no_grad():
            self.B.copy_(B)
            self.C.copy_(C)

        nn.init.normal_(self.D, mean=0.0, std=1.0)