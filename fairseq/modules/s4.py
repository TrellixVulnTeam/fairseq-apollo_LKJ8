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


def cauchy_naive(v, z, w):
    """
    v, w: (..., N)
    z: (..., L)
    returns: (..., L)
    """
    cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))  # (... N L)
    return torch.sum(cauchy_matrix, dim=-2)


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
        self.rank = 1
        self.disc = disc
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.log_dt = nn.Parameter(Tensor(kernel_dim, 1))
        self.inv_w_real = nn.Parameter(Tensor(kernel_dim, ndim))
        self.w_imaginary = nn.Parameter(Tensor(kernel_dim, ndim))
        self.B = nn.Parameter(Tensor(kernel_dim, ndim, 2))
        self.P = nn.Parameter(Tensor(self.rank, kernel_dim, ndim, 2))
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

    def nplr(self, N, rank=1, dtype=torch.float, diagonalize_precision=True):
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

        N = self.ndim * 2
        H = self.P.size(0)
        w, P, B, V = self.nplr(N, rank=self.rank)
        w = w.unsqueeze(0).expand(H, self.ndim)
        P = P.unsqueeze(1).expand(self.rank, H, self.ndim)
        B = B.unsqueeze(0).expand(H, self.ndim)

        C = _c2r(torch.randn(self.ndim, dtype=torch.cfloat))
        with torch.no_grad():
            self.P.copy_(_c2r(P))
            self.B.copy_(_c2r(B))
            self.C.copy_(C)
            self.inv_w_real.copy_(torch.log(-w.real))
            self.w_imaginary.copy_(w.imag)

        nn.init.normal_(self.D, mean=0.0, std=1.0)

    def _w(self):
        w_real = -torch.exp(self.inv_w_real)
        w = w_real + 1j * self.w_imaginary
        return w

    def _omega(self, L, dtype, device):
        """ Calculate (and cache) FFT nodes and their "unprocessed" version with the bilinear transform
        This should be called everytime the internal length self.L changes """

        omega = torch.tensor(np.exp(-2j * np.pi / (L)), dtype=dtype, device=device)  # \omega_{2L}
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega)
        return omega, z

    def compute_kernel(self, L):
        # H x 1
        dt = torch.exp(self.log_dt)
        # H x N
        B = _r2c(self.B)
        C = _r2c(self.C)
        # r x H x N
        P = _r2c(self.P)
        Q = P.conj()
        # H x N
        w = self._w()

        # Get FFT nodes of right length
        omega, z = self._omega(L, dtype=w.dtype, device=w.device)

        w = w * dt
        # Stack B and p, C and q for convenient batching
        B = torch.cat([B.unsqueeze(0), P], dim=-3)  # (B+1+R, H, N)
        C = torch.cat([C.unsqueeze(0), Q], dim=-3)  # (C+R, H, N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)  # (B+1+R, C+R, H, N)

        r = cauchy_naive(v, z, w)
        r = r * dt[None, None, :]  # (B+1+R, C+R, H, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, : -self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                    r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                    + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                    - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                    - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            raise NotImplementedError

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f, n=L)  # (B+1, C, H, L)

        # # Truncate to target length
        k = k[..., :L]
        k_B = k[-1, 0, :, :]  # (H L)

        return k_B

    def kernel(self, length: int):
        return self.compute_kernel(length)

    def step(self, x, length, hx=None):
        if length == 1:
            return self.one_step(x, hx=hx)

        raise NotImplementedError

    def one_step(self, x, hx=None):
        raise NotImplementedError

    def forward(
        self,
        x,
        padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """

        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        # L x B x D
        residual = x * self.D

        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        assert not self.bidirectional or incremental_state is None, 'Bidirectional S4D does not support incremental state'
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_state' in saved_state:
                h = saved_state['prev_state']
            else:
                h = None
            out, h = self.step(x, seq_len, hx=h)
            saved_state['prev_state'] = h
            self._set_input_buffer(incremental_state, saved_state)
            # B x D -> 1 x B x D
            out = F.silu(out + residual)
        else:
            # D x L
            k = self.kernel(seq_len)
            fft_len = seq_len
            s = 0
            kernel_size = k.size(1)
            if self.bidirectional:
                k1, k2 = torch.split(k, [self.embed_dim, self.embed_dim], dim=0)
                # D x 2*L-1
                k = F.pad(k1, (kernel_size - 1, 0)) + F.pad(k2.flip(-1), (0, kernel_size - 1))
                x = F.pad(x, (kernel_size - 1, 0))
                fft_len = fft_len + kernel_size - 1
                s = 2 * kernel_size - 2

            k_f = torch.fft.rfft(k.float(), n=2 * fft_len)
            x_f = torch.fft.rfft(x.float(), n=2 * fft_len)
            # B x D x L
            out = torch.fft.irfft(x_f * k_f, n=2 * fft_len)[..., s:s + seq_len]
            out = out.type_as(x)
            # B x D x L -> L x B x D
            out = F.silu(out.permute(2, 0, 1) + residual)

        return out

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "ema_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, "ema_state", buffer)

    @torch.jit.export
    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def extra_repr(self) -> str:
        return 'edim={}, ndim={}, bidirectional={}, rank={}'.format(self.embed_dim, self.ndim, self.bidirectional, self.rank)
