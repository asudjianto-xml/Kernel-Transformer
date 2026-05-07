"""Diffusion / spectral propagation kernels (Chapter 7).

Seven operators that take a symmetric affinity W and propagate it to a
multi-hop diffusion kernel. All are PSD by construction (with the
exception of Chebyshev with arbitrary coefficients).
"""
from tabkernels.diffusion.heat import HeatKernel
from tabkernels.diffusion.reg_laplacian import RegLaplacianKernel
from tabkernels.diffusion.ppr import PPRKernel
from tabkernels.diffusion.pstep import PStepKernel
from tabkernels.diffusion.chebyshev import ChebyshevKernel
from tabkernels.diffusion.commute_time import CommuteTimeKernel
from tabkernels.diffusion.learnable_spectral import LearnableSpectralKernel
from tabkernels.diffusion._laplacian import (
    laplacian_sym, laplacian_comb, transition_sym, degree_matrix,
)

__all__ = [
    "HeatKernel",
    "RegLaplacianKernel",
    "PPRKernel",
    "PStepKernel",
    "ChebyshevKernel",
    "CommuteTimeKernel",
    "LearnableSpectralKernel",
    "laplacian_sym",
    "laplacian_comb",
    "transition_sym",
    "degree_matrix",
]
