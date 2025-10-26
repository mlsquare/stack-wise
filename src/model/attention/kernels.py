"""
Pure kernel implementations for attention mechanisms.
Extracted from core.py for better organization.
"""

import torch
from typing import Literal

KernelType = Literal["linear", "gaussian", "laplacian", "uniform"]


def create_kernel_matrix(
    kernel_type: KernelType, 
    kernel_dim: int, 
    d_k: int
) -> torch.Tensor:
    """
    Create random kernel matrix based on kernel type.
    
    Args:
        kernel_type: Type of kernel to create
        kernel_dim: Dimension of kernel space
        d_k: Dimension of key space
        
    Returns:
        Kernel matrix or empty tensor for linear
    """
    if kernel_type == "linear":
        # Linear kernel (scaled_dot_product) doesn't need a matrix
        return torch.empty(0)
    elif kernel_type == "gaussian":
        # Gaussian random matrix
        matrix = torch.randn(kernel_dim, d_k)
    elif kernel_type == "laplacian":
        # Laplacian random matrix (using exponential distribution)
        matrix = -torch.log(torch.rand(kernel_dim, d_k))
    elif kernel_type == "uniform":
        # Uniform random matrix
        matrix = torch.rand(kernel_dim, d_k) * 2 - 1
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return matrix


def apply_kernel(
    x: torch.Tensor, 
    kernel_matrix: torch.Tensor, 
    kernel_type: KernelType
) -> torch.Tensor:
    """
    Apply kernel transformation using Random Kitchen Sinks.
    
    Args:
        x: Input tensor of shape (..., d_k)
        kernel_matrix: Kernel matrix (ignored for linear)
        kernel_type: Type of kernel to apply
        
    Returns:
        Kernel-transformed tensor of shape (..., kernel_dim) or original shape for linear
    """
    if kernel_type == "linear":
        # Linear kernel (scaled_dot_product): no transformation needed
        return x
    else:
        # Apply kernel transformation: phi(x) = cos(x @ W^T)
        x_proj = torch.matmul(x, kernel_matrix.T)
        return torch.cos(x_proj)


def get_kernel_info(
    kernel_type: KernelType,
    kernel_dim: int,
    d_k: int
) -> dict:
    """
    Get information about the kernel transformation.
    
    Args:
        kernel_type: Type of kernel
        kernel_dim: Dimension of kernel space
        d_k: Dimension of key space
        
    Returns:
        Dictionary with kernel information
    """
    return {
        "kernel_type": kernel_type,
        "kernel_dim": kernel_dim,
        "d_k": d_k,
        "compression_ratio": kernel_dim / d_k if kernel_type != "linear" else 1.0,
        "is_linear": kernel_type == "linear"
    }
