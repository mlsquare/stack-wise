"""
Test script for kernel functions.
"""

import torch
from ..kernels import create_kernel_matrix, apply_kernel, get_kernel_info, KernelType


def test_kernel_functions():
    """Test all kernel functions."""
    
    print("🧪 Testing Kernel Functions")
    print("=" * 50)
    
    # Test parameters
    kernel_dim = 32
    d_k = 16
    batch_size = 2
    seq_len = 8
    n_heads = 4
    
    # Create test input
    x = torch.randn(batch_size, n_heads, seq_len, d_k)
    
    # Test all kernel types
    kernel_types: list[KernelType] = ["dot_product", "gaussian", "laplacian", "uniform"]
    
    for kernel_type in kernel_types:
        print(f"\n🔍 Testing {kernel_type} kernel:")
        
        # Create kernel matrix
        kernel_matrix = create_kernel_matrix(kernel_type, kernel_dim, d_k)
        print(f"   Matrix shape: {kernel_matrix.shape}")
        
        # Apply kernel transformation
        x_transformed = apply_kernel(x, kernel_matrix, kernel_type)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {x_transformed.shape}")
        
        # Get kernel info
        info = get_kernel_info(kernel_type, kernel_dim, d_k)
        print(f"   Info: {info}")
        
        # Verify shapes
        if kernel_type == "dot_product":
            assert x_transformed.shape == x.shape, f"Dot product should preserve shape: {x.shape} vs {x_transformed.shape}"
        else:
            expected_shape = (*x.shape[:-1], kernel_dim)
            assert x_transformed.shape == expected_shape, f"Kernel should transform last dim: {expected_shape} vs {x_transformed.shape}"
        
        print(f"   ✅ {kernel_type} kernel working correctly!")
    
    print("\n✅ All kernel functions tested successfully!")


def test_kernel_edge_cases():
    """Test edge cases for kernel functions."""
    
    print("\n🧪 Testing Kernel Edge Cases")
    print("=" * 50)
    
    # Test with different dimensions
    test_cases = [
        (8, 4),   # kernel_dim < d_k
        (16, 16), # kernel_dim = d_k  
        (32, 8),  # kernel_dim > d_k
    ]
    
    for kernel_dim, d_k in test_cases:
        print(f"\n🔍 Testing kernel_dim={kernel_dim}, d_k={d_k}:")
        
        for kernel_type in ["gaussian", "laplacian", "uniform"]:
            try:
                kernel_matrix = create_kernel_matrix(kernel_type, kernel_dim, d_k)
                print(f"   ✅ {kernel_type}: {kernel_matrix.shape}")
            except Exception as e:
                print(f"   ❌ {kernel_type}: {e}")
    
    print("\n✅ Edge case testing completed!")


if __name__ == "__main__":
    test_kernel_functions()
    test_kernel_edge_cases()
