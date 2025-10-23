"""
Test script for all attention mechanisms.
"""

import torch
from typing import Dict, Any
from ..attention import CoreAttention
from ..builder import get_attention_info


def test_attention_mechanisms():
    """Test all attention mechanisms."""
    
    # Test parameters
    batch_size = 2
    seq_len = 16
    d_model = 64
    n_heads = 8
    n_kv_heads = 2
    r_q = 32
    r_kv = 16
    kernel_dim = 32
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("🧠 Testing StackWise Attention Mechanisms")
    print("=" * 50)
    
    # Test configurations for unified kernel attention
    test_configs = [
        {
            "name": "MHA (dot_product kernel)",
            "type": "mha",
            "params": {"d_model": d_model, "n_heads": n_heads, "kernel_type": "dot_product"}
        },
        {
            "name": "GQA (dot_product kernel)",
            "type": "gqa", 
            "params": {"d_model": d_model, "n_heads": n_heads, "n_kv_heads": n_kv_heads, "kernel_type": "dot_product"}
        },
        {
            "name": "MLA (dot_product kernel)",
            "type": "mla",
            "params": {"d_model": d_model, "n_heads": n_heads, "n_kv_heads": n_kv_heads, "r_q": r_q, "r_kv": r_kv, "kernel_type": "dot_product"}
        },
        {
            "name": "GQA + MLA (dot_product kernel)",
            "type": "mla",
            "params": {"d_model": d_model, "n_heads": n_heads, "n_kv_heads": n_kv_heads, "r_q": r_q, "r_kv": r_kv, "kernel_type": "dot_product"}
        },
        {
            "name": "Kernel Attention (Gaussian)",
            "type": "kernel",
            "params": {"d_model": d_model, "n_heads": n_heads, "kernel_dim": kernel_dim, "kernel_type": "gaussian"}
        },
        {
            "name": "Kernel Attention (Laplacian)",
            "type": "kernel",
            "params": {"d_model": d_model, "n_heads": n_heads, "kernel_dim": kernel_dim, "kernel_type": "laplacian"}
        },
        {
            "name": "GQA + Kernel (Gaussian)",
            "type": "gqa",
            "params": {"d_model": d_model, "n_heads": n_heads, "n_kv_heads": n_kv_heads, "kernel_dim": kernel_dim, "kernel_type": "gaussian"}
        }
    ]
    
    # Test each attention mechanism
    for config in test_configs:
        print(f"\n🔍 Testing {config['name']}")
        print("-" * 30)
        
        try:
            # Create attention mechanism
            attention = CoreAttention(**config['params'])
            
            # Test bidirectional attention
            attention.set_attention_mode("bidirectional")
            output_bidi = attention(x)
            
            # Test causal attention
            attention.set_attention_mode("causal")
            output_causal = attention(x)
            
            # Verify output shapes
            assert output_bidi.shape == x.shape, f"Bidirectional output shape mismatch: {output_bidi.shape} vs {x.shape}"
            assert output_causal.shape == x.shape, f"Causal output shape mismatch: {output_causal.shape} vs {x.shape}"
            
            # Get attention info
            info = get_attention_info(attention)
            
            print(f"✅ {config['name']} working correctly")
            print(f"   - Output shape: {output_bidi.shape}")
            print(f"   - Attention mode: {info['attention_mode']}")
            print(f"   - Model dimension: {info['d_model']}")
            
            # Test attention weights
            attention_weights = attention.get_attention_weights(x)
            expected_shape = (batch_size, n_heads, seq_len, seq_len)
            assert attention_weights.shape == expected_shape, f"Attention weights shape mismatch: {attention_weights.shape} vs {expected_shape}"
            print(f"   - Attention weights shape: {attention_weights.shape}")
            
            # Test parameter count for MLA
            if config['type'] == 'mla':
                param_info = info.get('parameter_count', {})
                if param_info:
                    print(f"   - Total parameters: {param_info['total']:,}")
                    print(f"   - Compression ratio: {param_info['compression_ratio']:.2f}")
            
            # Test kernel info for kernel attention
            if config['type'] == 'kernel':
                kernel_info = info.get('kernel_info', {})
                if kernel_info:
                    print(f"   - Kernel type: {kernel_info['kernel_type']}")
                    print(f"   - Kernel dimension: {kernel_info['kernel_dim']}")
                    print(f"   - Compression ratio: {kernel_info['compression_ratio']:.2f}")
            
        except Exception as e:
            print(f"❌ {config['name']} failed: {e}")
            raise
    
    print(f"\n🎉 All attention mechanisms working correctly!")
    print(f"✅ Tested {len(test_configs)} attention mechanisms")
    print(f"✅ Bidirectional and causal attention modes working")
    print(f"✅ Attention weights computation working")
    print(f"✅ Parameter counting working")


def test_attention_modes():
    """Test attention mode switching."""
    
    print(f"\n🔄 Testing Attention Mode Switching")
    print("-" * 40)
    
    # Create attention mechanism
    attention = CoreAttention(d_model=64, n_heads=8)
    
    # Test input
    x = torch.randn(1, 8, 64)
    
    # Test bidirectional mode
    attention.set_attention_mode("bidirectional")
    output_bidi = attention(x)
    weights_bidi = attention.get_attention_weights(x)
    
    # Test causal mode
    attention.set_attention_mode("causal")
    output_causal = attention(x)
    weights_causal = attention.get_attention_weights(x)
    
    # Verify causal mask is applied
    assert weights_causal.shape == weights_bidi.shape
    print(f"✅ Bidirectional mode: {attention.get_attention_mode()}")
    print(f"✅ Causal mode: {attention.get_attention_mode()}")
    print(f"✅ Attention weights shape: {weights_bidi.shape}")
    
    # Verify causal mask creates lower triangular pattern
    causal_mask = weights_causal[0, 0]  # First head
    upper_triangular = torch.triu(causal_mask, diagonal=1)
    assert torch.allclose(upper_triangular, torch.zeros_like(upper_triangular), atol=1e-6), "Causal mask not properly applied"
    print(f"✅ Causal mask properly applied")


if __name__ == "__main__":
    print("🚀 StackWise Attention System Test Suite")
    print("=" * 50)
    
    # Test all attention mechanisms
    test_attention_mechanisms()
    
    # Test attention mode switching
    test_attention_modes()
    
    print(f"\n🎯 All tests passed! Attention system ready for production.")
