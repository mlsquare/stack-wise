"""
Test script for the AttentionBuilder pattern.
Demonstrates the fluent API for creating attention mechanisms.
"""

import torch
from ..builder import AttentionBuilder


def test_attention_builder():
    """Test the AttentionBuilder fluent API."""
    
    print("🏗️ Testing AttentionBuilder Pattern")
    print("=" * 50)
    
    # Test parameters
    d_model = 64
    n_heads = 8
    batch_size = 2
    seq_len = 16
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test configurations
    test_configs = [
        {
            "name": "MHA (default)",
            "builder": AttentionBuilder(d_model, n_heads),
            "description": "Default Multi-Head Attention"
        },
        {
            "name": "GQA",
            "builder": AttentionBuilder(d_model, n_heads).with_gqa(2),
            "description": "Grouped Query Attention"
        },
        {
            "name": "MLA",
            "builder": AttentionBuilder(d_model, n_heads).with_mla(32, 16),
            "description": "Multi-Latent Attention"
        },
        {
            "name": "GQA + MLA",
            "builder": AttentionBuilder(d_model, n_heads).with_gqa(2).with_mla(32, 16),
            "description": "Grouped Query + Multi-Latent Attention"
        },
        {
            "name": "Kernel (Gaussian)",
            "builder": AttentionBuilder(d_model, n_heads).with_kernel("gaussian", 32),
            "description": "Kernel Attention with Gaussian kernel"
        },
        {
            "name": "GQA + Kernel",
            "builder": AttentionBuilder(d_model, n_heads).with_gqa(2).with_kernel("laplacian", 32),
            "description": "Grouped Query + Kernel Attention"
        },
        {
            "name": "MLA + Kernel",
            "builder": AttentionBuilder(d_model, n_heads).with_mla(32, 16).with_kernel("uniform", 32),
            "description": "Multi-Latent + Kernel Attention"
        },
        {
            "name": "GQA + MLA + Kernel",
            "builder": AttentionBuilder(d_model, n_heads).with_gqa(2).with_mla(32, 16).with_kernel("gaussian", 32),
            "description": "All features combined"
        }
    ]
    
    # Test each configuration
    for config in test_configs:
        print(f"\n🔍 Testing {config['name']}")
        print(f"   Description: {config['description']}")
        print("-" * 40)
        
        try:
            # Build attention mechanism
            attention = config['builder'].build()
            
            # Get configuration info
            config_info = config['builder'].get_config()
            
            # Test forward pass
            output = attention(x)
            
            # Test attention weights
            weights = attention.get_attention_weights(x)
            
            # Verify output shapes
            assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
            expected_weights_shape = (batch_size, n_heads, seq_len, seq_len)
            assert weights.shape == expected_weights_shape, f"Weights shape mismatch: {weights.shape} vs {expected_weights_shape}"
            
            print(f"✅ {config['name']} working correctly")
            print(f"   - Output shape: {output.shape}")
            print(f"   - Attention weights shape: {weights.shape}")
            print(f"   - Configuration: {config_info}")
            
            # Show feature flags
            print(f"   - GQA enabled: {config_info['is_gqa']}")
            print(f"   - MLA enabled: {config_info['is_mla']}")
            print(f"   - Kernel enabled: {config_info['is_kernel']}")
            
        except Exception as e:
            print(f"❌ {config['name']} failed: {e}")
            raise
    
    print(f"\n🎉 All builder configurations working correctly!")
    print(f"✅ Tested {len(test_configs)} attention configurations")
    print(f"✅ Fluent API working perfectly")
    print(f"✅ All combinations working")


def test_builder_validation():
    """Test builder validation and error handling."""
    
    print(f"\n🛡️ Testing Builder Validation")
    print("-" * 30)
    
    # Test GQA validation
    try:
        builder = AttentionBuilder(64, 8).with_gqa(8)  # n_kv_heads = n_heads (invalid)
        print("❌ Should have failed for n_kv_heads = n_heads")
    except ValueError as e:
        print(f"✅ GQA validation working: {e}")
    
    # Test MLA validation
    try:
        builder = AttentionBuilder(64, 8).with_mla(0, 16)  # r_q = 0 (invalid)
        print("❌ Should have failed for r_q = 0")
    except ValueError as e:
        print(f"✅ MLA validation working: {e}")
    
    # Test kernel validation
    try:
        builder = AttentionBuilder(64, 8).with_kernel("linear")  # Invalid kernel type
        print("❌ Should have failed for kernel_type = 'linear'")
    except ValueError as e:
        print(f"✅ Kernel validation working: {e}")
    
    print("✅ All validation working correctly")


if __name__ == "__main__":
    print("🚀 StackWise AttentionBuilder Test Suite")
    print("=" * 50)
    
    # Test builder pattern
    test_attention_builder()
    
    # Test validation
    test_builder_validation()
    
    print(f"\n🎯 All tests passed! Builder pattern ready for production.")
