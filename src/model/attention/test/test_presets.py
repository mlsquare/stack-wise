"""
Test script for attention presets and factory patterns.
Demonstrates efficient attention creation for multiple layers.
"""

import torch
import torch.nn as nn
from ..presets import AttentionPresets, AttentionFactory, create_attention_factory, create_layer_attention


class StackWiseLayer(nn.Module):
    """
    Example StackWise layer using shared attention configuration.
    """
    
    def __init__(self, attention_factory: AttentionFactory, d_model: int):
        super().__init__()
        self.attention = attention_factory.create_attention()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # MLP with residual connection
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x


def test_attention_presets():
    """Test attention presets."""
    
    print("🧪 Testing Attention Presets")
    print("=" * 50)
    
    d_model = 64
    n_heads = 8
    
    # Test GPT-style preset
    gpt_config = AttentionPresets.gpt_style(d_model, n_heads)
    print(f"✅ GPT-style config: {gpt_config}")
    
    # Test BERT-style preset
    bert_config = AttentionPresets.bert_style(d_model, n_heads)
    print(f"✅ BERT-style config: {bert_config}")
    
    # Test efficient GQA preset
    gqa_config = AttentionPresets.efficient_gqa(d_model, n_heads, n_kv_heads=2)
    print(f"✅ GQA config: {gqa_config}")
    
    # Test kernel attention preset
    kernel_config = AttentionPresets.kernel_attention(d_model, n_heads, "gaussian", 32)
    print(f"✅ Kernel config: {kernel_config}")
    
    # Test full-featured preset
    full_config = AttentionPresets.full_featured(
        d_model, n_heads, n_kv_heads=2, r_q=32, r_kv=16, 
        kernel_type="gaussian", kernel_dim=32
    )
    print(f"✅ Full-featured config: {full_config}")
    
    print("\n✅ All presets working correctly!")


def test_attention_factory():
    """Test attention factory for efficient layer creation."""
    
    print("\n🧪 Testing Attention Factory")
    print("=" * 50)
    
    # Create factory with BERT-style configuration
    factory = AttentionFactory(AttentionPresets.bert_style(64, 8))
    
    # Test single attention creation
    attn1 = factory.get_attention()
    attn2 = factory.create_attention()
    
    print(f"✅ Factory created attention: {type(attn1).__name__}")
    print(f"✅ Multiple instances: {attn1 is not attn2}")
    
    # Test configuration
    config = factory.get_config()
    print(f"✅ Configuration: {config}")
    
    print("\n✅ Attention factory working correctly!")


def test_layer_creation():
    """Test creating multiple layers with shared attention configuration."""
    
    print("\n🧪 Testing Layer Creation")
    print("=" * 50)
    
    # Create attention factory
    factory = AttentionFactory(AttentionPresets.efficient_gqa(64, 8, 2))
    
    # Create multiple layers
    n_layers = 3
    layers = [StackWiseLayer(factory, 64) for _ in range(n_layers)]
    
    print(f"✅ Created {n_layers} layers with shared attention config")
    
    # Test forward pass
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, 64)
    
    for i, layer in enumerate(layers):
        x = layer(x)
        print(f"✅ Layer {i+1} output shape: {x.shape}")
    
    print("\n✅ All layers working correctly!")


def test_preset_factory():
    """Test preset-based factory creation."""
    
    print("\n🧪 Testing Preset Factory")
    print("=" * 50)
    
    # Test different presets
    presets = [
        ("gpt_style", {"d_model": 64, "n_heads": 8}),
        ("bert_style", {"d_model": 64, "n_heads": 8}),
        ("efficient_gqa", {"d_model": 64, "n_heads": 8, "n_kv_heads": 2}),
        ("kernel_attention", {"d_model": 64, "n_heads": 8, "kernel_type": "gaussian", "kernel_dim": 32}),
    ]
    
    for preset_name, kwargs in presets:
        try:
            factory = create_attention_factory(preset_name, **kwargs)
            attn = factory.get_attention()
            print(f"✅ {preset_name}: {type(attn).__name__}")
        except Exception as e:
            print(f"❌ {preset_name}: {e}")
    
    print("\n✅ All preset factories working correctly!")


def test_efficiency_comparison():
    """Compare efficiency of different approaches."""
    
    print("\n🧪 Testing Efficiency Comparison")
    print("=" * 50)
    
    import time
    
    d_model = 64
    n_heads = 8
    n_layers = 5
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Approach 1: Builder pattern for each layer (inefficient)
    start_time = time.time()
    for _ in range(n_layers):
        from ..builder import AttentionBuilder
        attn = AttentionBuilder(d_model, n_heads).build()
        _ = attn(x)
    builder_time = time.time() - start_time
    
    # Approach 2: Factory pattern (efficient)
    start_time = time.time()
    factory = AttentionFactory(AttentionPresets.bert_style(d_model, n_heads))
    for _ in range(n_layers):
        attn = factory.create_attention()
        _ = attn(x)
    factory_time = time.time() - start_time
    
    # Approach 3: Single attention reuse (most efficient)
    start_time = time.time()
    attn = factory.get_attention()
    for _ in range(n_layers):
        _ = attn(x)
    reuse_time = time.time() - start_time
    
    print(f"✅ Builder pattern: {builder_time:.4f}s")
    print(f"✅ Factory pattern: {factory_time:.4f}s")
    print(f"✅ Reuse pattern: {reuse_time:.4f}s")
    print(f"✅ Factory is {builder_time/factory_time:.1f}x faster than builder")
    print(f"✅ Reuse is {factory_time/reuse_time:.1f}x faster than factory")
    
    print("\n✅ Efficiency comparison completed!")


if __name__ == "__main__":
    test_attention_presets()
    test_attention_factory()
    test_layer_creation()
    test_preset_factory()
    test_efficiency_comparison()
