"""
Usage examples for StackWise attention system.
Demonstrates efficient patterns for layer-wise attention.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from .presets import AttentionPresets, AttentionFactory, create_attention_factory
    from .builder import AttentionBuilder
except ImportError:
    # Fallback for direct execution
    from model.attention.presets import AttentionPresets, AttentionFactory, create_attention_factory
    from model.attention.builder import AttentionBuilder


def example_1_basic_usage():
    """
    Example 1: Basic usage with presets.
    Recommended for most use cases.
    """
    print("📚 Example 1: Basic Usage with Presets")
    print("=" * 50)
    
    # Create attention factory with BERT-style configuration
    factory = AttentionFactory(AttentionPresets.bert_style(d_model=512, n_heads=8))
    
    # Create multiple layers efficiently
    n_blocks = 6
    layers = []
    
    for i in range(n_blocks):
        # Each layer gets its own attention instance
        attention = factory.create_attention()
        layers.append(attention)
        print(f"✅ Layer {i+1}: {type(attention).__name__}")
    
    print(f"✅ Created {n_blocks} layers efficiently!")
    return layers


def example_2_preset_factory():
    """
    Example 2: Using preset factory for common configurations.
    """
    print("\n📚 Example 2: Preset Factory")
    print("=" * 50)
    
    # Create different types of attention factories
    factories = {
        "GPT": create_attention_factory("gpt_style", d_model=256, n_heads=4),
        "BERT": create_attention_factory("bert_style", d_model=256, n_heads=4),
        "GQA": create_attention_factory("efficient_gqa", d_model=256, n_heads=4, n_kv_heads=2),
        "Kernel": create_attention_factory("kernel_attention", d_model=256, n_heads=4, 
                                           kernel_type="gaussian", kernel_dim=32)
    }
    
    for name, factory in factories.items():
        attn = factory.get_attention()
        config = factory.get_config()
        print(f"✅ {name}: {config['attention_mode']} mode, {config['kernel_type']} kernel")
    
    return factories


def example_3_layer_wise_training():
    """
    Example 3: Layer-wise training with shared attention configuration.
    """
    print("\n📚 Example 3: Layer-wise Training")
    print("=" * 50)
    
    class StackWiseLayer(nn.Module):
        def __init__(self, attention_factory: AttentionFactory, d_model: int):
            super().__init__()
            self.attention = attention_factory.create_attention()
            self.norm = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
        
        def forward(self, x):
            # Self-attention
            attn_out = self.attention(x)
            x = self.norm(x + attn_out)
            
            # MLP
            mlp_out = self.mlp(x)
            x = x + mlp_out
            
            return x
    
    # Create attention factory for all layers
    factory = AttentionFactory(AttentionPresets.efficient_gqa(d_model=64, n_heads=8, n_kv_heads=2))
    
    # Create layers
    n_blocks = 4
    layers = [StackWiseLayer(factory, 64) for _ in range(n_blocks)]
    
    # Test forward pass
    x = torch.randn(2, 16, 64)
    for i, layer in enumerate(layers):
        x = layer(x)
        print(f"✅ Layer {i+1} output shape: {x.shape}")
    
    print(f"✅ Created {n_blocks} layers for layer-wise training!")
    return layers


def example_4_attention_mode_switching():
    """
    Example 4: Switching attention modes between training and inference.
    """
    print("\n📚 Example 4: Attention Mode Switching")
    print("=" * 50)
    
    # Create attention with bidirectional mode (for training)
    factory = AttentionFactory(AttentionPresets.bert_style(d_model=64, n_heads=8))
    attention = factory.get_attention()
    
    print(f"✅ Initial mode: {attention.get_attention_mode()}")
    
    # Switch to causal mode (for inference)
    attention.set_attention_mode("causal")
    print(f"✅ Switched to: {attention.get_attention_mode()}")
    
    # Test both modes
    x = torch.randn(1, 8, 64)
    
    # Bidirectional attention
    attention.set_attention_mode("bidirectional")
    out_bidirectional = attention(x)
    print(f"✅ Bidirectional output shape: {out_bidirectional.shape}")
    
    # Causal attention
    attention.set_attention_mode("causal")
    out_causal = attention(x)
    print(f"✅ Causal output shape: {out_causal.shape}")
    
    print("✅ Attention mode switching working!")
    return attention


def example_5_advanced_configurations():
    """
    Example 5: Advanced configurations with Builder Pattern.
    """
    print("\n📚 Example 5: Advanced Configurations")
    print("=" * 50)
    
    # Create complex attention configuration
    builder = (AttentionBuilder(d_model=128, n_heads=8)
              .with_gqa(n_kv_heads=2)
              .with_mla(r_q=64, r_kv=32)
              .with_kernel("gaussian", kernel_dim=32))
    
    # Get configuration
    config = builder.get_config()
    print(f"✅ Complex config: GQA={config['is_gqa']}, MLA={config['is_mla']}, Kernel={config['is_kernel']}")
    
    # Create attention
    attention = builder.build()
    print(f"✅ Created: {type(attention).__name__}")
    
    # Test forward pass
    x = torch.randn(2, 16, 128)
    output = attention(x)
    print(f"✅ Output shape: {output.shape}")
    
    print("✅ Advanced configuration working!")
    return attention


def example_6_efficiency_comparison():
    """
    Example 6: Efficiency comparison of different approaches.
    """
    print("\n📚 Example 6: Efficiency Comparison")
    print("=" * 50)
    
    import time
    
    d_model = 64
    n_heads = 8
    n_blocks = 10
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Approach 1: Builder pattern (inefficient for multiple layers)
    start_time = time.time()
    for _ in range(n_blocks):
        builder = AttentionBuilder(d_model, n_heads)
        attn = builder.build()
        _ = attn(x)
    builder_time = time.time() - start_time
    
    # Approach 2: Factory pattern (efficient)
    start_time = time.time()
    factory = AttentionFactory(AttentionPresets.bert_style(d_model, n_heads))
    for _ in range(n_blocks):
        attn = factory.create_attention()
        _ = attn(x)
    factory_time = time.time() - start_time
    
    # Approach 3: Single attention reuse (most efficient)
    start_time = time.time()
    attn = factory.get_attention()
    for _ in range(n_blocks):
        _ = attn(x)
    reuse_time = time.time() - start_time
    
    print(f"✅ Builder pattern: {builder_time:.4f}s")
    print(f"✅ Factory pattern: {factory_time:.4f}s")
    print(f"✅ Reuse pattern: {reuse_time:.4f}s")
    print(f"✅ Factory is {builder_time/factory_time:.1f}x faster than builder")
    print(f"✅ Reuse is {factory_time/reuse_time:.1f}x faster than factory")
    
    print("\n💡 Recommendation: Use Factory pattern for multiple layers!")
    return builder_time, factory_time, reuse_time


def main():
    """Run all examples."""
    print("🚀 StackWise Attention Usage Examples")
    print("=" * 60)
    
    # Run all examples
    example_1_basic_usage()
    example_2_preset_factory()
    example_3_layer_wise_training()
    example_4_attention_mode_switching()
    example_5_advanced_configurations()
    example_6_efficiency_comparison()
    
    print("\n🎉 All examples completed successfully!")
    print("\n📋 Summary of Recommended Patterns:")
    print("1. Use AttentionFactory for multiple layers")
    print("2. Use presets for common configurations")
    print("3. Use Builder pattern for one-off complex configurations")
    print("4. Switch attention modes between training and inference")
    print("5. Reuse attention instances when possible for maximum efficiency")


if __name__ == "__main__":
    main()
