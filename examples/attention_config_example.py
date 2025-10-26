#!/usr/bin/env python3
"""
Attention Configuration Example

This example demonstrates the updated attention configuration:
- attention_preset: "bert_style", "efficient_gqa", "mla_attention", etc. (GQA is determined by n_kv_heads)
- kernel_type: "linear", "gaussian", "laplacian", "uniform"
- How to create attention modules from configuration

Usage:
    python examples/attention_config_example.py
"""

import sys
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.attention.attention import CoreAttention
from config.base import ModelConfig, StackWiseConfig, AttentionConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_mha_attention():
    """Example 1: Multi-Head Attention (MHA)"""
    logger.info("🔧 Example 1: Multi-Head Attention (MHA)")
    logger.info("=" * 50)
    
    # Create MHA configuration
    config = ModelConfig(
        d_model=512,
        n_heads=8,
        n_kv_heads=8,  # Same as n_heads for standard MHA
        attention_preset="bert_style",
        dropout=0.1
    )
    
    # Create attention module from config
    attention = CoreAttention.from_config(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    with torch.no_grad():
        output = attention(x)
    
    logger.info(f"✅ MHA Attention created:")
    logger.info(f"   Input shape: {x.shape}")
    logger.info(f"   Output shape: {output.shape}")
    logger.info(f"   n_heads: {config.n_heads}")
    logger.info(f"   n_kv_heads: {config.n_kv_heads}")
    logger.info(f"   attention_preset: {config.attention_preset}")
    logger.info(f"   Parameters: {sum(p.numel() for p in attention.parameters()):,}")
    
    return attention


def example_2_gqa_attention():
    """Example 2: Grouped Query Attention (GQA) - determined by n_kv_heads < n_heads"""
    logger.info("\n🔧 Example 2: Grouped Query Attention (GQA)")
    logger.info("=" * 50)
    
    # Create GQA configuration (GQA is determined by n_kv_heads < n_heads)
    config = ModelConfig(
        d_model=512,
        n_heads=8,
        n_kv_heads=2,  # Fewer KV heads creates GQA
        attention_preset="efficient_gqa",  # Still MHA, but with GQA grouping
        dropout=0.1
    )
    
    # Create attention module from config
    attention = CoreAttention.from_config(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    with torch.no_grad():
        output = attention(x)
    
    logger.info(f"✅ GQA Attention created:")
    logger.info(f"   Input shape: {x.shape}")
    logger.info(f"   Output shape: {output.shape}")
    logger.info(f"   n_heads: {config.n_heads}")
    logger.info(f"   n_kv_heads: {config.n_kv_heads}")
    logger.info(f"   Group size: {config.n_heads // config.n_kv_heads}")
    logger.info(f"   attention_preset: {config.attention_preset}")
    logger.info(f"   Parameters: {sum(p.numel() for p in attention.parameters()):,}")
    
    return attention


def example_3_mla_attention():
    """Example 3: Multi-Latent Attention (MLA)"""
    logger.info("\n🔧 Example 3: Multi-Latent Attention (MLA)")
    logger.info("=" * 50)
    
    # Create MLA configuration
    config = ModelConfig(
        d_model=512,
        n_heads=8,
        n_kv_heads=8,
        attention_preset="custom",  # Use custom for MLA
        attention_custom=AttentionConfig(
            attention_type="mla",
            mla_rq=256,  # Low-rank dimension for queries
            mla_rkv=128,  # Low-rank dimension for keys/values
            kernel_dim=64
        ),
        dropout=0.1
    )
    
    # Create attention module from config
    attention = CoreAttention.from_config(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    with torch.no_grad():
        output = attention(x)
    
    logger.info(f"✅ MLA Attention created:")
    logger.info(f"   Input shape: {x.shape}")
    logger.info(f"   Output shape: {output.shape}")
    logger.info(f"   n_heads: {config.n_heads}")
    logger.info(f"   n_kv_heads: {config.n_kv_heads}")
    logger.info(f"   mla_rq: {config.attention_custom.mla_rq}")
    logger.info(f"   mla_rkv: {config.attention_custom.mla_rkv}")
    logger.info(f"   attention_preset: {config.attention_preset}")
    logger.info(f"   Parameters: {sum(p.numel() for p in attention.parameters()):,}")
    
    return attention


def example_4_kernel_attention():
    """Example 4: Kernel-based Attention (Gaussian)"""
    logger.info("\n🔧 Example 4: Kernel-based Attention (Gaussian)")
    logger.info("=" * 50)
    
    # Create kernel attention configuration
    config = ModelConfig(
        d_model=512,
        n_heads=8,
        n_kv_heads=8,
        attention_preset="custom",  # Use custom for kernel attention
        attention_custom=AttentionConfig(
            attention_type="mha",
            kernel_type="gaussian",
            kernel_dim=64  # Kernel feature dimension
        ),
        dropout=0.1
    )
    
    # Create attention module from config
    attention = CoreAttention.from_config(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    with torch.no_grad():
        output = attention(x)
    
    logger.info(f"✅ Kernel Attention created:")
    logger.info(f"   Input shape: {x.shape}")
    logger.info(f"   Output shape: {output.shape}")
    logger.info(f"   n_heads: {config.n_heads}")
    logger.info(f"   n_kv_heads: {config.n_kv_heads}")
    logger.info(f"   attention_preset: {config.attention_preset}")
    logger.info(f"   kernel_dim: {config.attention_custom.kernel_dim}")
    logger.info(f"   Parameters: {sum(p.numel() for p in attention.parameters()):,}")
    
    # Get kernel information
    kernel_info = attention.get_kernel_info()
    logger.info(f"   Kernel info: {kernel_info}")
    
    return attention


def example_5_full_config():
    """Example 5: Full StackWise Configuration"""
    logger.info("\n🔧 Example 5: Full StackWise Configuration")
    logger.info("=" * 50)
    
    # Create full configuration
    config = StackWiseConfig.from_yaml("../config.yaml")
    
    # Update attention configuration
    config.model.attention_preset = "mla_attention"
    config.model.mla_rq = 256
    config.model.mla_rkv = 128
    config.model.kernel_type = "gaussian"
    config.model.kernel_dim = 64
    
    # Create attention module from full config
    attention = CoreAttention.from_config(config.model)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.model.d_model)
    
    with torch.no_grad():
        output = attention(x)
    
    logger.info(f"✅ Full Config Attention created:")
    logger.info(f"   Input shape: {x.shape}")
    logger.info(f"   Output shape: {output.shape}")
    logger.info(f"   attention_preset: {config.model.attention_preset}")
    logger.info(f"   n_heads: {config.model.n_heads}")
    logger.info(f"   n_kv_heads: {config.model.n_kv_heads}")
    logger.info(f"   kernel_type: {config.model.attention_custom.kernel_type if config.model.attention_custom else 'linear'}")
    logger.info(f"   Parameters: {sum(p.numel() for p in attention.parameters()):,}")
    
    return attention


def main():
    """Run all examples"""
    logger.info("🚀 Attention Configuration Examples")
    logger.info("=" * 60)
    
    # Run examples
    example_1_mha_attention()
    example_2_gqa_attention()
    example_3_mla_attention()
    example_4_kernel_attention()
    example_5_full_config()
    
    logger.info("\n✅ All examples completed successfully!")
    logger.info("\nKey Points:")
    logger.info("- attention_preset: 'bert_style', 'efficient_gqa', 'mla_attention', etc.")
    logger.info("- GQA is determined by n_kv_heads < n_heads")
    logger.info("- kernel_type: 'linear', 'gaussian', 'laplacian', 'uniform'")
    logger.info("- Use CoreAttention.from_config() to create from configuration")


if __name__ == "__main__":
    main()
