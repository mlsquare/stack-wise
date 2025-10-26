#!/usr/bin/env python3
"""
MLGKA Text Classification Example

This example demonstrates how to use MLGKALayer (complete transformer blocks)
to build a simple text classification model with wandb monitoring.

MLGKALayer combines:
- Multi-Latent Attention (MLA)
- Grouped Query Attention (GQA) 
- Laplacian Kernel Attention
- SwiGLU Feed-Forward Network
- Layer Normalization and Residual Connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from typing import Optional
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    # python-dotenv not available, continue without loading .env
    pass

from model.layers import MLGKALayer
from config.base import AttentionConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLGKATextClassifier(nn.Module):
    """
    Text classification model using MLGKA layers (complete transformer blocks).
    
    Architecture:
    - Input embedding
    - Multiple MLGKA layers (complete transformer blocks)
    - Global average pooling
    - Classification head
    """
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 n_heads: int = 8,
                 n_kv_heads: int = 2,  # GQA: 4x reduction in K/V heads
                 n_layers: int = 6,
                 num_classes: int = 2,
                 max_seq_len: int = 512,
                 dropout: float = 0.1,
                 freeze_up_proj: bool = True):
        """
        Initialize MLGKA text classifier.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            d_ff: Feed-forward dimension
            n_heads: Number of attention heads
            n_kv_heads: Number of key-value heads (GQA when < n_heads)
            n_layers: Number of MLGKA layers
            num_classes: Number of classification classes
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            freeze_up_proj: Whether to freeze up-projection in SwiGLU
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # MLGKA layers (complete transformer blocks)
        self.layers = nn.ModuleList([
            MLGKALayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                kernel_type="laplacian",  # MLGKA uses Laplacian kernel
                kernel_dim=64,
                attention_mode="bidirectional",  # For classification
                freeze_up_proj=freeze_up_proj
            ) for _ in range(n_layers)
        ])
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len = input_ids.shape
        
        # Input embeddings
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        # Pass through MLGKA layers
        for layer in self.layers:
            x = layer(x, attention_mask=None)  # MLGKA handles its own attention patterns
        
        # Global average pooling
        if attention_mask is not None:
            # Mask out padding tokens
            mask = attention_mask.unsqueeze(-1).float()
            x = x * mask
            pooled = x.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = x.mean(dim=1)
        
        # Classification head
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits


def create_sample_data(batch_size: int = 4, seq_len: int = 128, vocab_size: int = 10000):
    """Create sample data for testing."""
    # Random input tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Random attention mask (some padding)
    attention_mask = torch.ones(batch_size, seq_len)
    for i in range(batch_size):
        # Randomly mask some tokens at the end
        pad_start = torch.randint(seq_len // 2, seq_len, (1,)).item()
        attention_mask[i, pad_start:] = 0
    
    # Random labels
    labels = torch.randint(0, 2, (batch_size,))
    
    return input_ids, attention_mask, labels


def main():
    """Main function demonstrating MLGKA text classification with wandb monitoring."""
    logger.info("🚀 MLGKA Text Classification Example")
    logger.info("=" * 50)
    
    # Initialize wandb if API key is available
    wandb_logger = None
    if os.getenv("WANDB_API_KEY"):
        try:
            import wandb
            # Check if API key is valid (not just a placeholder)
            api_key = os.getenv("WANDB_API_KEY")
            if api_key and api_key != "your_wandb_api_key_here" and len(api_key) > 10:
                wandb.init(
                    project="stack-wise-mlgka",
                    name="mlgka-text-classification",
                    tags=["mlgka", "text-classification", "example"],
                    notes="MLGKA text classification example with MLA + GQA + Laplacian kernel"
                )
                wandb_logger = wandb
                logger.info("📊 wandb initialized for monitoring")
            else:
                logger.info("🔑 WANDB_API_KEY appears to be a placeholder, running without wandb monitoring")
        except ImportError:
            logger.warning("wandb not available, continuing without monitoring")
        except Exception as e:
            logger.warning(f"wandb initialization failed: {e}")
            logger.info("Continuing without wandb monitoring")
    else:
        logger.info("🔑 WANDB_API_KEY not found, running without wandb monitoring")
    
    # Model configuration
    config = {
        "vocab_size": 10000,
        "d_model": 512,
        "d_ff": 2048,
        "n_heads": 8,
        "n_kv_heads": 2,  # GQA: 4x reduction
        "n_layers": 6,
        "num_classes": 2,
        "max_seq_len": 512,
        "dropout": 0.1,
        "freeze_up_proj": True
    }
    
    logger.info("📋 Model Configuration:")
    for key, value in config.items():
        logger.info(f"   {key}: {value}")
    
    # Log config to wandb
    if wandb_logger:
        wandb_logger.config.update(config)
    
    # Create model
    model = MLGKATextClassifier(**config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\n🏗️  Model Created:")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,}")
    logger.info(f"   MLGKA layers: {config['n_layers']}")
    logger.info(f"   Attention heads: {config['n_heads']} (GQA with {config['n_kv_heads']} K/V heads)")
    logger.info(f"   Kernel: Laplacian (dim=64)")
    
    # Log model info to wandb
    if wandb_logger:
        wandb_logger.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/layers": config['n_layers'],
            "model/attention_heads": config['n_heads'],
            "model/kv_heads": config['n_kv_heads']
        })
    
    # Create sample data
    input_ids, attention_mask, labels = create_sample_data(
        batch_size=4, 
        seq_len=128, 
        vocab_size=config["vocab_size"]
    )
    
    logger.info(f"\n📊 Sample Data:")
    logger.info(f"   Input shape: {input_ids.shape}")
    logger.info(f"   Attention mask shape: {attention_mask.shape}")
    logger.info(f"   Labels shape: {labels.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=-1)
        probabilities = F.softmax(logits, dim=-1)
    
    logger.info(f"\n🔮 Model Output:")
    logger.info(f"   Logits shape: {logits.shape}")
    logger.info(f"   Predictions: {predictions.tolist()}")
    logger.info(f"   True labels: {labels.tolist()}")
    logger.info(f"   Probabilities: {probabilities.max(dim=-1)[0].tolist()}")
    
    # Calculate accuracy
    accuracy = (predictions == labels).float().mean().item()
    logger.info(f"   Accuracy: {accuracy:.2%}")
    
    # Log results to wandb
    if wandb_logger:
        wandb_logger.log({
            "evaluation/accuracy": accuracy,
            "evaluation/predictions": predictions.tolist(),
            "evaluation/true_labels": labels.tolist(),
            "evaluation/avg_confidence": probabilities.max(dim=-1)[0].mean().item()
        })
    
    # Demonstrate different configurations
    logger.info(f"\n🔧 Configuration Examples:")
    
    # Small model
    small_model = MLGKATextClassifier(
        vocab_size=5000, d_model=256, d_ff=1024, n_heads=4, n_kv_heads=1,
        n_layers=3, num_classes=3
    )
    small_params = sum(p.numel() for p in small_model.parameters())
    logger.info(f"   Small model: {small_params:,} parameters")
    
    # Large model
    large_model = MLGKATextClassifier(
        vocab_size=50000, d_model=1024, d_ff=4096, n_heads=16, n_kv_heads=4,
        n_layers=12, num_classes=10
    )
    large_params = sum(p.numel() for p in large_model.parameters())
    logger.info(f"   Large model: {large_params:,} parameters")
    
    # Log model size comparison to wandb
    if wandb_logger:
        wandb_logger.log({
            "model_comparison/small_params": small_params,
            "model_comparison/large_params": large_params,
            "model_comparison/main_params": total_params
        })
    
    logger.info(f"\n✅ MLGKA Text Classification Example Complete!")
    logger.info(f"   MLGKA layers provide efficient attention with:")
    logger.info(f"   - Multi-Latent Attention (low-rank factorization)")
    logger.info(f"   - Grouped Query Attention (shared K/V heads)")
    logger.info(f"   - Laplacian Kernel Attention (non-linear patterns)")
    logger.info(f"   - SwiGLU Feed-Forward (efficient activation)")
    
    # Finish wandb run
    if wandb_logger:
        wandb_logger.finish()
        logger.info("📊 wandb run completed")


if __name__ == "__main__":
    main()
