#!/usr/bin/env python3
"""
Simple MLGKA Example

A minimal example showing how to use MLGKALayer as a complete transformer block
for text classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model.layers import MLGKALayer


class SimpleMLGKAClassifier(nn.Module):
    """Simple text classifier using MLGKA layers (complete transformer blocks)."""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_classes: int = 2):
        super().__init__()
        
        # Input embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # MLGKA layer (complete transformer block)
        self.mlgka_layer = MLGKALayer(
            d_model=d_model,
            d_ff=d_model * 4,  # Standard 4x expansion
            n_heads=8,
            n_kv_heads=2,  # GQA: 4x reduction in K/V heads
            kernel_type="laplacian",  # MLGKA uses Laplacian kernel
            kernel_dim=64,
            attention_mode="bidirectional",
            freeze_up_proj=True
        )
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Embedding
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        
        # MLGKA layer (complete transformer block)
        x = self.mlgka_layer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits


def main():
    """Demonstrate simple MLGKA usage."""
    print("🚀 Simple MLGKA Example")
    print("=" * 30)
    
    # Create model
    model = SimpleMLGKAClassifier(vocab_size=1000, d_model=256, num_classes=2)
    params = sum(p.numel() for p in model.parameters())
    
    print(f"📊 Model: {params:,} parameters")
    print(f"   - MLGKA layer (complete transformer block)")
    print(f"   - Multi-Latent + GQA + Laplacian kernel")
    print(f"   - SwiGLU feed-forward network")
    
    # Test with sample data
    input_ids = torch.randint(0, 1000, (2, 32))  # batch_size=2, seq_len=32
    
    with torch.no_grad():
        logits = model(input_ids)
        predictions = torch.argmax(logits, dim=-1)
    
    print(f"\n🔮 Test Results:")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Predictions: {predictions.tolist()}")
    
    print(f"\n✅ MLGKA layer works as a complete transformer block!")
    print(f"   - Combines attention + FFN + layer norm + residual")
    print(f"   - Uses MLGKA preset (MLA + GQA + Laplacian kernel)")
    print(f"   - Ready to use in any architecture")


if __name__ == "__main__":
    main()
