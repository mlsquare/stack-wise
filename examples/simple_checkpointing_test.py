#!/usr/bin/env python3
"""
Simple Checkpointing Test for Stack-Wise Progressive Training

This test verifies the checkpointing functionality without external dependencies.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleBlock(nn.Module):
    """Simple block for testing"""
    
    def __init__(self, d_model: int = 64, d_ff: int = 128):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class SimpleStack(nn.Module):
    """Simple stack for testing"""
    
    def __init__(self, n_blocks: int = 2, d_model: int = 64, d_ff: int = 128):
        super().__init__()
        self.blocks = nn.ModuleList([
            SimpleBlock(d_model, d_ff) for _ in range(n_blocks)
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SimpleRack(nn.Module):
    """Simple rack for testing"""
    
    def __init__(self, n_stacks: int = 2, n_blocks: int = 2, d_model: int = 64, d_ff: int = 128, vocab_size: int = 1000):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.stacks = nn.ModuleList([
            SimpleStack(n_blocks, d_model, d_ff) for _ in range(n_stacks)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for stack in self.stacks:
            x = stack(x)
        return self.lm_head(x)


class SimpleDataset(Dataset):
    """Simple dataset for testing"""
    
    def __init__(self, num_samples: int = 50, seq_len: int = 16, vocab_size: int = 1000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        target_ids = input_ids.clone()
        return input_ids, target_ids


def test_basic_checkpointing():
    """Test basic PyTorch checkpointing functionality"""
    logger.info("=== Testing Basic Checkpointing ===")
    
    # Create simple model
    model = SimpleRack(n_stacks=2, n_blocks=2, d_model=64, d_ff=128, vocab_size=1000)
    
    # Create some dummy data
    input_ids = torch.randint(0, 1000, (2, 16))
    target_ids = torch.randint(0, 1000, (2, 16))
    
    # Forward pass
    outputs = model(input_ids)
    loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
    
    logger.info(f"Initial loss: {loss.item():.4f}")
    
    # Save checkpoint
    checkpoint_path = "./test_checkpoints/simple_model.pt"
    Path("./test_checkpoints").mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss': loss.item(),
        'timestamp': '2024-01-01T00:00:00'
    }, checkpoint_path)
    
    logger.info(f"Saved model to: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Verify loading
    outputs_loaded = model(input_ids)
    loss_loaded = nn.CrossEntropyLoss()(outputs_loaded.view(-1, outputs_loaded.size(-1)), target_ids.view(-1))
    
    logger.info(f"Loaded loss: {loss_loaded.item():.4f}")
    
    # Check if losses match
    if abs(loss.item() - loss_loaded.item()) < 1e-6:
        logger.info("✅ Basic checkpointing test passed!")
        return True
    else:
        logger.error("❌ Basic checkpointing test failed!")
        return False


def test_stack_checkpointing():
    """Test stack-level checkpointing"""
    logger.info("=== Testing Stack Checkpointing ===")
    
    # Create rack with multiple stacks
    rack = SimpleRack(n_stacks=3, n_blocks=2, d_model=64, d_ff=128, vocab_size=1000)
    
    # Save individual stacks
    stack_paths = []
    for i, stack in enumerate(rack.stacks):
        stack_path = f"./test_checkpoints/stack_{i}.pt"
        torch.save({
            'stack_state_dict': stack.state_dict(),
            'stack_idx': i,
            'timestamp': '2024-01-01T00:00:00'
        }, stack_path)
        stack_paths.append(stack_path)
        logger.info(f"Saved stack {i} to: {stack_path}")
    
    # Save embeddings and lm_head separately
    embeddings_path = "./test_checkpoints/embeddings.pt"
    lm_head_path = "./test_checkpoints/lm_head.pt"
    torch.save(rack.embeddings.state_dict(), embeddings_path)
    torch.save(rack.lm_head.state_dict(), lm_head_path)
    logger.info(f"Saved embeddings to: {embeddings_path}")
    logger.info(f"Saved lm_head to: {lm_head_path}")
    
    # Create new rack and load stacks
    new_rack = SimpleRack(n_stacks=3, n_blocks=2, d_model=64, d_ff=128, vocab_size=1000)
    
    for i, stack_path in enumerate(stack_paths):
        checkpoint = torch.load(stack_path, map_location='cpu')
        new_rack.stacks[i].load_state_dict(checkpoint['stack_state_dict'])
        logger.info(f"Loaded stack {i} from: {stack_path}")
    
    # Load embeddings and lm_head
    new_rack.embeddings.load_state_dict(torch.load(embeddings_path, map_location='cpu'))
    new_rack.lm_head.load_state_dict(torch.load(lm_head_path, map_location='cpu'))
    logger.info("Loaded embeddings and lm_head")
    
    # Test with dummy data
    input_ids = torch.randint(0, 1000, (2, 16))
    target_ids = torch.randint(0, 1000, (2, 16))
    
    # Compare outputs
    outputs_original = rack(input_ids)
    outputs_loaded = new_rack(input_ids)
    
    logger.info(f"Original output shape: {outputs_original.shape}")
    logger.info(f"Loaded output shape: {outputs_loaded.shape}")
    logger.info(f"Max difference: {torch.max(torch.abs(outputs_original - outputs_loaded)).item()}")
    
    if torch.allclose(outputs_original, outputs_loaded, atol=1e-6):
        logger.info("✅ Stack checkpointing test passed!")
        return True
    else:
        logger.error("❌ Stack checkpointing test failed!")
        return False


def test_rack_checkpointing():
    """Test rack-level checkpointing"""
    logger.info("=== Testing Rack Checkpointing ===")
    
    # Create rack
    rack = SimpleRack(n_stacks=2, n_blocks=3, d_model=64, d_ff=128, vocab_size=1000)
    
    # Save complete rack
    rack_path = "./test_checkpoints/complete_rack.pt"
    torch.save({
        'rack_state_dict': rack.state_dict(),
        'n_stacks': 2,
        'n_blocks': 3,
        'd_model': 64,
        'd_ff': 128,
        'vocab_size': 1000,
        'timestamp': '2024-01-01T00:00:00'
    }, rack_path)
    
    logger.info(f"Saved complete rack to: {rack_path}")
    
    # Load complete rack
    checkpoint = torch.load(rack_path, map_location='cpu')
    new_rack = SimpleRack(
        n_stacks=checkpoint['n_stacks'],
        n_blocks=checkpoint['n_blocks'],
        d_model=checkpoint['d_model'],
        d_ff=checkpoint['d_ff'],
        vocab_size=checkpoint['vocab_size']
    )
    new_rack.load_state_dict(checkpoint['rack_state_dict'])
    
    logger.info("Loaded complete rack")
    
    # Test with dummy data
    input_ids = torch.randint(0, 1000, (2, 16))
    target_ids = torch.randint(0, 1000, (2, 16))
    
    # Compare outputs
    outputs_original = rack(input_ids)
    outputs_loaded = new_rack(input_ids)
    
    if torch.allclose(outputs_original, outputs_loaded, atol=1e-6):
        logger.info("✅ Rack checkpointing test passed!")
        return True
    else:
        logger.error("❌ Rack checkpointing test failed!")
        return False


def main():
    """Main function to run all checkpointing tests"""
    logger.info("Starting Simple Checkpointing Tests")
    
    # Create test directory
    Path("./test_checkpoints").mkdir(exist_ok=True)
    
    try:
        # Run tests
        basic_success = test_basic_checkpointing()
        stack_success = test_stack_checkpointing()
        rack_success = test_rack_checkpointing()
        
        # Summary
        logger.info("=== Test Summary ===")
        logger.info(f"Basic checkpointing: {'✅' if basic_success else '❌'}")
        logger.info(f"Stack checkpointing: {'✅' if stack_success else '❌'}")
        logger.info(f"Rack checkpointing: {'✅' if rack_success else '❌'}")
        
        if all([basic_success, stack_success, rack_success]):
            logger.info("🎉 All checkpointing tests passed!")
            return True
        else:
            logger.error("❌ Some checkpointing tests failed!")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
