"""
Test script to demonstrate proper handling of varying mask patterns in batches.
"""

import torch

def test_varying_mask_patterns():
    """Test that the loss computation handles varying mask patterns correctly."""
    
    # Simulate a batch with different mask patterns
    batch_size, seq_len, vocab_size = 3, 5, 1000
    
    # Create logits (batch_size, seq_len, vocab_size)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Create targets (batch_size, seq_len)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create masks with varying patterns per sample
    masks = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    # Sample 0: mask positions [0, 2] (2 masked positions)
    masks[0, [0, 2]] = True
    
    # Sample 1: mask positions [1, 3, 4] (3 masked positions)  
    masks[1, [1, 3, 4]] = True
    
    # Sample 2: mask positions [0, 1, 2, 3] (4 masked positions)
    masks[2, [0, 1, 2, 3]] = True
    
    print("Mask patterns:")
    print("Sample 0:", masks[0].tolist(), "->", masks[0].sum().item(), "masked positions")
    print("Sample 1:", masks[1].tolist(), "->", masks[1].sum().item(), "masked positions") 
    print("Sample 2:", masks[2].tolist(), "->", masks[2].sum().item(), "masked positions")
    print()
    
    # Test the efficient loss computation
    def compute_efficient_loss(logits, targets, masks):
        """Efficient loss computation that handles varying mask patterns."""
        # Flatten for easier indexing
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        targets_flat = targets.view(-1)  # (batch_size * seq_len,)
        masks_flat = masks.view(-1)  # (batch_size * seq_len,)
        
        # Get indices where mask is True
        mask_indices = masks_flat.nonzero(as_tuple=False).squeeze(-1)  # (num_masked,)
        
        if len(mask_indices) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Extract logits and targets only for masked positions
        masked_logits = logits_flat[mask_indices]  # (num_masked, vocab_size)
        masked_targets = targets_flat[mask_indices]  # (num_masked,)
        
        # Compute cross-entropy loss only on masked positions
        loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets.long())
        return loss
    
    # Compute loss
    loss = compute_efficient_loss(logits, targets, masks)
    
    print(f"Total masked positions: {masks.sum().item()}")
    print(f"Loss computed on {masks.sum().item()} positions out of {batch_size * seq_len} total positions")
    print(f"Efficiency: {masks.sum().item() / (batch_size * seq_len) * 100:.1f}% of positions used")
    print(f"Loss value: {loss.item():.4f}")
    
    # Verify that we're only using masked positions
    mask_indices = masks.view(-1).nonzero(as_tuple=False).squeeze(-1)
    print(f"Mask indices: {mask_indices.tolist()}")
    print(f"Number of masked positions: {len(mask_indices)}")
    
    return loss

if __name__ == "__main__":
    print("Testing varying mask patterns in batches...")
    print("=" * 50)
    loss = test_varying_mask_patterns()
    print("=" * 50)
    print("✅ Test completed successfully!")
