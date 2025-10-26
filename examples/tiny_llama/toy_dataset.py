"""
Toy dataset generator for CLM (Causal Language Modeling) training.
Generates sequences for next-token prediction without masking.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class CLMToyDataset(Dataset):
    """
    Toy dataset for CLM training.
    Generates random sequences where each token predicts the next token.
    """
    
    def __init__(self, 
                 vocab_size: int = 32000,
                 max_length: int = 128,
                 num_samples: int = 10000,
                 seed: int = 42):
        """
        Initialize CLM toy dataset.
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_samples = num_samples
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate samples
        self.samples = self._generate_samples()
        
        logger.info(f"Generated {len(self.samples)} samples for CLM task")
        logger.info(f"Vocabulary size: {vocab_size}, Sequence length: {max_length}")
    
    def _generate_samples(self) -> List[Dict[str, torch.Tensor]]:
        """Generate CLM samples."""
        samples = []
        
        for i in range(self.num_samples):
            # Generate random sequence length (between 16 and max_length)
            seq_len = random.randint(16, self.max_length)
            
            # Generate random token sequence
            input_ids = torch.randint(0, self.vocab_size, (seq_len,))
            
            # For CLM, target_ids are input_ids shifted by 1
            # target_ids[i] = input_ids[i+1] for i < seq_len-1
            target_ids = torch.zeros_like(input_ids)
            target_ids[:-1] = input_ids[1:]  # Shift by 1
            target_ids[-1] = 0  # Last token has no target (or use special token)
            
            # Create attention mask (all True for causal attention)
            attention_mask = torch.ones(seq_len, dtype=torch.bool)
            
            sample = {
                'input_ids': input_ids,
                'target_ids': target_ids,
                'attention_mask': attention_mask,
                'task': 'clm'
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]

def create_clm_datasets(vocab_size: int = 32000,
                       max_length: int = 128,
                       train_samples: int = 12000,
                       val_samples: int = 1500,
                       test_samples: int = 1500,
                       batch_size: int = 8,
                       num_workers: int = 0,
                       pin_memory: bool = True,
                       shuffle: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for CLM.
    
    Args:
        vocab_size: Size of vocabulary
        max_length: Maximum sequence length
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    
    # Create datasets
    train_dataset = CLMToyDataset(
        vocab_size=vocab_size,
        max_length=max_length,
        num_samples=train_samples,
        seed=42
    )
    
    val_dataset = CLMToyDataset(
        vocab_size=vocab_size,
        max_length=max_length,
        num_samples=val_samples,
        seed=43
    )
    
    test_dataset = CLMToyDataset(
        vocab_size=vocab_size,
        max_length=max_length,
        num_samples=test_samples,
        seed=44
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=clm_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=clm_collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=clm_collate_fn
    )
    
    logger.info(f"Created datasets: {train_samples} train, {val_samples} val, {test_samples} test")
    
    return train_dataloader, val_dataloader, test_dataloader

def clm_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for CLM batches.
    Pads sequences to the same length with left padding for causal attention.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched tensors
    """
    # Get maximum sequence length in batch
    max_len = max(sample['input_ids'].shape[0] for sample in batch)
    
    batch_size = len(batch)
    
    # Initialize batched tensors
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    target_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    # Fill tensors with left padding
    for i, sample in enumerate(batch):
        seq_len = sample['input_ids'].shape[0]
        
        # Left padding: put sequence at the end
        input_ids[i, -seq_len:] = sample['input_ids']
        target_ids[i, -seq_len:] = sample['target_ids']
        attention_mask[i, -seq_len:] = sample['attention_mask']
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'attention_mask': attention_mask,
        'task': 'clm'
    }

if __name__ == "__main__":
    # Test the dataset
    logging.basicConfig(level=logging.INFO)
    
    # Create small test dataset
    train_loader, val_loader, test_loader = create_clm_datasets(
        vocab_size=1000,
        max_length=32,
        train_samples=100,
        val_samples=20,
        test_samples=20,
        batch_size=4
    )
    
    # Test a batch
    batch = next(iter(train_loader))
    print("CLM Dataset Test:")
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Target shape: {batch['target_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Task: {batch['task']}")
    
    # Show first sample
    print(f"\nFirst sample:")
    print(f"Input IDs: {batch['input_ids'][0]}")
    print(f"Target IDs: {batch['target_ids'][0]}")
    print(f"Attention mask: {batch['attention_mask'][0]}")
