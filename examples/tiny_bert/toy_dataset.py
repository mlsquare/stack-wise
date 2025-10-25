#!/usr/bin/env python3
"""
Toy Dataset for Tiny BERT Training

This module provides a synthetic dataset for training and testing tiny BERT models.
It generates realistic text data with configurable parameters.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class ToyDataset(Dataset):
    """
    Toy dataset for tiny BERT training.
    
    Generates synthetic text data with configurable parameters:
    - Vocabulary size
    - Sequence length
    - Number of samples
    - Task type (MLM, CLM, classification)
    """
    
    def __init__(self, 
                 num_samples: int = 10000,
                 vocab_size: int = 1000,
                 max_length: int = 64,
                 min_length: int = 16,
                 task: str = "mlm",
                 mask_probability: float = 0.15,
                 random_token_probability: float = 0.1,
                 unchanged_probability: float = 0.1,
                 special_tokens: Optional[Dict[str, int]] = None,
                 seed: int = 42):
        """
        Initialize toy dataset.
        
        Args:
            num_samples: Number of samples to generate
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            task: Task type ("mlm", "clm", "classification")
            mask_probability: Probability of masking tokens (MLM)
            random_token_probability: Probability of random token replacement
            unchanged_probability: Probability of keeping token unchanged
            special_tokens: Special token mappings
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.min_length = min_length
        self.task = task
        self.mask_probability = mask_probability
        self.random_token_probability = random_token_probability
        self.unchanged_probability = unchanged_probability
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Special tokens
        self.special_tokens = special_tokens or {
            'pad': 0,
            'cls': 101,
            'sep': 102,
            'mask': 103,
            'unk': 100
        }
        
        # Generate vocabulary
        self.vocab = list(range(4, vocab_size))  # Exclude special tokens
        
        # Generate samples
        self.samples = self._generate_samples()
        
        logger.info(f"Generated {len(self.samples)} samples for {task} task")
        logger.info(f"Vocabulary size: {vocab_size}, Sequence length: {min_length}-{max_length}")
    
    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate synthetic samples"""
        samples = []
        
        for i in range(self.num_samples):
            # Generate random sequence length
            seq_len = random.randint(self.min_length, self.max_length)
            
            # Generate random token sequence
            tokens = [random.choice(self.vocab) for _ in range(seq_len)]
            
            # Add special tokens based on task
            if self.task == "mlm":
                sample = self._create_mlm_sample(tokens)
            elif self.task == "clm":
                sample = self._create_clm_sample(tokens)
            elif self.task == "classification":
                sample = self._create_classification_sample(tokens)
            else:
                raise ValueError(f"Unknown task: {self.task}")
            
            samples.append(sample)
        
        return samples
    
    def _create_mlm_sample(self, tokens: List[int]) -> Dict[str, Any]:
        """Create masked language modeling sample"""
        # Add CLS and SEP tokens
        input_ids = [self.special_tokens['cls']] + tokens + [self.special_tokens['sep']]
        
        # Create targets (same as input for MLM)
        target_ids = input_ids.copy()
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Apply masking
        masked_input_ids, masked_target_ids, mask_indices = self._apply_masking(
            input_ids, target_ids
        )
        
        return {
            'input_ids': torch.tensor(masked_input_ids, dtype=torch.long),
            'target_ids': torch.tensor(masked_target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'mask_indices': torch.tensor(mask_indices, dtype=torch.bool),
            'task': 'mlm'
        }
    
    def _create_clm_sample(self, tokens: List[int]) -> Dict[str, Any]:
        """Create causal language modeling sample"""
        # Add CLS token at beginning
        input_ids = [self.special_tokens['cls']] + tokens
        
        # Targets are shifted input (next token prediction)
        target_ids = input_ids[1:] + [self.special_tokens['sep']]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'task': 'clm'
        }
    
    def _create_classification_sample(self, tokens: List[int]) -> Dict[str, Any]:
        """Create classification sample"""
        # Add CLS and SEP tokens
        input_ids = [self.special_tokens['cls']] + tokens + [self.special_tokens['sep']]
        
        # Generate random label (binary classification)
        label = random.randint(0, 1)
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor([label], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'task': 'classification'
        }
    
    def _apply_masking(self, input_ids: List[int], target_ids: List[int]) -> Tuple[List[int], List[int], List[bool]]:
        """Apply BERT-style masking"""
        masked_input_ids = input_ids.copy()
        masked_target_ids = [-100] * len(target_ids)  # -100 for ignored tokens
        mask_indices = [False] * len(input_ids)
        
        # Don't mask special tokens
        maskable_positions = []
        for i, token in enumerate(input_ids):
            if token not in [self.special_tokens['cls'], self.special_tokens['sep']]:
                maskable_positions.append(i)
        
        if not maskable_positions:
            return masked_input_ids, masked_target_ids, mask_indices
        
        # Select positions to mask
        num_to_mask = max(1, int(len(maskable_positions) * self.mask_probability))
        positions_to_mask = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))
        
        for pos in positions_to_mask:
            mask_indices[pos] = True
            masked_target_ids[pos] = target_ids[pos]  # Set target to original token
            
            # Apply masking strategy
            rand = random.random()
            if rand < self.mask_probability:
                # Replace with MASK token
                masked_input_ids[pos] = self.special_tokens['mask']
            elif rand < self.mask_probability + self.random_token_probability:
                # Replace with random token
                masked_input_ids[pos] = random.choice(self.vocab)
            # else: keep unchanged (unchanged_probability)
        
        return masked_input_ids, masked_target_ids, mask_indices
    
    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index"""
        return self.samples[idx]
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings"""
        return self.special_tokens
    
    def create_dataloader(self, 
                         batch_size: int = 16,
                         shuffle: bool = True,
                         num_workers: int = 0) -> DataLoader:
        """Create DataLoader for this dataset"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching"""
        # Get max length in batch
        max_len = max(len(sample['input_ids']) for sample in batch)
        
        # Pad sequences
        input_ids = []
        target_ids = []
        attention_masks = []
        mask_indices = []
        
        for sample in batch:
            # Pad input_ids
            padded_input = sample['input_ids'].tolist()
            padded_input += [self.special_tokens['pad']] * (max_len - len(padded_input))
            input_ids.append(padded_input)
            
            # Pad target_ids
            padded_target = sample['target_ids'].tolist()
            if sample['task'] == 'classification':
                # For classification, target is a single value
                padded_target = padded_target + [-100] * (max_len - 1)
            else:
                padded_target += [-100] * (max_len - len(padded_target))
            target_ids.append(padded_target)
            
            # Pad attention_mask
            padded_mask = sample['attention_mask'].tolist()
            padded_mask += [0] * (max_len - len(padded_mask))
            attention_masks.append(padded_mask)
            
            # Pad mask_indices (for MLM)
            if 'mask_indices' in sample:
                padded_mask_indices = sample['mask_indices'].tolist()
                padded_mask_indices += [False] * (max_len - len(padded_mask_indices))
                mask_indices.append(padded_mask_indices)
            else:
                mask_indices.append([False] * max_len)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'mask_indices': torch.tensor(mask_indices, dtype=torch.bool),
            'task': batch[0]['task']
        }


def create_toy_datasets(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Extract configuration
    data_config = config.get('data', {})
    toy_config = data_config.get('toy_dataset', {})
    
    # Dataset parameters
    num_samples = data_config.get('num_samples', 10000)
    vocab_size = data_config.get('vocab_size', 1000)
    max_length = data_config.get('max_length', 64)
    task = toy_config.get('task', 'mlm')
    mask_prob = toy_config.get('mask_probability', 0.15)
    
    # Training parameters
    batch_size = config.get('training', {}).get('batch_size', 16)
    num_workers = data_config.get('num_workers', 0)
    
    # Create datasets
    train_dataset = ToyDataset(
        num_samples=int(num_samples * 0.8),
        vocab_size=vocab_size,
        max_length=max_length,
        task=task,
        mask_probability=mask_prob,
        seed=42
    )
    
    val_dataset = ToyDataset(
        num_samples=int(num_samples * 0.1),
        vocab_size=vocab_size,
        max_length=max_length,
        task=task,
        mask_probability=mask_prob,
        seed=43
    )
    
    test_dataset = ToyDataset(
        num_samples=int(num_samples * 0.1),
        vocab_size=vocab_size,
        max_length=max_length,
        task=task,
        mask_probability=mask_prob,
        seed=44
    )
    
    # Create dataloaders
    train_loader = train_dataset.create_dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = val_dataset.create_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = test_dataset.create_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info(f"Created datasets: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the toy dataset
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Create test dataset
    dataset = ToyDataset(num_samples=100, vocab_size=100, max_length=32, task='mlm')
    
    # Test dataloader
    dataloader = dataset.create_dataloader(batch_size=4)
    
    print("Testing Toy Dataset:")
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {dataset.get_vocab_size()}")
    print(f"Special tokens: {dataset.get_special_tokens()}")
    
    # Test a batch
    for batch in dataloader:
        print(f"Batch shape: {batch['input_ids'].shape}")
        print(f"Task: {batch['task']}")
        print(f"Input IDs: {batch['input_ids'][0]}")
        print(f"Target IDs: {batch['target_ids'][0]}")
        print(f"Attention mask: {batch['attention_mask'][0]}")
        print(f"Mask indices: {batch['mask_indices'][0]}")
        break
    
    print("✅ Toy dataset test completed!")
