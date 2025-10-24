#!/usr/bin/env python3
"""
Data loader for GPT-2 fusion training.

This script handles data preparation, tokenization, and loading for the GPT-2 fusion training example.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2Dataset(Dataset):
    """Dataset class for GPT-2 training data."""
    
    def __init__(self, texts: List[str], tokenizer: GPT2Tokenizer, max_length: int = 512):
        """
        Initialize GPT-2 dataset.
        
        Args:
            texts: List of text samples
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

def create_dummy_corpus(num_samples: int = 10000, output_path: str = "./data/english_corpus_10k.json"):
    """
    Create a dummy English corpus for training.
    
    Args:
        num_samples: Number of samples to generate
        output_path: Path to save the corpus
    """
    logger.info(f"Creating dummy corpus with {num_samples} samples...")
    
    # Sample English sentences for training
    sample_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models can learn complex patterns from data.",
        "Transformers have revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Large language models can generate human-like text.",
        "Fine-tuning adapts pre-trained models for specific tasks.",
        "Transfer learning leverages knowledge from one domain to another.",
        "Neural networks consist of interconnected processing units.",
        "Backpropagation is used to train neural networks.",
        "Gradient descent optimizes model parameters.",
        "Overfitting occurs when models memorize training data.",
        "Regularization techniques prevent overfitting.",
        "Cross-validation evaluates model performance.",
        "Hyperparameter tuning optimizes model configuration.",
        "Feature engineering creates meaningful input representations.",
        "Data preprocessing cleans and prepares datasets.",
        "Model evaluation measures performance on test data.",
        "Ensemble methods combine multiple models for better performance."
    ]
    
    # Generate corpus by repeating and varying sample sentences
    corpus = []
    for i in range(num_samples):
        # Select a random sentence and add variations
        base_sentence = sample_sentences[i % len(sample_sentences)]
        
        # Add some variation
        variations = [
            f"This is sample {i+1}: {base_sentence}",
            f"Example {i+1} demonstrates: {base_sentence}",
            f"Case study {i+1}: {base_sentence}",
            f"Training example {i+1}: {base_sentence}",
            f"Data point {i+1}: {base_sentence}"
        ]
        
        # Select variation
        text = variations[i % len(variations)]
        corpus.append(text)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save corpus
    with open(output_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    logger.info(f"Created corpus with {len(corpus)} samples at {output_path}")
    return corpus

def load_corpus(data_path: str) -> List[str]:
    """
    Load corpus from file.
    
    Args:
        data_path: Path to corpus file
        
    Returns:
        List of text samples
    """
    logger.info(f"Loading corpus from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Corpus file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        corpus = json.load(f)
    
    logger.info(f"Loaded {len(corpus)} samples")
    return corpus

def create_data_loaders(
    corpus: List[str],
    tokenizer: GPT2Tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    train_split: float = 0.9,
    num_workers: int = 4
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        corpus: List of text samples
        tokenizer: GPT-2 tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        train_split: Fraction of data for training
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating data loaders...")
    
    # Split data
    split_idx = int(len(corpus) * train_split)
    train_texts = corpus[:split_idx]
    val_texts = corpus[split_idx:]
    
    logger.info(f"Train samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    # Create datasets
    train_dataset = GPT2Dataset(train_texts, tokenizer, max_length)
    val_dataset = GPT2Dataset(val_texts, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info("Data loaders created successfully")
    return train_loader, val_loader

def get_tokenizer(tokenizer_path: str = "gpt2") -> GPT2Tokenizer:
    """
    Get GPT-2 tokenizer.
    
    Args:
        tokenizer_path: Path to tokenizer
        
    Returns:
        GPT-2 tokenizer
    """
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
    return tokenizer

def main():
    """Main function for data preparation."""
    parser = argparse.ArgumentParser(description="GPT-2 Data Loader")
    parser.add_argument("--prepare", action="store_true", help="Prepare dummy corpus")
    parser.add_argument("--data_path", default="./data/english_corpus_10k.json", help="Path to corpus file")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--test_loader", action="store_true", help="Test data loader")
    
    args = parser.parse_args()
    
    if args.prepare:
        # Create dummy corpus
        corpus = create_dummy_corpus(args.num_samples, args.data_path)
        logger.info("✅ Corpus preparation completed")
    
    if args.test_loader:
        # Test data loader
        logger.info("Testing data loader...")
        
        # Load corpus
        corpus = load_corpus(args.data_path)
        
        # Get tokenizer
        tokenizer = get_tokenizer()
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            corpus, tokenizer, batch_size=4, max_length=128
        )
        
        # Test loading a batch
        for batch in train_loader:
            logger.info(f"Batch shape: {batch['input_ids'].shape}")
            logger.info(f"Sample tokens: {batch['input_ids'][0][:10]}")
            break
        
        logger.info("✅ Data loader test completed")

if __name__ == "__main__":
    main()
