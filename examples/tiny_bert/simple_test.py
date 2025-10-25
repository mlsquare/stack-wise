#!/usr/bin/env python3
"""
Simple test for tiny BERT example

This script tests the basic functionality without progressive training.
"""

import sys
import os
import torch
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.base import StackWiseConfig
from toy_dataset import create_toy_datasets

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_toy_dataset():
    """Test the toy dataset creation"""
    logger.info("Testing toy dataset...")
    
    # Create a simple config for toy dataset
    simple_config = {
        'data': {
            'num_samples': 100,
            'vocab_size': 100,
            'max_length': 32,
            'toy_dataset': {
                'task': 'mlm',
                'mask_probability': 0.15
            }
        },
        'training': {
            'batch_size': 4
        }
    }
    
    # Create datasets
    train_loader, val_loader, test_loader = create_toy_datasets(simple_config)
    
    logger.info(f"Created datasets: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")
    
    # Test a batch
    for batch in train_loader:
        logger.info(f"Batch shape: {batch['input_ids'].shape}")
        logger.info(f"Task: {batch['task']}")
        logger.info(f"Input IDs sample: {batch['input_ids'][0][:10]}")
        logger.info(f"Target IDs sample: {batch['target_ids'][0][:10]}")
        break
    
    logger.info("✅ Toy dataset test completed!")


def test_config_loading():
    """Test configuration loading"""
    logger.info("Testing configuration loading...")
    
    try:
        config = StackWiseConfig.from_yaml('tiny_bert_config.yaml')
        logger.info("✅ Configuration loaded successfully!")
        logger.info(f"Model vocab size: {config.model.vocab_size}")
        logger.info(f"Model d_model: {config.model.d_model}")
        logger.info(f"Training lr: {config.training.lr}")
        logger.info(f"Training batch_size: {config.training.batch_size}")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return False


def main():
    """Main test function"""
    logger.info("Starting tiny BERT simple tests...")
    
    # Test configuration loading
    config_ok = test_config_loading()
    
    # Test toy dataset
    test_toy_dataset()
    
    if config_ok:
        logger.info("✅ All basic tests passed!")
    else:
        logger.info("❌ Some tests failed!")


if __name__ == "__main__":
    main()
