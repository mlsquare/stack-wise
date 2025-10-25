#!/usr/bin/env python3
"""
Train Tiny BERT Model using Stack-Wise Framework

This script demonstrates how to train a tiny BERT model using the Stack-Wise
progressive training system with a toy dataset.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.base import StackWiseConfig
from src.training import ProgressiveTrainer, ProgressiveRackBuilder, ProgressiveDataLoader
from src.model.architecture import create_stack_from_config, create_rack_from_config
from toy_dataset import create_toy_datasets

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TinyBERTTrainer:
    """Trainer for Tiny BERT model using Stack-Wise framework"""
    
    def __init__(self, config_path: str):
        # Load configuration
        self.config = StackWiseConfig.from_yaml(config_path)
        
        # Create base datasets (for creating ProgressiveDataLoader)
        # Create a simple config dict for toy dataset
        simple_config = {
            'data': {
                'num_samples': 10000,
                'vocab_size': 1000,
                'max_length': 64,
                'toy_dataset': {
                    'task': 'mlm',
                    'mask_probability': 0.15
                }
            },
            'training': {
                'batch_size': 16
            }
        }
        self.train_loader, self.val_loader, self.test_loader = create_toy_datasets(simple_config)
        
        # Create progressive trainer and rack builder
        self.trainer = ProgressiveTrainer(config=self.config)
        self.rack_builder = ProgressiveRackBuilder(config=self.config, building_mode="append")
        
        logger.info(f"Initialized TinyBERT trainer using Stack-Wise framework")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
    
    def train_progressively(self, target_stacks: int = 4) -> Dict[str, Any]:
        """Train model progressively using Stack-Wise framework"""
        logger.info(f"Starting progressive training for {target_stacks} stacks")
        
        # Create ProgressiveDataLoader for progressive training
        progressive_dataloader = ProgressiveDataLoader(
            base_dataloader=self.train_loader,
            masking_strategy=None,  # Will be set by trainer
            stack_idx=0,  # Will be updated during training
            trunk_activations=None,
            cache_activations=self.config.training.progressive.cache_activations
        )
        
        # Train using progressive training
        results = self.trainer.train_rack(
            rack_builder=self.rack_builder,
            dataloader=progressive_dataloader,
            target_stacks=target_stacks
        )
        
        logger.info("Progressive training completed!")
        return results
    
    def train_complete_model(self) -> Dict[str, Any]:
        """Train complete model at once"""
        logger.info("Starting complete model training")
        
        # Create ProgressiveDataLoader for complete training
        progressive_dataloader = ProgressiveDataLoader(
            base_dataloader=self.train_loader,
            masking_strategy=None,  # Will be set by trainer
            stack_idx=0,  # Will be updated during training
            trunk_activations=None,
            cache_activations=self.config.training.progressive.cache_activations
        )
        
        # Train complete rack
        results = self.trainer.train_rack(
            rack_builder=self.rack_builder,
            dataloader=progressive_dataloader,
            target_stacks=None  # Train complete model
        )
        
        logger.info("Complete model training completed!")
        return results
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        
        # Get the built rack
        rack = self.rack_builder.build_rack()
        rack.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Move to device
                input_ids = batch['input_ids']
                target_ids = batch['target_ids']
                attention_mask = batch['attention_mask']
                
                # Forward pass
                outputs = rack(input_ids)
                logits = outputs['logits']
                
                # Compute loss
                if batch['task'] == 'mlm':
                    if 'mask_indices' in batch:
                        mask_indices = batch['mask_indices']
                        loss = self._compute_mlm_loss(logits, target_ids, mask_indices)
                    else:
                        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                elif batch['task'] == 'clm':
                    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                elif batch['task'] == 'classification':
                    pooled_output = outputs['pooled_output']
                    loss = nn.CrossEntropyLoss()(pooled_output, target_ids.squeeze())
                else:
                    raise ValueError(f"Unknown task: {batch['task']}")
                
                total_loss += loss.item()
                
                # Compute accuracy
                if batch['task'] in ['mlm', 'clm']:
                    predictions = torch.argmax(logits, dim=-1)
                    correct = (predictions == target_ids).sum().item()
                    total_correct += correct
                    total_tokens += target_ids.numel()
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item()
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def _compute_mlm_loss(self, logits: torch.Tensor, targets: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        """Compute masked language modeling loss"""
        # Flatten tensors
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        mask_flat = mask_indices.view(-1)
        
        # Only compute loss on masked positions
        masked_logits = logits_flat[mask_flat]
        masked_targets = targets_flat[mask_flat]
        
        if len(masked_targets) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        loss = nn.CrossEntropyLoss()(masked_logits, masked_targets)
        return loss
    
    def save_model(self, output_dir: str):
        """Save model and training results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save rack
        rack_path = os.path.join(output_dir, 'tiny_bert_rack.pt')
        self.rack_builder.save_rack(rack_path)
        
        # Save training results
        results_path = os.path.join(output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.trainer.get_training_info(), f, indent=2)
        
        # Save configuration
        config_path = os.path.join(output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Saved model and results to {output_dir}")
    
    def load_model(self, model_path: str):
        """Load model from checkpoint"""
        success = self.rack_builder.load_rack(model_path)
        if success:
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.error(f"Failed to load model from {model_path}")
        return success


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Tiny BERT model using Stack-Wise framework')
    parser.add_argument('--config', type=str, default='tiny_bert_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./tiny_bert_outputs',
                       help='Directory to save outputs')
    parser.add_argument('--mode', type=str, choices=['progressive', 'complete'], default='progressive',
                       help='Training mode: progressive or complete')
    parser.add_argument('--target-stacks', type=int, default=4,
                       help='Number of stacks for progressive training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create trainer
    trainer = TinyBERTTrainer(args.config)
    
    # Train model
    if args.mode == 'progressive':
        logger.info(f"Training progressively with {args.target_stacks} stacks")
        results = trainer.train_progressively(args.target_stacks)
    else:
        logger.info("Training complete model")
        results = trainer.train_complete_model()
    
    # Evaluate model
    eval_metrics = trainer.evaluate_model()
    
    # Save model
    trainer.save_model(args.output_dir)
    
    # Print results
    logger.info("=" * 50)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 50)
    logger.info(f"Training mode: {args.mode}")
    if args.mode == 'progressive':
        logger.info(f"Target stacks: {args.target_stacks}")
    logger.info(f"Test accuracy: {eval_metrics['accuracy']:.4f}")
    logger.info(f"Test perplexity: {eval_metrics['perplexity']:.4f}")
    logger.info(f"Test loss: {eval_metrics['loss']:.4f}")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()