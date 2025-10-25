#!/usr/bin/env python3
"""
Evaluate Tiny BERT Model

This script demonstrates how to evaluate a trained tiny BERT model
using the Stack-Wise framework.
"""

import sys
import os
import torch
import torch.nn as nn
import logging
import argparse
import json
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training import ProgressiveRackBuilder
from src.config.base import StackWiseConfig
from toy_dataset import create_toy_datasets

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TinyBERTEvaluator:
    """Evaluator for Tiny BERT model"""
    
    def __init__(self, config_path: str, model_path: str):
        # Load configuration
        self.config = StackWiseConfig.from_yaml(config_path)
        
        # Create datasets
        self.train_loader, self.val_loader, self.test_loader = create_toy_datasets(self.config.to_dict())
        
        # Create rack builder and load model
        self.rack_builder = ProgressiveRackBuilder(config=self.config, building_mode="append")
        self.rack_builder.load_rack(model_path)
        
        logger.info(f"Initialized TinyBERT evaluator")
        logger.info(f"Test samples: {len(self.test_loader.dataset)}")
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on test set"""
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        rack = self.rack_builder.build_rack()
        
        # Count parameters
        total_params = sum(p.numel() for p in rack.parameters() if p.requires_grad)
        
        # Get rack info
        rack_info = self.rack_builder.get_rack_info()
        
        return {
            'total_parameters': total_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'rack_info': rack_info
        }


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Tiny BERT model')
    parser.add_argument('--config', type=str, default='tiny_bert_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--output-file', type=str, default='evaluation_results.json',
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = TinyBERTEvaluator(args.config, args.model_path)
    
    # Evaluate model
    metrics = evaluator.evaluate()
    
    # Get model info
    model_info = evaluator.get_model_info()
    
    # Combine results
    results = {
        'evaluation_metrics': metrics,
        'model_info': model_info
    }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Model parameters: {model_info['total_parameters']:,}")
    logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test perplexity: {metrics['perplexity']:.4f}")
    logger.info(f"Test loss: {metrics['loss']:.4f}")
    logger.info(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
