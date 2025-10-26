"""
TinyLLaMA Training Script
Demonstrates progressive training of a small LLaMA-style model with CLM objective.
"""

import sys
import os
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from config.base import StackWiseConfig
from training.progressive_trainer import ProgressiveTrainer
from training.progressive_rack_builder import ProgressiveRackBuilder
from training.progressive_dataloader import ProgressiveDataLoader
from toy_dataset import create_clm_datasets

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TinyLLaMATrainer:
    """Trainer for TinyLLaMA model with progressive training."""
    
    def __init__(self, config_path: str):
        """
        Initialize TinyLLaMA trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.device = self._setup_device()
        
        # Initialize components
        self.rack_builder = ProgressiveRackBuilder(self.config)
        self.trainer = ProgressiveTrainer(self.config)
        
        logger.info("Initialized TinyLLaMA trainer using Stack-Wise framework")
    
    def _load_config(self) -> StackWiseConfig:
        """Load configuration from YAML file."""
        return StackWiseConfig.from_yaml(self.config_path)
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.training.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.training.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _create_dataloaders(self) -> tuple:
        """Create train, validation, and test dataloaders."""
        return create_clm_datasets(
            vocab_size=self.config.model.vocab_size,
            max_length=self.config.data.max_length,
            train_samples=self.config.data.num_samples,
            val_samples=1500,
            test_samples=1500,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            shuffle=self.config.data.shuffle
        )
    
    def train(self) -> Dict[str, Any]:
        """Train the TinyLLaMA model progressively."""
        logger.info(f"Training samples: {self.config.data.num_samples}")
        logger.info(f"Training progressively with {self.config.training.progressive.target_stacks} stacks")
        
        # Create dataloaders
        train_dataloader, val_dataloader, test_dataloader = self._create_dataloaders()
        
        # Start progressive training
        logger.info("Starting progressive training for CLM task")
        results = self.trainer.train_rack(
            rack_builder=self.rack_builder,
            dataloader=train_dataloader,
            target_stacks=self.config.training.progressive.target_stacks
        )
        
        logger.info("Progressive training completed!")
        return results
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the trained model."""
        logger.info("Evaluating model...")
        
        # Build the complete rack
        rack = self.rack_builder.build_rack()
        rack.eval()
        
        # Create test dataloader
        _, _, test_dataloader = self._create_dataloaders()
        
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits = rack(input_ids)
                
                # Compute CLM loss (next token prediction)
                # Shift logits and targets for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = target_ids[..., 1:].contiguous()
                shift_attention = attention_mask[..., 1:].contiguous()
                
                # Flatten for loss computation
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_targets = shift_targets.view(-1)
                flat_attention = shift_attention.view(-1)
                
                # Compute loss only on non-padded tokens
                loss = nn.CrossEntropyLoss(reduction='none')(flat_logits, flat_targets)
                masked_loss = loss * flat_attention.float()
                
                # Accumulate loss and token count
                total_loss += masked_loss.sum().item()
                total_tokens += flat_attention.sum().item()
                
                # Compute accuracy
                predictions = torch.argmax(flat_logits, dim=-1)
                correct = (predictions == flat_targets) & flat_attention
                correct_predictions += correct.sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, output_dir: str = "./tiny_llama_outputs"):
        """Save the trained model and results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save the complete rack
        rack = self.rack_builder.build_rack()
        rack_path = output_path / "tiny_llama_rack.pt"
        torch.save(rack.state_dict(), rack_path)
        
        # Save configuration
        config_path = output_path / "config.yaml"
        self.config.to_yaml(config_path)
        
        logger.info(f"Saved model and results to {output_path}")
        return str(output_path)

def main():
    """Main training function."""
    # Configuration
    config_path = "tiny_llama_config.yaml"
    
    # Check if config exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Please run this script from the tiny_llama directory")
        return
    
    try:
        # Initialize trainer
        trainer = TinyLLaMATrainer(config_path)
        
        # Train model
        training_results = trainer.train()
        
        # Evaluate model
        eval_metrics = trainer.evaluate_model()
        
        # Save model
        output_dir = trainer.save_model()
        
        # Print results
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        print(f"Training mode: progressive")
        print(f"Target stacks: {trainer.config.training.progressive.target_stacks}")
        print(f"Test accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"Test perplexity: {eval_metrics['perplexity']:.4f}")
        print(f"Test loss: {eval_metrics['loss']:.4f}")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
