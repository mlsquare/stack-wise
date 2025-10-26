"""
TinyLLaMA Evaluation Script
Evaluates a trained TinyLLaMA model on CLM tasks.
"""

import sys
import os
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from config.base import StackWiseConfig
from model.architecture import Rack
from toy_dataset import create_clm_datasets

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TinyLLaMAEvaluator:
    """Evaluator for TinyLLaMA model."""
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        self.config = self._load_config()
        self.device = self._setup_device()
        self.model = self._load_model()
        
        logger.info("Initialized TinyLLaMA evaluator")
    
    def _load_config(self) -> StackWiseConfig:
        """Load configuration from YAML file."""
        return StackWiseConfig.from_yaml(self.config_path)
    
    def _setup_device(self) -> torch.device:
        """Setup evaluation device."""
        if self.config.training.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.training.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self) -> Rack:
        """Load the trained model."""
        # Build the rack architecture
        from training.progressive_rack_builder import ProgressiveRackBuilder
        rack_builder = ProgressiveRackBuilder(self.config)
        
        # Build all stacks first (to match the saved model)
        for stack_idx in range(self.config.training.progressive.target_stacks):
            rack_builder.append_stack()
        
        # Build the complete rack
        rack = rack_builder.build_rack()
        
        # Load trained weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        rack.load_state_dict(state_dict)
        rack.eval()
        
        logger.info("Loaded trained model")
        return rack
    
    def evaluate_perplexity(self, dataloader) -> float:
        """Evaluate perplexity on a dataset."""
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Compute CLM loss
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
                
                total_loss += masked_loss.sum().item()
                total_tokens += flat_attention.sum().item()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def evaluate_accuracy(self, dataloader) -> float:
        """Evaluate next-token prediction accuracy."""
        correct_predictions = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Get predictions for next tokens
                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = target_ids[..., 1:].contiguous()
                shift_attention = attention_mask[..., 1:].contiguous()
                
                # Get predictions
                predictions = torch.argmax(shift_logits, dim=-1)
                
                # Compute accuracy
                correct = (predictions == shift_targets) & shift_attention
                correct_predictions += correct.sum().item()
                total_tokens += shift_attention.sum().item()
        
        accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
        return accuracy
    
    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 1.0) -> str:
        """
        Generate text continuation from a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        # Convert prompt to token IDs (simplified - using random tokens for demo)
        # In a real implementation, you'd use a proper tokenizer
        prompt_tokens = torch.randint(0, self.config.model.vocab_size, (1, len(prompt.split())))
        
        generated_tokens = prompt_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for next token
                logits = self.model(generated_tokens)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit max length
                if generated_tokens.shape[1] >= max_length:
                    break
        
        # Convert back to text (simplified - using token IDs as text for demo)
        generated_text = f"Generated tokens: {generated_tokens[0].tolist()}"
        return generated_text
    
    def comprehensive_evaluation(self) -> Dict[str, float]:
        """Run comprehensive evaluation on test set."""
        logger.info("Running comprehensive evaluation...")
        
        # Create test dataloader
        _, _, test_dataloader = create_clm_datasets(
            vocab_size=self.config.model.vocab_size,
            max_length=self.config.data.max_length,
            train_samples=100,  # Small sample for evaluation
            val_samples=100,
            test_samples=1000,
            batch_size=self.config.training.batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=False
        )
        
        # Evaluate perplexity
        perplexity = self.evaluate_perplexity(test_dataloader)
        
        # Evaluate accuracy
        accuracy = self.evaluate_accuracy(test_dataloader)
        
        # Test text generation
        generated_text = self.generate_text("The quick brown fox", max_length=20)
        
        results = {
            'perplexity': perplexity,
            'accuracy': accuracy,
            'generated_text': generated_text
        }
        
        logger.info(f"Evaluation results: {results}")
        return results

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate TinyLLaMA model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--prompt", default="The quick brown fox", help="Text prompt for generation")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return
    
    try:
        # Initialize evaluator
        evaluator = TinyLLaMAEvaluator(args.model, args.config)
        
        # Run comprehensive evaluation
        results = evaluator.comprehensive_evaluation()
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Perplexity: {results['perplexity']:.4f}")
        print(f"Next-token Accuracy: {results['accuracy']:.4f}")
        print(f"Generated text: {results['generated_text']}")
        
        # Test custom prompt
        custom_generation = evaluator.generate_text(
            args.prompt, 
            max_length=args.max_length, 
            temperature=args.temperature
        )
        print(f"\nCustom generation (prompt: '{args.prompt}'):")
        print(f"{custom_generation}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
