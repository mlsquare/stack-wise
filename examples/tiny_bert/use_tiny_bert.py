#!/usr/bin/env python3
"""
Use Tiny BERT Model

This script demonstrates how to load and use a trained tiny BERT model
for inference and further operations.
"""

import sys
import os
import torch
import logging
import argparse
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.base import StackWiseConfig
from src.training import ProgressiveRackBuilder
from toy_dataset import ToyDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TinyBERTModel:
    """Wrapper for using trained Tiny BERT model"""
    
    def __init__(self, config_path: str, model_path: str):
        # Load configuration
        self.config = StackWiseConfig.from_yaml(config_path)
        
        # Create rack builder and load model
        self.rack_builder = ProgressiveRackBuilder(config=self.config, building_mode="append")
        self.rack_builder.load_rack(model_path)
        
        # Get the trained model
        self.model = self.rack_builder.build_rack()
        self.model.eval()
        
        logger.info(f"Loaded Tiny BERT model from {model_path}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def predict(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Make predictions with the model"""
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        return outputs
    
    def generate_text(self, input_ids: torch.Tensor, max_length: int = 50, temperature: float = 1.0) -> torch.Tensor:
        """Generate text using the model"""
        self.model.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model outputs
                outputs = self.model(generated)
                logits = outputs['logits']
                
                # Get next token probabilities
                next_token_logits = logits[:, -1, :] / temperature
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(next_token_probs, 1)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit a separator token (102)
                if next_token.item() == 102:
                    break
        
        return generated
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for input tokens"""
        with torch.no_grad():
            embeddings = self.model.get_embeddings()(input_ids)
        return embeddings
    
    def get_attention_weights(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Get attention weights from all layers"""
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            attention_weights = outputs['attention_probs']
        return attention_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'num_stacks': len(self.rack_builder.stacks),
            'config': self.config.to_dict()
        }
    
    def save_model(self, path: str):
        """Save the model to a file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'rack_builder_state': {
                'stacks': [stack.state_dict() for stack in self.rack_builder.stacks],
                'current_stacks': self.rack_builder.current_stacks,
                'precision_settings': self.rack_builder.precision_settings,
                'qlora_adapters': self.rack_builder.qlora_adapters
            }
        }, path)
        logger.info(f"Saved model to {path}")
    
    @classmethod
    def from_saved_model(cls, model_path: str, config_path: Optional[str] = None):
        """Load model from a saved checkpoint"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if config_path is None:
            # Try to load config from the same directory
            config_path = os.path.join(os.path.dirname(model_path), 'config.yaml')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Create instance
        instance = cls.__new__(cls)
        instance.config = StackWiseConfig.from_yaml(config_path)
        
        # Load rack builder
        instance.rack_builder = ProgressiveRackBuilder(config=instance.config, building_mode="append")
        
        # Restore rack builder state
        if 'rack_builder_state' in checkpoint:
            rack_state = checkpoint['rack_builder_state']
            instance.rack_builder.current_stacks = rack_state['current_stacks']
            instance.rack_builder.precision_settings = rack_state['precision_settings']
            instance.rack_builder.qlora_adapters = rack_state['qlora_adapters']
            
            # Load stack states
            for i, stack_state in enumerate(rack_state['stacks']):
                if i < len(instance.rack_builder.stacks):
                    instance.rack_builder.stacks[i].load_state_dict(stack_state)
        
        # Get the model
        instance.model = instance.rack_builder.build_rack()
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.eval()
        
        logger.info(f"Loaded Tiny BERT model from {model_path}")
        return instance


def demo_model_usage():
    """Demonstrate how to use the Tiny BERT model"""
    logger.info("Tiny BERT Model Usage Demo")
    logger.info("=" * 40)
    
    # Create a simple test dataset
    test_dataset = ToyDataset(num_samples=10, vocab_size=100, max_length=20, task='mlm')
    test_loader = test_dataset.create_dataloader(batch_size=2, shuffle=False)
    
    # Get a sample batch
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        logger.info(f"Input shape: {input_ids.shape}")
        logger.info(f"Sample input: {input_ids[0][:10]}")
        break
    
    logger.info("✅ Demo completed!")


def main():
    """Main function for using Tiny BERT model"""
    parser = argparse.ArgumentParser(description='Use Tiny BERT model')
    parser.add_argument('--config', type=str, default='tiny_bert_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo without loading model')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_model_usage()
        return
    
    try:
        # Load model
        model = TinyBERTModel(args.config, args.model_path)
        
        # Get model info
        info = model.get_model_info()
        logger.info("Model Information:")
        logger.info(f"  Total parameters: {info['total_parameters']:,}")
        logger.info(f"  Trainable parameters: {info['trainable_parameters']:,}")
        logger.info(f"  Model size: {info['model_size_mb']:.2f} MB")
        logger.info(f"  Number of stacks: {info['num_stacks']}")
        
        # Create test data
        test_dataset = ToyDataset(num_samples=5, vocab_size=100, max_length=20, task='mlm')
        test_loader = test_dataset.create_dataloader(batch_size=2, shuffle=False)
        
        # Test inference
        logger.info("\nTesting inference...")
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # Get predictions
            outputs = model.predict(input_ids, attention_mask)
            logits = outputs['logits']
            
            logger.info(f"Input shape: {input_ids.shape}")
            logger.info(f"Output logits shape: {logits.shape}")
            logger.info(f"Sample predictions: {torch.argmax(logits, dim=-1)[0][:10]}")
            break
        
        logger.info("✅ Model usage completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error using model: {e}")


if __name__ == "__main__":
    main()
