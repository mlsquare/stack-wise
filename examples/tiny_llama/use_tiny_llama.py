"""
TinyLLaMA Usage Script
Interactive text generation using a trained TinyLLaMA model.
"""

import sys
import os
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from config.base import StackWiseConfig
from model.architecture import Rack

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TinyLLaMAGenerator:
    """Text generator using TinyLLaMA model."""
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initialize generator.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        self.config = self._load_config()
        self.device = self._setup_device()
        self.model = self._load_model()
        
        logger.info("Initialized TinyLLaMA generator")
    
    def _load_config(self) -> StackWiseConfig:
        """Load configuration from YAML file."""
        return StackWiseConfig.from_yaml(self.config_path)
    
    def _setup_device(self) -> torch.device:
        """Setup generation device."""
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
    
    def simple_tokenize(self, text: str) -> List[int]:
        """
        Simple tokenization for demo purposes.
        In a real implementation, you'd use a proper tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        # Simple word-based tokenization with hash
        words = text.lower().split()
        tokens = []
        
        for word in words:
            # Use hash to map words to token IDs within vocab size
            token_id = hash(word) % self.config.model.vocab_size
            tokens.append(token_id)
        
        return tokens
    
    def simple_detokenize(self, tokens: List[int]) -> str:
        """
        Simple detokenization for demo purposes.
        In a real implementation, you'd use a proper tokenizer.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        # For demo purposes, just return token IDs as text
        return f"[Tokens: {', '.join(map(str, tokens))}]"
    
    def generate(self, 
                prompt: str, 
                max_length: int = 50, 
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None) -> str:
        """
        Generate text continuation from a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (keep only top k tokens)
            top_p: Top-p (nucleus) sampling (keep tokens with cumulative prob <= p)
            
        Returns:
            Generated text
        """
        # Tokenize prompt
        prompt_tokens = self.simple_tokenize(prompt)
        
        if not prompt_tokens:
            return "Error: Empty prompt"
        
        # Convert to tensor
        input_tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        generated_tokens = input_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for next token
                logits = self.model(generated_tokens)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit max length
                if generated_tokens.shape[1] >= max_length:
                    break
        
        # Convert back to text
        all_tokens = generated_tokens[0].tolist()
        generated_text = self.simple_detokenize(all_tokens)
        
        return generated_text
    
    def interactive_generation(self):
        """Run interactive text generation session."""
        print("🤖 TinyLLaMA Interactive Text Generator")
        print("=" * 50)
        print("Commands:")
        print("  - Type your prompt and press Enter to generate text")
        print("  - Type 'quit' or 'exit' to stop")
        print("  - Type 'help' for generation options")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n📝 Enter prompt: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\n🔧 Generation Options:")
                    print("  - max_length: Maximum tokens to generate (default: 50)")
                    print("  - temperature: Randomness level 0.1-2.0 (default: 1.0)")
                    print("  - top_k: Keep only top k tokens (default: None)")
                    print("  - top_p: Nucleus sampling threshold (default: None)")
                    print("\nExample: 'Hello world' with max_length=30, temperature=0.8")
                    continue
                
                if not user_input:
                    print("Please enter a prompt.")
                    continue
                
                # Parse generation parameters (simple format)
                parts = user_input.split(' with ')
                prompt = parts[0]
                params = {}
                
                if len(parts) > 1:
                    param_str = parts[1]
                    for param in param_str.split(', '):
                        if '=' in param:
                            key, value = param.split('=')
                            if key == 'max_length':
                                params['max_length'] = int(value)
                            elif key == 'temperature':
                                params['temperature'] = float(value)
                            elif key == 'top_k':
                                params['top_k'] = int(value)
                            elif key == 'top_p':
                                params['top_p'] = float(value)
                
                # Generate text
                print("🔄 Generating...")
                generated = self.generate(prompt, **params)
                
                print(f"\n🤖 Generated text:")
                print(f"   {generated}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Use TinyLLaMA for text generation")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--prompt", help="Single prompt to generate from")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, help="Top-p sampling")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return
    
    try:
        # Initialize generator
        generator = TinyLLaMAGenerator(args.model, args.config)
        
        if args.interactive:
            # Run interactive mode
            generator.interactive_generation()
        else:
            # Single generation
            if not args.prompt:
                print("Please provide a prompt with --prompt or use --interactive mode")
                return
            
            generated = generator.generate(
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            print(f"Prompt: {args.prompt}")
            print(f"Generated: {generated}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise

if __name__ == "__main__":
    main()
