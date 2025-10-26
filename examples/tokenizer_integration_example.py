"""
Example of integrating tokenizer with StackWise configuration.
Shows how to set vocabulary size from actual tokenizer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import StackWiseConfig


def example_tokenizer_integration():
    """Example of how to integrate tokenizer with configuration."""
    
    # Load configuration
    config = StackWiseConfig.from_yaml("config.yaml")
    
    # Simulate loading a tokenizer (replace with actual tokenizer loading)
    # Example with different tokenizers:
    
    # 1. HuggingFace tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = len(tokenizer)
        print(f"✅ Loaded HuggingFace tokenizer with vocab size: {vocab_size}")
    except ImportError:
        print("⚠️  transformers not available, using dummy tokenizer")
        # Dummy tokenizer for demonstration
        class DummyTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
            def __len__(self):
                return self.vocab_size
        
        tokenizer = DummyTokenizer(50257)  # GPT-2 vocab size
        vocab_size = len(tokenizer)
        print(f"✅ Using dummy tokenizer with vocab size: {vocab_size}")
    
    # 2. Set vocabulary size in configuration
    config.set_vocab_size(vocab_size)
    
    # 3. Validate configuration
    config.validate()
    
    print(f"✅ Configuration updated with vocab size: {config.model.vocab_size}")
    print(f"✅ Model ready with {config.model.d_model}D, {config.model.architecture.n_stacks * config.model.architecture.blocks_per_stack} blocks")
    
    return config, tokenizer


def example_different_tokenizers():
    """Example with different tokenizer sizes."""
    
    # Common tokenizer vocabulary sizes
    tokenizer_sizes = {
        "gpt2": 50257,
        "gpt2-medium": 50257,
        "gpt2-large": 50257,
        "gpt2-xl": 50257,
        "bert-base-uncased": 30522,
        "roberta-base": 50265,
        "t5-base": 32128,
        "custom": 128000
    }
    
    config = StackWiseConfig.from_yaml("config.yaml")
    
    for tokenizer_name, vocab_size in tokenizer_sizes.items():
        print(f"\n--- Testing with {tokenizer_name} ---")
        config.set_vocab_size(vocab_size)
        config.validate()
        print(f"✅ {tokenizer_name}: vocab_size={config.model.vocab_size}")


if __name__ == "__main__":
    print("StackWise Tokenizer Integration Examples")
    print("=" * 50)
    
    print("\n1. Basic Tokenizer Integration:")
    config, tokenizer = example_tokenizer_integration()
    
    print("\n2. Different Tokenizer Sizes:")
    example_different_tokenizers()
    
    print("\n✅ Tokenizer integration working!")
