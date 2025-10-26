#!/usr/bin/env python3
"""
Quick Start Example for StackWise Baselines

This example demonstrates how to use the baselines module
for training and evaluation with minimal configuration.
"""

import sys
from pathlib import Path

# Add baselines src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import logging
from omegaconf import OmegaConf

from evaluation import UnifiedEvaluator, BenchmarkConfig
from config.base import StackWiseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run a quick start example."""
    
    logger.info("StackWise Baselines Quick Start Example")
    
    # 1. Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "experiments" / "bert_reproduction" / "bert_base_glue.yaml"
    cfg = OmegaConf.load(config_path)
    
    logger.info(f"Loaded configuration: {cfg.experiment.name}")
    
    # 2. Create a simple model for demonstration
    model = create_demo_model()
    
    # 3. Create a dummy tokenizer
    tokenizer = create_demo_tokenizer()
    
    # 4. Set up evaluation
    benchmark_config = BenchmarkConfig(
        name=cfg.benchmark.name,
        description=cfg.benchmark.description,
        version=cfg.benchmark.version,
        tasks=cfg.benchmark.tasks,
        dataset=cfg.benchmark.dataset,
        evaluation=cfg.benchmark.evaluation,
        metrics=cfg.benchmark.metrics,
        baseline=cfg.benchmark.get("baseline"),
        target_scores=cfg.benchmark.get("target_scores"),
    )
    
    # 5. Run evaluation
    evaluator = UnifiedEvaluator(benchmark_config)
    
    logger.info("Running evaluation...")
    results = evaluator.evaluate_model(model, tokenizer, device="cpu")
    
    # 6. Display results
    logger.info("Evaluation Results:")
    for key, result in results.items():
        logger.info(f"  {key}: {result.score:.4f}")
    
    # 7. Generate report
    report = evaluator.generate_report(results)
    logger.info(f"\nGenerated Report:\n{report}")
    
    logger.info("Quick start example completed!")


def create_demo_model():
    """Create a simple demo model for testing."""
    
    class DemoModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(30522, 768)
            self.linear = torch.nn.Linear(768, 2)
            
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            x = x.mean(dim=1)  # Simple pooling
            x = self.linear(x)
            return x
    
    return DemoModel()


def create_demo_tokenizer():
    """Create a simple demo tokenizer for testing."""
    
    class DemoTokenizer:
        def __init__(self):
            self.vocab_size = 30522
            
        def encode(self, text):
            # Simple tokenization
            return [1, 2, 3, 4, 5]  # Dummy tokens
        
        def __call__(self, texts, **kwargs):
            # Return dummy tokenized data
            if isinstance(texts, str):
                texts = [texts]
            
            input_ids = []
            attention_mask = []
            
            for text in texts:
                tokens = self.encode(text)
                input_ids.append(tokens)
                attention_mask.append([1] * len(tokens))
            
            return {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)
            }
    
    return DemoTokenizer()


if __name__ == "__main__":
    main()
