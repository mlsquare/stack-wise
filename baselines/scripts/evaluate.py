#!/usr/bin/env python3
"""
StackWise Baselines Evaluation Script

This script provides evaluation capabilities for trained models
on various benchmark tasks.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from evaluation import UnifiedEvaluator, BenchmarkConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function with Hydra configuration."""
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format
    )
    
    logger.info("Starting StackWise Baselines Evaluation")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    try:
        # Load model and tokenizer
        model, tokenizer = _load_model_and_tokenizer(cfg)
        
        # Create benchmark configuration
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
        
        # Initialize evaluator
        evaluator = UnifiedEvaluator(benchmark_config)
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluator.evaluate_model(model, tokenizer, cfg.training.device)
        
        # Save results
        output_dir = Path(cfg.hydra.run.dir)
        evaluator.save_results(results, output_dir)
        
        # Generate and save report
        report = evaluator.generate_report(results)
        report_file = output_dir / "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def _load_model_and_tokenizer(cfg: DictConfig):
    """Load model and tokenizer from checkpoint."""
    
    model_path = cfg.get("model_path")
    if model_path is None:
        raise ValueError("model_path must be specified for evaluation")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Load model
    if model_path.suffix == ".pt":
        # PyTorch checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        else:
            model_state_dict = checkpoint
        
        # Create model architecture
        from model.architecture import Rack
        model = Rack(
            vocab_size=cfg.model.vocab_size,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_kv_heads=cfg.model.n_kv_heads,
            d_ff=cfg.model.d_ff,
            n_stacks=cfg.model.architecture.n_stacks,
            blocks_per_stack=cfg.model.architecture.blocks_per_stack,
            attention_preset=cfg.model.attention_preset,
            dropout=cfg.model.get("dropout", 0.1),
            tie_embeddings=cfg.model.get("tie_embeddings", True),
            freeze_up_proj=cfg.model.get("freeze_up_proj", False),
            use_rope=cfg.model.get("use_rope", False),
            rope_theta=cfg.model.get("rope_theta", 10000.0),
        )
        
        # Load state dict
        model.load_state_dict(model_state_dict)
        
    else:
        # HuggingFace model
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path)
    
    # Load tokenizer
    tokenizer_name = cfg.model.tokenizer_embedding.get("model_name", "bert-base-uncased")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer


if __name__ == "__main__":
    main()
