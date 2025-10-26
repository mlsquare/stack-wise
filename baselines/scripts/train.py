#!/usr/bin/env python3
"""
StackWise Baselines Training Script with Hydra Integration

This script provides a unified training interface for all baseline models
with comprehensive configuration management and experimental tracking.
"""

import logging
import os
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
from training.progressive_trainer import ProgressiveTrainer
from training.trainer import StackWiseTrainer
from config.base import StackWiseConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format
    )
    
    # Set random seeds for reproducibility
    if cfg.get("seed") is not None:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)
    
    # Set deterministic behavior
    if cfg.get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info("Starting StackWise Baselines Training")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    try:
        # Convert Hydra config to StackWise config
        stackwise_config = _convert_hydra_to_stackwise(cfg)
        
        # Initialize trainer based on training strategy
        if cfg.training.strategy == "progressive":
            trainer = ProgressiveTrainer(stackwise_config)
        else:
            trainer = StackWiseTrainer(stackwise_config)
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Evaluate model if benchmark is configured
        if cfg.get("benchmark") is not None:
            logger.info("Starting evaluation...")
            _run_evaluation(trainer.model, trainer.tokenizer, cfg)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def _convert_hydra_to_stackwise(cfg: DictConfig) -> StackWiseConfig:
    """Convert Hydra configuration to StackWise configuration."""
    
    # Extract model configuration
    model_config = {
        "vocab_size": cfg.model.get("vocab_size"),
        "d_model": cfg.model.d_model,
        "n_heads": cfg.model.n_heads,
        "n_kv_heads": cfg.model.n_kv_heads,
        "d_ff": cfg.model.d_ff,
        "architecture": {
            "n_stacks": cfg.model.architecture.n_stacks,
            "blocks_per_stack": cfg.model.architecture.blocks_per_stack,
        },
        "attention_preset": cfg.model.attention_preset,
        "attention_mode": cfg.model.get("attention_mode", "bidirectional"),
        "dropout": cfg.model.get("dropout", 0.1),
        "tie_embeddings": cfg.model.get("tie_embeddings", True),
        "freeze_up_proj": cfg.model.get("freeze_up_proj", False),
        "use_rope": cfg.model.get("use_rope", False),
        "rope_theta": cfg.model.get("rope_theta", 10000.0),
        "mask_fraction_min": cfg.model.get("mask_fraction_min", 0.15),
        "mask_fraction_max": cfg.model.get("mask_fraction_max", 0.90),
        "special_mask_id": cfg.model.get("special_mask_id"),
        "tokenizer_embedding": cfg.model.tokenizer_embedding,
    }
    
    # Extract training configuration
    training_config = {
        "batch_size": cfg.training.batch_size,
        "seq_len": cfg.training.get("seq_len", 512),
        "max_steps": cfg.training.max_steps,
        "optimizer": {
            "optimizer_type": cfg.training.optimizer.optimizer_type,
            "lr": cfg.training.optimizer.lr,
            "weight_decay": cfg.training.optimizer.weight_decay,
            "betas": cfg.training.optimizer.betas,
            "eps": cfg.training.optimizer.eps,
        },
        "device": cfg.training.device,
        "gradient_checkpointing": cfg.training.get("gradient_checkpointing", False),
        "strategy": cfg.training.strategy,
        "end_to_end_scope": cfg.training.get("end_to_end_scope", "rackwise"),
        "progressive": {
            "enabled": cfg.training.progressive.enabled,
            "max_stacks": cfg.training.progressive.max_stacks,
            "target_stacks": cfg.training.progressive.target_stacks,
            "building_mode": cfg.training.progressive.building_mode,
            "trunk_strategy": cfg.training.progressive.trunk_strategy,
            "new_stack_precision": cfg.training.progressive.new_stack_precision,
            "cache_activations": cfg.training.progressive.cache_activations,
            "time_interpretation": cfg.training.progressive.time_interpretation,
            "training_objective": cfg.training.progressive.training_objective,
        },
        "qlora": {
            "enabled": cfg.training.qlora.enabled,
            "rank": cfg.training.qlora.rank,
            "alpha": cfg.training.qlora.alpha,
            "dropout": cfg.training.qlora.dropout,
            "lr": cfg.training.qlora.lr,
            "strategy": cfg.training.qlora.strategy,
            "mixed_precision": cfg.training.qlora.mixed_precision,
        },
        "log_interval": cfg.training.get("log_interval", 100),
        "save_interval": cfg.training.get("save_interval", 1000),
        "checkpoint_dir": cfg.training.get("checkpoint_dir", "./checkpoints"),
        "use_wandb": cfg.training.get("use_wandb", False),
        "wandb_project": cfg.training.get("wandb_project", "stackwise-baselines"),
        "wandb_entity": cfg.training.get("wandb_entity"),
        "wandb_run_name": cfg.training.get("wandb_run_name"),
        "wandb_tags": cfg.training.get("wandb_tags", []),
        "wandb_notes": cfg.training.get("wandb_notes"),
    }
    
    # Extract data configuration
    data_config = {
        "dataset_path": cfg.data.get("dataset_path"),
        "use_dummy_data": cfg.data.get("use_dummy_data", True),
        "num_samples": cfg.data.get("num_samples", 128),
        "tokenizer_path": cfg.data.get("tokenizer_path"),
        "max_length": cfg.data.get("max_length", 512),
        "padding": cfg.data.get("padding", "right"),
        "num_workers": cfg.data.get("num_workers", 0),
        "pin_memory": cfg.data.get("pin_memory", True),
        "shuffle": cfg.data.get("shuffle", True),
    }
    
    # Create StackWise configuration
    stackwise_config = StackWiseConfig(
        model=model_config,
        training=training_config,
        data=data_config
    )
    
    return stackwise_config


def _run_evaluation(model: torch.nn.Module, tokenizer: Any, cfg: DictConfig) -> None:
    """Run evaluation on the trained model."""
    
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


if __name__ == "__main__":
    main()
