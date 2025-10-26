"""
Weights & Biases (wandb) integration for Stack-Wise training monitoring.

This module provides wandb logging capabilities for tracking training metrics,
model parameters, and experiment management.
"""

import os
import logging
from typing import Dict, Any, Optional, List
import torch

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


class WandBLogger:
    """
    Weights & Biases logger for Stack-Wise training.
    
    Provides comprehensive logging of:
    - Training metrics (loss, accuracy, perplexity)
    - Model hyperparameters
    - Progressive training progress
    - System metrics
    - Model artifacts
    """
    
    def __init__(self, config, run_name: Optional[str] = None):
        """
        Initialize wandb logger.
        
        Args:
            config: Training configuration object
            run_name: Optional custom run name
        """
        self.config = config
        self.run_name = run_name
        self.initialized = False
        
        if not WANDB_AVAILABLE:
            logger.warning("wandb not available. Logging will be disabled.")
            return
        
        # Handle both StackWiseConfig and TrainingConfig
        if hasattr(config, 'training'):
            training_config = config.training
        else:
            training_config = config
        
        # Handle both dict and object configs
        if isinstance(training_config, dict):
            use_wandb = training_config.get('use_wandb', False)
        else:
            use_wandb = getattr(training_config, 'use_wandb', False)
        
        if not use_wandb:
            logger.info("wandb logging disabled in configuration")
            return
        
        self._initialize_wandb()
    
    def _initialize_wandb(self):
        """Initialize wandb run."""
        try:
            # Get wandb API key from environment
            api_key = os.getenv('WANDB_API_KEY')
            if not api_key:
                logger.warning("WANDB_API_KEY not found in environment. Using offline mode.")
                os.environ['WANDB_MODE'] = 'offline'
            
            # Prepare wandb configuration
            wandb_config = self._prepare_wandb_config()
            
            # Get training config for wandb settings
            if hasattr(self.config, 'training'):
                training_config = self.config.training
            else:
                training_config = self.config
            
            # Handle both dict and object configs
            if isinstance(training_config, dict):
                project = training_config.get('wandb_project', 'stack-wise')
                entity = training_config.get('wandb_entity', None)
                run_name = self.run_name or training_config.get('wandb_run_name', None)
                tags = training_config.get('wandb_tags', [])
                notes = training_config.get('wandb_notes', None)
            else:
                project = getattr(training_config, 'wandb_project', 'stack-wise')
                entity = getattr(training_config, 'wandb_entity', None)
                run_name = self.run_name or getattr(training_config, 'wandb_run_name', None)
                tags = getattr(training_config, 'wandb_tags', [])
                notes = getattr(training_config, 'wandb_notes', None)
            
            # Initialize wandb run
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=wandb_config,
                tags=tags,
                notes=notes,
                reinit=True
            )
            
            self.initialized = True
            logger.info(f"Initialized wandb run: {wandb.run.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            self.initialized = False
    
    def _prepare_wandb_config(self) -> Dict[str, Any]:
        """Prepare configuration for wandb logging."""
        config_dict = {}
        
        # Model configuration
        if hasattr(self.config, 'model'):
            model_config = self.config.model
            config_dict.update({
                'model/vocab_size': getattr(model_config, 'vocab_size', None),
                'model/d_model': getattr(model_config, 'd_model', None),
                'model/n_heads': getattr(model_config, 'n_heads', None),
                'model/n_kv_heads': getattr(model_config, 'n_kv_heads', None),
                'model/d_ff': getattr(model_config, 'd_ff', None),
                'model/attention_preset': getattr(model_config, 'attention_preset', None),
                'model/use_rope': getattr(model_config, 'use_rope', None),
                'model/rope_theta': getattr(model_config, 'rope_theta', None),
                'model/dropout': getattr(model_config, 'dropout', None),
            })
            
            # Architecture configuration
            if hasattr(model_config, 'architecture'):
                arch_config = model_config.architecture
                config_dict.update({
                    'model/architecture/n_stacks': getattr(arch_config, 'n_stacks', None),
                    'model/architecture/blocks_per_stack': getattr(arch_config, 'blocks_per_stack', None),
                })
        
        # Training configuration
        if hasattr(self.config, 'training'):
            training_config = self.config.training
        else:
            training_config = self.config
        
        # Handle both dict and object configs
        if isinstance(training_config, dict):
            config_dict.update({
                'training/batch_size': training_config.get('batch_size', None),
                'training/max_steps': training_config.get('max_steps', None),
                'training/strategy': training_config.get('strategy', None),
                'training/device': training_config.get('device', None),
                'training/gradient_checkpointing': training_config.get('gradient_checkpointing', None),
            })
        else:
            config_dict.update({
                'training/batch_size': getattr(training_config, 'batch_size', None),
                'training/max_steps': getattr(training_config, 'max_steps', None),
                'training/strategy': getattr(training_config, 'strategy', None),
                'training/device': getattr(training_config, 'device', None),
                'training/gradient_checkpointing': getattr(training_config, 'gradient_checkpointing', None),
            })
        
        # Optimizer configuration
        if isinstance(training_config, dict):
            optimizer_config = training_config.get('optimizer', {})
            if isinstance(optimizer_config, dict):
                config_dict.update({
                    'optimizer/lr': optimizer_config.get('lr', None),
                    'optimizer/weight_decay': optimizer_config.get('weight_decay', None),
                    'optimizer/betas': optimizer_config.get('betas', None),
                })
        else:
            if hasattr(training_config, 'optimizer'):
                opt_config = training_config.optimizer
                config_dict.update({
                    'optimizer/lr': getattr(opt_config, 'lr', None),
                    'optimizer/weight_decay': getattr(opt_config, 'weight_decay', None),
                    'optimizer/betas': getattr(opt_config, 'betas', None),
                })
        
        # Progressive training configuration
        if isinstance(training_config, dict):
            progressive_config = training_config.get('progressive', {})
            if isinstance(progressive_config, dict):
                config_dict.update({
                    'progressive/enabled': progressive_config.get('enabled', None),
                    'progressive/max_stacks': progressive_config.get('max_stacks', None),
                    'progressive/target_stacks': progressive_config.get('target_stacks', None),
                    'progressive/trunk_strategy': progressive_config.get('trunk_strategy', None),
                    'progressive/training_objective': progressive_config.get('training_objective', None),
                })
        else:
            if hasattr(training_config, 'progressive'):
                prog_config = training_config.progressive
                config_dict.update({
                    'progressive/enabled': getattr(prog_config, 'enabled', None),
                    'progressive/max_stacks': getattr(prog_config, 'max_stacks', None),
                    'progressive/target_stacks': getattr(prog_config, 'target_stacks', None),
                    'progressive/trunk_strategy': getattr(prog_config, 'trunk_strategy', None),
                    'progressive/training_objective': getattr(prog_config, 'training_objective', None),
                })
        
        # Data configuration
        if hasattr(self.config, 'data'):
            data_config = self.config.data
            if isinstance(data_config, dict):
                config_dict.update({
                    'data/num_samples': data_config.get('num_samples', None),
                    'data/max_length': data_config.get('max_length', None),
                    'data/padding': data_config.get('padding', None),
                })
            else:
                config_dict.update({
                    'data/num_samples': getattr(data_config, 'num_samples', None),
                    'data/max_length': getattr(data_config, 'max_length', None),
                    'data/padding': getattr(data_config, 'padding', None),
                })
        
        return config_dict
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.initialized:
            return
        
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics to wandb: {e}")
    
    def log_stack_metrics(self, stack_idx: int, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics for a specific stack.
        
        Args:
            stack_idx: Stack index
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.initialized:
            return
        
        # Prefix metrics with stack index
        stack_metrics = {f"stack_{stack_idx}/{k}": v for k, v in metrics.items()}
        self.log_metrics(stack_metrics, step)
    
    def log_progressive_metrics(self, current_stacks: int, total_stacks: int, 
                               stack_metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log progressive training metrics.
        
        Args:
            current_stacks: Current number of stacks
            total_stacks: Total number of stacks
            stack_metrics: Metrics for the current stack
            step: Optional step number
        """
        if not self.initialized:
            return
        
        progressive_metrics = {
            'progressive/current_stacks': current_stacks,
            'progressive/total_stacks': total_stacks,
            'progressive/progress': current_stacks / total_stacks,
        }
        progressive_metrics.update(stack_metrics)
        
        self.log_metrics(progressive_metrics, step)
    
    def log_model_artifacts(self, model_path: str, artifact_name: str, 
                           artifact_type: str = "model"):
        """
        Log model artifacts to wandb.
        
        Args:
            model_path: Path to the model file
            artifact_name: Name for the artifact
            artifact_type: Type of artifact (model, checkpoint, etc.)
        """
        if not self.initialized:
            return
        
        try:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            logger.info(f"Logged artifact: {artifact_name}")
        except Exception as e:
            logger.error(f"Failed to log artifact to wandb: {e}")
    
    def log_system_metrics(self):
        """Log system metrics (GPU memory, CPU usage, etc.)."""
        if not self.initialized:
            return
        
        try:
            # GPU metrics
            if torch.cuda.is_available():
                gpu_metrics = {
                    'system/gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                    'system/gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                    'system/gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0,
                }
                self.log_metrics(gpu_metrics)
            
            # CPU metrics
            import psutil
            cpu_metrics = {
                'system/cpu_percent': psutil.cpu_percent(),
                'system/memory_percent': psutil.virtual_memory().percent,
            }
            self.log_metrics(cpu_metrics)
            
        except Exception as e:
            logger.error(f"Failed to log system metrics to wandb: {e}")
    
    def watch_model(self, model: torch.nn.Module, log: str = "gradients", 
                   log_freq: int = 100):
        """
        Watch model for gradients and parameters.
        
        Args:
            model: PyTorch model to watch
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency
        """
        if not self.initialized:
            return
        
        try:
            wandb.watch(model, log=log, log_freq=log_freq)
            logger.info(f"Started watching model with log={log}, log_freq={log_freq}")
        except Exception as e:
            logger.error(f"Failed to watch model in wandb: {e}")
    
    def finish(self):
        """Finish wandb run."""
        if not self.initialized:
            return
        
        try:
            wandb.finish()
            logger.info("Finished wandb run")
        except Exception as e:
            logger.error(f"Failed to finish wandb run: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


def create_wandb_logger(config, run_name: Optional[str] = None) -> Optional[WandBLogger]:
    """
    Create a wandb logger if wandb is available and enabled.
    
    Args:
        config: Training configuration (StackWiseConfig or TrainingConfig)
        run_name: Optional custom run name
        
    Returns:
        WandBLogger instance or None if not available/enabled
    """
    if not WANDB_AVAILABLE:
        logger.warning("wandb not available. Install with: pip install wandb")
        return None
    
    # Handle both StackWiseConfig and TrainingConfig
    if hasattr(config, 'training'):
        training_config = config.training
    else:
        training_config = config
    
    # Handle both dict and object configs
    if isinstance(training_config, dict):
        use_wandb = training_config.get('use_wandb', False)
    else:
        use_wandb = getattr(training_config, 'use_wandb', False)
    
    if not use_wandb:
        logger.info("wandb logging disabled in configuration")
        return None
    
    return WandBLogger(config, run_name)
