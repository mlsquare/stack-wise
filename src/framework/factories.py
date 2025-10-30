"""
Factories for building data loaders, models, and optimizers.

These factories accept either config dictionaries or pre-built objects
and delegate to existing builders in the codebase.
"""

from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import logging

from .adapters import DataAdapter, ModelAdapter, OptimizerAdapter
from .specs import (
    BatchSpec, EmbeddingSpec, RackSpec, HeadSpec,
    OptimizerSpec, DataSpec, validate_batch_model_compatibility
)

logger = logging.getLogger(__name__)


def build_data_loader(config_or_data: Union[Dict, DataLoader], 
                     phase_config: Optional[Dict] = None) -> DataAdapter:
    """
    Build data loader from config or use existing dataloader.
    
    Args:
        config_or_data: Either a config dict or an existing DataLoader
        phase_config: Optional phase-specific config overrides
        
    Returns:
        Data adapter wrapping the loader
    """
    if isinstance(config_or_data, DataLoader):
        # Use existing dataloader
        # Extract spec from dataloader if possible
        batch_spec = _infer_batch_spec_from_dataloader(config_or_data)
        return DataAdapter(config_or_data, batch_spec)
    
    # Build from config
    config = dict(config_or_data or {})
    if phase_config:
        config.update(phase_config)
    
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    # Extract parameters
    batch_size = training_config.get('batch_size', 4)
    seq_len = training_config.get('seq_len', 512)
    vocab_size = config.get('model', {}).get('vocab_size', 32000)
    
    # Create dummy dataset for now (in real usage, load from config)
    use_dummy = data_config.get('use_dummy_data', True)
    num_samples = data_config.get('num_samples', 128)
    
    if use_dummy:
        dataset = _create_dummy_dataset(batch_size, seq_len, vocab_size, num_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        raise NotImplementedError("Real dataset loading not yet implemented")
    
    # Create spec and adapter
    batch_spec = BatchSpec(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size
    )
    
    return DataAdapter(dataloader, batch_spec)


def build_model(config_or_model: Union[Dict, nn.Module],
               phase_config: Optional[Dict] = None) -> ModelAdapter:
    """
    Build model from config or use existing model.
    
    Args:
        config_or_model: Either a config dict or an existing model
        phase_config: Optional phase-specific config overrides
        
    Returns:
        Model adapter wrapping the model
    """
    if isinstance(config_or_model, nn.Module):
        # Use existing model
        return ModelAdapter(rack=config_or_model)
    
    # Build from config
    config = dict(config_or_model or {})
    if phase_config:
        config.update(phase_config)
    
    # Import here to avoid circular imports
    try:
        from ..model.architecture import create_rack_from_config, Rack
        from ..config.base import StackWiseConfig
    except ImportError:
        from model.architecture import create_rack_from_config, Rack
        from config.base import StackWiseConfig
    
    # Convert dict to config if needed
    if isinstance(config, dict):
        if 'model' in config and 'training' in config and 'data' in config:
            # Full StackWise config
            config_obj = StackWiseConfig.from_dict(config)
        else:
            # Model-only config
            raise ValueError("Full config dictionary required when building from config")
    else:
        config_obj = config
    
    # Build rack from config
    rack = create_rack_from_config(config_obj.to_dict())
    
    return ModelAdapter(rack=rack)


def build_optimizer(config_or_params: Union[Dict, List[torch.nn.Parameter]],
                   model_adapter: ModelAdapter,
                   phase_config: Optional[Dict] = None) -> OptimizerAdapter:
    """
    Build optimizer from config or parameters.
    
    Args:
        config_or_params: Either a config dict or list of model parameters
        model_adapter: Model adapter to get trainable parameters from
        phase_config: Optional phase-specific config overrides
        
    Returns:
        Optimizer adapter
    """
    # Get trainable parameters
    if isinstance(config_or_params, list):
        params = config_or_params
        config = {}
    else:
        params = model_adapter.get_trainable_parameters()
        config = dict(config_or_params or {})
    
    if phase_config:
        config.update(phase_config)
    
    # Extract optimizer config
    training_config = config.get('training', {})
    optimizer_config = training_config.get('optimizer', {})
    
    # Create optimizer spec
    if isinstance(optimizer_config, dict):
        spec = OptimizerSpec.from_dict(optimizer_config)
    else:
        # Try to extract from config object
        spec = OptimizerSpec(
            optimizer_type=getattr(optimizer_config, 'optimizer_type', 'AdamW'),
            lr=getattr(optimizer_config, 'lr', 1e-4),
            weight_decay=getattr(optimizer_config, 'weight_decay', 0.1),
            betas=getattr(optimizer_config, 'betas', (0.9, 0.95)),
            eps=getattr(optimizer_config, 'eps', 1e-8)
        )
    
    # Use existing optimizer factory if available
    try:
        from ..config.base import create_optimizer, OptimizerConfig
    except ImportError:
        from config.base import create_optimizer, OptimizerConfig
    
    # Convert spec to OptimizerConfig
    opt_config = OptimizerConfig(
        optimizer_type=spec.optimizer_type,
        lr=spec.lr,
        weight_decay=spec.weight_decay,
        betas=list(spec.betas),
        eps=spec.eps
    )
    
    # Create optimizer
    optimizer = create_optimizer(params, opt_config)
    
    return OptimizerAdapter(optimizer, spec)


def _create_dummy_dataset(batch_size: int, seq_len: int, vocab_size: int, num_samples: int) -> Dataset:
    """Create a dummy dataset for testing."""
    # Generate random input and labels
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = input_ids.clone()
    
    return TensorDataset(input_ids, labels)


def _infer_batch_spec_from_dataloader(dataloader: DataLoader) -> BatchSpec:
    """Infer batch spec from dataloader by sampling a batch."""
    try:
        # Sample a batch
        batch_iter = iter(dataloader)
        batch = next(batch_iter)
        
        # Try to extract information from batch
        if isinstance(batch, dict):
            input_ids = batch.get('input_ids', batch.get('input'))
            if input_ids is None and len(batch) > 0:
                input_ids = next(iter(batch.values()))
        elif isinstance(batch, (tuple, list)):
            input_ids = batch[0]
        else:
            input_ids = batch
        
        if isinstance(input_ids, torch.Tensor):
            batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
            # Infer vocab_size from values (not ideal but best we can do)
            vocab_size = input_ids.max().item() + 1 if input_ids.numel() > 0 else 32000
            
            return BatchSpec(
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                dtype=input_ids.dtype
            )
    except Exception as e:
        logger.warning(f"Could not infer batch spec from dataloader: {e}")
    
    # Fallback to default spec
    return BatchSpec(
        batch_size=4,
        seq_len=512,
        vocab_size=32000
    )


def validate_training_state(data_adapter: DataAdapter,
                           model_adapter: ModelAdapter,
                           optimizer_adapter: OptimizerAdapter) -> bool:
    """
    Validate that data, model, and optimizer specs are compatible.
    
    Args:
        data_adapter: Data adapter
        model_adapter: Model adapter
        optimizer_adapter: Optimizer adapter
        
    Returns:
        True if all components are compatible
    """
    # Get specs
    batch_spec = data_adapter.get_batch_spec()
    embedding_spec = model_adapter.get_embedding_spec()
    head_spec = model_adapter.get_head_spec()
    
    if embedding_spec is None or head_spec is None:
        logger.warning("Could not extract embedding or head specs, skipping validation")
        return True
    
    # Validate compatibility
    return validate_batch_model_compatibility(
        batch_spec,
        embedding_spec,
        head_spec
    )

