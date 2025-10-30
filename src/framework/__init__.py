"""
Framework glue for StackWise.

This module provides the specification, ETL, scheduling, and orchestration
layer that glues together data, models, and optimizers without owning
concrete implementations.
"""

from .specs import (
    BatchSpec, EmbeddingSpec, RackSpec, HeadSpec, 
    OptimizerSpec, DataSpec
)
from .adapters import (
    DataAdapter, ModelAdapter, OptimizerAdapter
)
from .factories import build_data_loader, build_model, build_optimizer

__all__ = [
    # Specs
    "BatchSpec",
    "EmbeddingSpec", 
    "RackSpec",
    "HeadSpec",
    "OptimizerSpec",
    "DataSpec",
    
    # Adapters
    "DataAdapter",
    "ModelAdapter",
    "OptimizerAdapter",
    
    # Factories
    "build_data_loader",
    "build_model",
    "build_optimizer",
]

__version__ = "1.0.0"

