"""
Adapters for bridging existing model/data/optimizer to framework specs.

These adapters provide the glue between concrete implementations and
the abstract specifications.
"""

from typing import Dict, List, Optional, Union, Any, Iterator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .specs import (
    BatchSpec, EmbeddingSpec, RackSpec, HeadSpec,
    OptimizerSpec, DataSpec
)


class DataAdapter:
    """Adapter for data loaders to conform to BatchSpec."""
    
    def __init__(self, dataloader: DataLoader, spec: BatchSpec):
        """
        Initialize data adapter.
        
        Args:
            dataloader: Concrete data loader
            spec: Batch specification
        """
        self.dataloader = dataloader
        self.spec = spec
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over batches."""
        for batch in self.dataloader:
            # Ensure batch matches spec
            adapted_batch = self._adapt_batch(batch)
            
            # Validate batch matches spec
            if not self.spec.validate_batch(adapted_batch):
                continue  # Skip invalid batches
            
            yield adapted_batch
    
    def _adapt_batch(self, batch: Union[Dict[str, torch.Tensor], tuple]) -> Dict[str, torch.Tensor]:
        """Adapt batch to match spec."""
        # Convert tuple to dict if needed
        if isinstance(batch, (tuple, list)):
            if len(batch) >= 2:
                batch_dict = {'input_ids': batch[0], 'labels': batch[1]}
            else:
                batch_dict = {'input_ids': batch[0]}
        else:
            batch_dict = batch.copy()
        
        # Ensure all required keys are present
        for key in self.spec.keys:
            if key not in batch_dict:
                # Create dummy tensor
                batch_dict[key] = torch.zeros(
                    (self.spec.batch_size, self.spec.seq_len),
                    dtype=self.spec.dtype,
                    device=self.spec.device
                )
        
        return batch_dict
    
    def get_batch_spec(self) -> BatchSpec:
        """Get batch specification."""
        return self.spec


class ModelAdapter:
    """Adapter for model components to conform to framework specs."""
    
    def __init__(self, 
                 embedding: Optional[nn.Module] = None,
                 rack: Optional[nn.Module] = None,
                 head: Optional[nn.Module] = None):
        """
        Initialize model adapter.
        
        Args:
            embedding: Embedding layer
            rack: Rack (stack collection)
            head: Head/projection layer
        """
        self.embedding = embedding
        self.rack = rack
        self.head = head
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            
        Returns:
            Output logits
        """
        # Get embeddings
        if self.embedding is not None:
            x = self.embedding(input_ids)
        else:
            # If no embedding, assume input_ids are already embeddings
            x = input_ids
        
        # Forward through rack
        if self.rack is not None:
            if hasattr(self.rack, 'forward_from_stack'):
                # Progressive rack with forward_from_stack
                x = self.rack.forward_from_stack(x, 0, None, attention_mask)
            else:
                # Standard rack
                x = self.rack.forward(x, attention_mask) if hasattr(self.rack, 'forward') else x
        
        # Forward through head
        if self.head is not None:
            logits = self.head(x)
        else:
            logits = x
        
        return logits
    
    def get_embedding_spec(self) -> Optional[EmbeddingSpec]:
        """Extract embedding spec from embedding layer."""
        if self.embedding is None:
            return None
        
        # Try to extract spec from embedding
        if hasattr(self.embedding, 'get_spec'):
            return self.embedding.get_spec()
        
        # Infer spec from embedding parameters
        if hasattr(self.embedding, 'weight'):
            vocab_size, d_model = self.embedding.weight.shape
            return EmbeddingSpec(
                vocab_size=vocab_size,
                d_model=d_model,
                tie_to_head=False  # Unknown
            )
        
        return None
    
    def get_rack_spec(self) -> Optional[RackSpec]:
        """Extract rack spec from rack."""
        if self.rack is None:
            return None
        
        # Try to extract spec from rack
        if hasattr(self.rack, 'to_spec'):
            return self.rack.to_spec()
        
        # Try to infer spec from rack structure
        if hasattr(self.rack, 'stacks'):
            n_stacks = len(self.rack.stacks)
            d_model = getattr(self.rack, 'd_model', 4096)
            
            return RackSpec(
                n_stacks=n_stacks,
                d_model=d_model
            )
        
        return None
    
    def get_head_spec(self) -> Optional[HeadSpec]:
        """Extract head spec from head layer."""
        if self.head is None:
            return None
        
        # Try to extract spec from head
        if hasattr(self.head, 'get_spec'):
            return self.head.get_spec()
        
        # Infer spec from head parameters
        if hasattr(self.head, 'weight'):
            d_model, output_dim = self.head.weight.shape
            has_bias = hasattr(self.head, 'bias') and self.head.bias is not None
            
            return HeadSpec(
                d_model=d_model,
                output_dim=output_dim,
                has_bias=has_bias,
                tie_to_embedding=False  # Unknown
            )
        
        return None
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters."""
        params = []
        
        if self.embedding is not None:
            params.extend([p for p in self.embedding.parameters() if p.requires_grad])
        
        if self.rack is not None:
            params.extend([p for p in self.rack.parameters() if p.requires_grad])
        
        if self.head is not None:
            params.extend([p for p in self.head.parameters() if p.requires_grad])
        
        return params
    
    def freeze_components(self, components: List[str]):
        """
        Freeze specified components.
        
        Args:
            components: List of components to freeze ('embedding', 'rack', 'head')
        """
        if 'embedding' in components and self.embedding is not None:
            for param in self.embedding.parameters():
                param.requires_grad = False
        
        if 'rack' in components and self.rack is not None:
            for param in self.rack.parameters():
                param.requires_grad = False
        
        if 'head' in components and self.head is not None:
            for param in self.head.parameters():
                param.requires_grad = False


class OptimizerAdapter:
    """Adapter for optimizer to conform to OptimizerSpec."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, spec: OptimizerSpec):
        """
        Initialize optimizer adapter.
        
        Args:
            optimizer: Concrete optimizer
            spec: Optimizer specification
        """
        self.optimizer = optimizer
        self.spec = spec
    
    def step(self, loss: torch.Tensor):
        """Perform optimization step."""
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0].get('lr', self.spec.lr)
    
    def set_lr(self, lr: float):
        """Set learning rate."""
        for group in self.optimizer.param_groups:
            group['lr'] = lr
    
    def get_spec(self) -> OptimizerSpec:
        """Get optimizer specification."""
        return self.spec
    
    def state_dict(self) -> Dict:
        """Get optimizer state dict."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)


def create_data_adapter(dataloader: DataLoader, 
                       batch_size: int,
                       seq_len: int,
                       vocab_size: int,
                       dtype: torch.dtype = torch.long,
                       device: Optional[torch.device] = None) -> DataAdapter:
    """
    Create data adapter from dataloader and parameters.
    
    Args:
        dataloader: Data loader
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        dtype: Data type
        device: Device
        
    Returns:
        Data adapter
    """
    spec = BatchSpec(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        dtype=dtype,
        device=device
    )
    return DataAdapter(dataloader, spec)


def create_model_adapter(model: nn.Module) -> ModelAdapter:
    """
    Create model adapter from model.
    
    Tries to extract embedding, rack, and head from the model.
    
    Args:
        model: Model to adapt
        
    Returns:
        Model adapter
    """
    embedding = None
    rack = None
    head = None
    
    # Try to extract components from model
    if hasattr(model, 'embedding'):
        embedding = model.embedding
    
    if hasattr(model, 'stacks') or hasattr(model, 'rack'):
        rack = model if hasattr(model, 'stacks') else getattr(model, 'rack', None)
    
    if hasattr(model, 'lm_head'):
        head = model.lm_head
    
    return ModelAdapter(embedding=embedding, rack=rack, head=head)


def create_optimizer_adapter(model_params: List[torch.nn.Parameter],
                            spec: OptimizerSpec) -> OptimizerAdapter:
    """
    Create optimizer adapter from model parameters and spec.
    
    Args:
        model_params: Model parameters to optimize
        spec: Optimizer specification
        
    Returns:
        Optimizer adapter
    """
    # Create optimizer based on spec
    if spec.optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model_params,
            lr=spec.lr,
            weight_decay=spec.weight_decay,
            betas=spec.betas,
            eps=spec.eps
        )
    elif spec.optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            model_params,
            lr=spec.lr,
            weight_decay=spec.weight_decay,
            betas=spec.betas,
            eps=spec.eps
        )
    elif spec.optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model_params,
            lr=spec.lr,
            weight_decay=spec.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {spec.optimizer_type}")
    
    return OptimizerAdapter(optimizer, spec)

