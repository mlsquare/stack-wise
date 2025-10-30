"""
Spec dataclasses for framework glue.

These specs provide contracts between data, model, and optimizer planes
without owning concrete implementations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import torch


@dataclass
class BatchSpec:
    """Specification for a batch of data."""
    
    batch_size: int
    seq_len: int
    vocab_size: int
    dtype: torch.dtype = torch.long
    device: Optional[torch.device] = None
    keys: List[str] = field(default_factory=lambda: ['input_ids', 'labels', 'masks'])
    
    def __post_init__(self):
        """Validate batch spec."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
    
    def validate_batch(self, batch: Dict[str, torch.Tensor]) -> bool:
        """
        Validate that a batch matches this spec.
        
        Args:
            batch: Batch dictionary
            
        Returns:
            True if batch matches spec
        """
        if not isinstance(batch, dict):
            return False
        
        # Check required keys
        for key in self.keys:
            if key not in batch:
                return False
            
            tensor = batch[key]
            
            # Check tensor properties
            if not isinstance(tensor, torch.Tensor):
                return False
            
            if len(tensor.shape) < 2:
                return False
            
            batch_dim, seq_dim = tensor.shape[0], tensor.shape[1]
            
            # Check batch size
            if batch_dim != self.batch_size:
                return False
            
            # Check sequence length
            if seq_dim != self.seq_len:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary."""
        return {
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'vocab_size': self.vocab_size,
            'dtype': str(self.dtype),
            'device': str(self.device) if self.device else None,
            'keys': self.keys
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BatchSpec':
        """Create spec from dictionary."""
        # Convert dtype string back to torch.dtype
        dtype_str = d.get('dtype', 'torch.int64')
        if isinstance(dtype_str, str):
            dtype = getattr(torch, dtype_str.split('.')[-1])
        else:
            dtype = dtype_str
        
        # Convert device string back to torch.device
        device_str = d.get('device')
        if device_str and isinstance(device_str, str):
            device = torch.device(device_str)
        else:
            device = None
        
        return cls(
            batch_size=d['batch_size'],
            seq_len=d['seq_len'],
            vocab_size=d['vocab_size'],
            dtype=dtype,
            device=device,
            keys=d.get('keys', ['input_ids', 'labels', 'masks'])
        )


@dataclass
class EmbeddingSpec:
    """Specification for embedding layer."""
    
    vocab_size: int
    d_model: int
    tie_to_head: bool = True
    
    def __post_init__(self):
        """Validate embedding spec."""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'tie_to_head': self.tie_to_head
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EmbeddingSpec':
        """Create spec from dictionary."""
        return cls(**d)


@dataclass
class RackSpec:
    """Specification for rack (stack collection)."""
    
    n_stacks: int
    stacks_per_layer: int = field(default_factory=list)
    d_model: int = 4096
    frozen_stacks: List[int] = field(default_factory=list)
    precision: Union[str, torch.dtype] = torch.float32
    
    def __post_init__(self):
        """Validate rack spec."""
        if self.n_stacks <= 0:
            raise ValueError("n_stacks must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary."""
        precision_str = self.precision
        if isinstance(self.precision, torch.dtype):
            precision_str = str(self.precision)
        
        return {
            'n_stacks': self.n_stacks,
            'stacks_per_layer': self.stacks_per_layer,
            'd_model': self.d_model,
            'frozen_stacks': self.frozen_stacks,
            'precision': precision_str
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RackSpec':
        """Create spec from dictionary."""
        return cls(**d)


@dataclass
class HeadSpec:
    """Specification for head/projection layer."""
    
    d_model: int
    output_dim: int  # e.g., vocab_size for language modeling
    has_bias: bool = True
    tie_to_embedding: bool = True
    
    def __post_init__(self):
        """Validate head spec."""
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary."""
        return {
            'd_model': self.d_model,
            'output_dim': self.output_dim,
            'has_bias': self.has_bias,
            'tie_to_embedding': self.tie_to_embedding
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HeadSpec':
        """Create spec from dictionary."""
        return cls(**d)


@dataclass
class OptimizerSpec:
    """Specification for optimizer configuration."""
    
    optimizer_type: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    
    # Per-parameter group overrides
    param_groups: List[Dict[str, Any]] = field(default_factory=list)
    
    # Mixed precision settings
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float16
    
    def __post_init__(self):
        """Validate optimizer spec."""
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if len(self.betas) != 2:
            raise ValueError("betas must have 2 elements")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary."""
        return {
            'optimizer_type': self.optimizer_type,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'betas': list(self.betas),
            'eps': self.eps,
            'param_groups': self.param_groups,
            'use_amp': self.use_amp,
            'amp_dtype': str(self.amp_dtype) if isinstance(self.amp_dtype, torch.dtype) else self.amp_dtype
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OptimizerSpec':
        """Create spec from dictionary."""
        # Convert betas tuple
        betas = d.get('betas', (0.9, 0.95))
        if isinstance(betas, list):
            betas = tuple(betas)
        
        # Convert amp_dtype
        amp_dtype = d.get('amp_dtype', torch.float16)
        if isinstance(amp_dtype, str):
            amp_dtype = getattr(torch, amp_dtype.split('.')[-1])
        
        return cls(
            optimizer_type=d['optimizer_type'],
            lr=d['lr'],
            weight_decay=d['weight_decay'],
            betas=betas,
            eps=d['eps'],
            param_groups=d.get('param_groups', []),
            use_amp=d.get('use_amp', False),
            amp_dtype=amp_dtype
        )


@dataclass
class DataSpec:
    """Specification for data loader."""
    
    dataset_name: str
    batch_size: int
    seq_len: int
    shuffle: bool = True
    num_workers: int = 0
    
    # Transforms and augmentation
    transforms: List[str] = field(default_factory=list)
    augmentation: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data spec."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary."""
        return {
            'dataset_name': self.dataset_name,
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            'transforms': self.transforms,
            'augmentation': self.augmentation
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataSpec':
        """Create spec from dictionary."""
        return cls(**d)


def validate_batch_model_compatibility(
    batch_spec: BatchSpec,
    embedding_spec: EmbeddingSpec,
    head_spec: HeadSpec
) -> bool:
    """
    Validate that batch, embedding, and head specs are compatible.
    
    Args:
        batch_spec: Batch specification
        embedding_spec: Embedding specification
        head_spec: Head specification
        
    Returns:
        True if all specs are compatible
    """
    # Check vocabulary size alignment
    if batch_spec.vocab_size != embedding_spec.vocab_size:
        return False
    
    # Check output dimension alignment
    if head_spec.output_dim != embedding_spec.vocab_size:
        return False
    
    # Check d_model alignment
    if embedding_spec.d_model != head_spec.d_model:
        return False
    
    return True

