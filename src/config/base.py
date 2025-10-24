"""
Base configuration classes for StackWise.
Provides hierarchical configuration with validation and defaults.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Union
import yaml
from pathlib import Path


# Type definitions for better type checking
AttentionType = Literal["standard", "gqa", "mla", "kernel"]
AttentionMode = Literal["bidirectional", "causal"]
FineTuneMode = Literal["clm", "mlm", "diffusion"]
KernelType = Literal["gaussian", "laplacian", "uniform"]


@dataclass
class BaseConfig:
    """Base configuration class with common functionality."""
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        pass
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'BaseConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'BaseConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


@dataclass
class ModelConfig(BaseConfig):
    """Model architecture configuration."""
    
    # Model dimensions
    vocab_size: Optional[int] = None  # Will be set from tokenizer
    d_model: int = 4096
    n_layers: int = 8
    n_heads: int = 32
    n_kv_heads: int = 8
    d_ff: int = 14336
    
    # Attention configuration
    attention_type: AttentionType = "standard"
    attention_mode: AttentionMode = "bidirectional"
    
    # MLA specific parameters
    mla_rq: int = 1024
    mla_rkv: int = 512
    
    # Kernel attention parameters
    kernel_type: KernelType = "gaussian"
    kernel_dim: int = 64
    
    # Normalization and MLP
    dropout: float = 0.0
    tie_embeddings: bool = True

    # Positional encoding
    use_rope: bool = True
    rope_theta: float = 10000.0
    
    # Mask-diffusion parameters
    mask_fraction_min: float = 0.15
    mask_fraction_max: float = 0.90
    special_mask_id: int = 4
    
    # Tokenizer and embedding configuration
    tokenizer_embedding: dict = field(default_factory=lambda: {
        "family": "gpt2",
        "embedding_option": "embed_tokens", 
        "freeze_embeddings": True,
        "adapter_hidden_dim": None
    })
    
    def validate(self) -> None:
        """Validate model configuration."""
        super().validate()
        
        # Validate dimensions
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        
        # Validate attention configuration
        if self.attention_type == "gqa" and self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        
        if self.attention_type == "mla":
            if self.mla_rq <= 0:
                raise ValueError("mla_rq must be positive")
            if self.mla_rkv <= 0:
                raise ValueError("mla_rkv must be positive")

        # Validate normalization and embedding behaviour
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")

        # Validate mask fractions
        if not 0 <= self.mask_fraction_min <= 1:
            raise ValueError("mask_fraction_min must be between 0 and 1")
        if not 0 <= self.mask_fraction_max <= 1:
            raise ValueError("mask_fraction_max must be between 0 and 1")
        if self.mask_fraction_min >= self.mask_fraction_max:
            raise ValueError("mask_fraction_min must be less than mask_fraction_max")

        if self.special_mask_id < 0:
            raise ValueError("special_mask_id must be non-negative")
        
        # Validate RoPE parameters
        if self.use_rope and self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
    
    def set_vocab_size(self, vocab_size: int) -> None:
        """Set vocabulary size from tokenizer."""
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
    
    def set_tokenizer_embedding(self, family: str, embedding_option: str = "embed_tokens", 
                               freeze_embeddings: bool = True, adapter_hidden_dim: Optional[int] = None) -> None:
        """Set tokenizer and embedding configuration."""
        self.tokenizer_embedding = {
            "family": family,
            "embedding_option": embedding_option,
            "freeze_embeddings": freeze_embeddings,
            "adapter_hidden_dim": adapter_hidden_dim
        }


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration."""
    
    # Training parameters
    lr: float = 1e-4
    weight_decay: float = 0.1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    batch_size: int = 4
    seq_len: int = 512
    max_steps: int = 200
    
    # Device and memory
    device: str = "cuda"
    gradient_checkpointing: bool = False
    mixed_precision: bool = False
    
    # Layer-wise training
    layerwise_training: bool = True
    activation_cache_dir: str = "./cache"
    save_activations: bool = True
    
    # Caching configuration
    cache_mode: str = "layerwise"  # "layerwise" or "fusion"
    fusion_evaluation: bool = False
    save_fused_checkpoints: bool = False
    
    # Mask-diffusion training
    min_mask_fraction: float = 0.15
    max_mask_fraction: float = 0.90
    mask_schedule_type: str = "linear"  # "linear", "exponential", "cosine"
    mask_token_id: int = 0
    epochs_per_layer: int = 1
    learning_rate: float = 1e-4
    
    # Fusion and fine-tuning
    fusion_enabled: bool = True
    joint_tuning_steps: int = 50
    fine_tune_mode: FineTuneMode = "clm"
    
    # Training modes
    mode: str = "layerwise"  # layerwise | blockwise | fused
    block_size: int = 4
    fusion_mode: str = "frozen"  # frozen | trainable
    
    # Run identification and organization
    run_id: str = "default_run"
    total_blocks: int = 2
    
    # QLoRA and quantization settings
    qlora_enabled: bool = True
    qlora_lr: float = 1e-5
    current_block_lr: float = 1e-4
    quantization_enabled: bool = True
    quantization_type: str = "fp16"  # fp4 | fp8 | fp16 | fp32
    
    # Time-step-based masking
    time_step_masking: bool = True
    num_time_steps: int = 8
    time_step_mask_fractions: List[float] = field(default_factory=lambda: [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 100
    checkpoint_dir: str = "./checkpoints"
    
    def validate(self) -> None:
        """Validate training configuration."""
        super().validate()
        
        # Validate learning parameters
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if len(self.betas) != 2:
            raise ValueError("betas must have exactly 2 elements")
        if not all(0 <= b <= 1 for b in self.betas):
            raise ValueError("betas must be between 0 and 1")
        
        # Validate batch parameters
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        # Validate device
        if self.device not in ["cuda", "cpu", "auto"]:
            raise ValueError("device must be 'cuda', 'cpu', or 'auto'")
        
        # Validate run identification
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError("run_id must be a non-empty string")
        if self.total_blocks <= 0:
            raise ValueError("total_blocks must be positive")
        
        # Validate QLoRA and quantization settings
        if self.qlora_lr <= 0:
            raise ValueError("qlora_lr must be positive")
        if self.current_block_lr <= 0:
            raise ValueError("current_block_lr must be positive")
        if self.quantization_type not in ["fp4", "fp8", "fp16", "fp32"]:
            raise ValueError("quantization_type must be one of: fp4, fp8, fp16, fp32")
        
        # Validate time-step masking
        if self.num_time_steps <= 0:
            raise ValueError("num_time_steps must be positive")
        if len(self.time_step_mask_fractions) != self.num_time_steps:
            raise ValueError("time_step_mask_fractions length must match num_time_steps")
        if not all(0 <= frac <= 1 for frac in self.time_step_mask_fractions):
            raise ValueError("all time_step_mask_fractions must be between 0 and 1")
        if not all(self.time_step_mask_fractions[i] <= self.time_step_mask_fractions[i+1] 
                   for i in range(len(self.time_step_mask_fractions)-1)):
            raise ValueError("time_step_mask_fractions must be in ascending order")


@dataclass
class DataConfig(BaseConfig):
    """Data configuration."""
    
    # Dataset parameters
    dataset_path: Optional[str] = None
    use_dummy_data: bool = True
    num_samples: int = 128
    
    # Data preprocessing
    tokenizer_path: Optional[str] = None
    max_length: int = 512
    padding: str = "right"
    
    # Data loading
    num_workers: int = 0
    pin_memory: bool = True
    shuffle: bool = True
    
    def validate(self) -> None:
        """Validate data configuration."""
        super().validate()

        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.padding not in ["left", "right"]:
            raise ValueError("padding must be 'left' or 'right'")

        if not self.use_dummy_data and not self.dataset_path:
            raise ValueError("dataset_path must be provided when use_dummy_data is False")


@dataclass
class StackWiseConfig(BaseConfig):
    """Main configuration class combining all sub-configurations."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def validate(self) -> None:
        """Validate entire configuration."""
        super().validate()
        self.model.validate()
        self.training.validate()
        self.data.validate()
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'StackWiseConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract sub-configurations
        model_config = ModelConfig.from_dict(config_dict.get('model', {}))
        training_config = TrainingConfig.from_dict(config_dict.get('training', {}))
        data_config = DataConfig.from_dict(config_dict.get('data', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config
        )
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'data': self.data.to_dict()
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def set_vocab_size(self, vocab_size: int) -> None:
        """Set vocabulary size from tokenizer."""
        self.model.set_vocab_size(vocab_size)
