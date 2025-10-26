"""
Base configuration classes for StackWise.
Provides hierarchical configuration with validation and defaults.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Union, Any, Dict
import yaml
from pathlib import Path


# Type definitions for better type checking
AttentionType = Literal["mha", "mla"]  # GQA is determined by n_kv_heads
AttentionMode = Literal["bidirectional", "causal"]
FineTuneMode = Literal["clm", "mlm", "diffusion"]
KernelType = Literal["linear", "gaussian", "laplacian", "uniform"]
AttentionPreset = Literal["bert_style", "gpt_style", "efficient_gqa", "mla_attention", "kernel_attention", "mlgka", "custom"]


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
class AttentionConfig(BaseConfig):
    """Custom attention configuration (used when preset is 'custom')."""
    
    attention_type: AttentionType = "mha"
    attention_mode: AttentionMode = "bidirectional"
    
    # MLA specific parameters
    mla_rq: int = 1024
    mla_rkv: int = 512
    
    # Kernel attention parameters
    kernel_type: KernelType = "linear"
    kernel_dim: int = 64


@dataclass
class ArchitectureConfig(BaseConfig):
    """Architecture configuration for stacks and blocks."""
    
    n_stacks: int = 2
    blocks_per_stack: int = 4
    
    def validate(self) -> None:
        """Validate architecture configuration."""
        super().validate()
        
        if self.n_stacks <= 0:
            raise ValueError("n_stacks must be positive")
        if self.blocks_per_stack <= 0:
            raise ValueError("blocks_per_stack must be positive")


@dataclass
class ModelConfig(BaseConfig):
    """Model architecture configuration."""
    
    # Model dimensions
    vocab_size: Optional[int] = None  # Will be set from tokenizer
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    d_ff: int = 14336
    
    # Architecture configuration
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    
    # Attention configuration
    attention_preset: AttentionPreset = "bert_style"
    attention_custom: AttentionConfig = field(default_factory=AttentionConfig)
    
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
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create ModelConfig from dictionary, handling nested architecture."""
        # Copy to avoid mutating input
        cfg = dict(config_dict or {})
        
        # Handle architecture configuration
        arch = cfg.get('architecture')
        if isinstance(arch, dict):
            cfg['architecture'] = ArchitectureConfig.from_dict(arch)
        
        # Handle attention_custom configuration
        attention_custom = cfg.get('attention_custom')
        if isinstance(attention_custom, dict):
            cfg['attention_custom'] = AttentionConfig.from_dict(attention_custom)
        
        return cls(**cfg)
    
    def validate(self) -> None:
        """Validate model configuration."""
        super().validate()
        
        # Validate architecture
        self.architecture.validate()
        
        # Validate dimensions
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        
        # Validate attention configuration
        # GQA is determined by n_kv_heads < n_heads, not by attention_type
        if self.n_kv_heads < self.n_heads and self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        
        # Validate custom attention configuration if preset is 'custom'
        if self.attention_preset == "custom":
            self.attention_custom.validate()
            if self.attention_custom.attention_type == "mla":
                if self.attention_custom.mla_rq <= 0:
                    raise ValueError("mla_rq must be positive")
                if self.attention_custom.mla_rkv <= 0:
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
class OptimizerGroupConfig(BaseConfig):
    """Configuration for a single optimizer parameter group."""
    
    # Learning rate for this group
    lr: float = 1e-4
    
    # Weight decay for this group
    weight_decay: float = 0.1
    
    # Adam/AdamW specific parameters
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    
    # SGD specific parameters (ignored for AdamW)
    momentum: float = 0.9
    dampening: float = 0.0
    nesterov: bool = False
    
    # Custom parameters for this group
    custom_params: Dict = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate optimizer group configuration."""
        super().validate()
        
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if len(self.betas) != 2:
            raise ValueError("betas must have exactly 2 elements")
        if not all(0 <= b <= 1 for b in self.betas):
            raise ValueError("betas must be between 0 and 1")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if not 0 <= self.momentum <= 1:
            raise ValueError("momentum must be between 0 and 1")
        if not 0 <= self.dampening <= 1:
            raise ValueError("dampening must be between 0 and 1")


@dataclass
class OptimizerConfig(BaseConfig):
    """Optimizer configuration with support for custom optimizers and grouped parameters."""
    
    # Optimizer type - can be any PyTorch optimizer class or string
    optimizer_type: str = "AdamW"  # Default to AdamW
    
    # Global parameters (used as defaults for all groups)
    lr: float = 1e-4
    weight_decay: float = 0.1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    momentum: float = 0.9
    dampening: float = 0.0
    nesterov: bool = False
    
    # Custom optimizer parameters (for advanced users)
    custom_params: Dict = field(default_factory=dict)
    
    # Grouped parameters - list of OptimizerGroupConfig
    # If empty, uses global parameters for all parameters
    # If provided, each group can have different settings
    groups: List[OptimizerGroupConfig] = field(default_factory=list)
    
    # Whether to use grouped parameters
    use_groups: bool = False
    
    def validate(self) -> None:
        """Validate optimizer configuration."""
        super().validate()
        
        # Validate global parameters
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if len(self.betas) != 2:
            raise ValueError("betas must have exactly 2 elements")
        if not all(0 <= b <= 1 for b in self.betas):
            raise ValueError("betas must be between 0 and 1")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if not 0 <= self.momentum <= 1:
            raise ValueError("momentum must be between 0 and 1")
        if not 0 <= self.dampening <= 1:
            raise ValueError("dampening must be between 0 and 1")
        
        # Validate groups if using grouped parameters
        if self.use_groups:
            if not self.groups:
                raise ValueError("use_groups is True but no groups provided")
            for i, group in enumerate(self.groups):
                try:
                    group.validate()
                except ValueError as e:
                    raise ValueError(f"Group {i} validation failed: {e}")
    
    def get_optimizer_kwargs(self) -> Dict:
        """Get optimizer-specific keyword arguments for global parameters."""
        if self.optimizer_type.lower() in ["adam", "adamw"]:
            return {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "betas": self.betas,
                "eps": self.eps,
                **self.custom_params
            }
        elif self.optimizer_type.lower() == "sgd":
            return {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "momentum": self.momentum,
                "dampening": self.dampening,
                "nesterov": self.nesterov,
                **self.custom_params
            }
        else:
            # For custom optimizers, use all parameters
            return {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                **self.custom_params
            }
    
    def get_group_kwargs(self, group_index: int) -> Dict:
        """Get optimizer-specific keyword arguments for a specific group."""
        if not self.use_groups or group_index >= len(self.groups):
            return self.get_optimizer_kwargs()
        
        group = self.groups[group_index]
        if self.optimizer_type.lower() in ["adam", "adamw"]:
            return {
                "lr": group.lr,
                "weight_decay": group.weight_decay,
                "betas": group.betas,
                "eps": group.eps,
                **group.custom_params
            }
        elif self.optimizer_type.lower() == "sgd":
            return {
                "lr": group.lr,
                "weight_decay": group.weight_decay,
                "momentum": group.momentum,
                "dampening": group.dampening,
                "nesterov": group.nesterov,
                **group.custom_params
            }
        else:
            # For custom optimizers, use all parameters
            return {
                "lr": group.lr,
                "weight_decay": group.weight_decay,
                **group.custom_params
            }
    
    def add_group(self, lr: Optional[float] = None, weight_decay: Optional[float] = None, 
                  betas: Optional[List[float]] = None, eps: Optional[float] = None,
                  momentum: Optional[float] = None, dampening: Optional[float] = None,
                  nesterov: Optional[bool] = None, custom_params: Optional[Dict] = None) -> None:
        """Add a new parameter group with optional overrides."""
        group = OptimizerGroupConfig(
            lr=lr if lr is not None else self.lr,
            weight_decay=weight_decay if weight_decay is not None else self.weight_decay,
            betas=betas if betas is not None else self.betas.copy(),
            eps=eps if eps is not None else self.eps,
            momentum=momentum if momentum is not None else self.momentum,
            dampening=dampening if dampening is not None else self.dampening,
            nesterov=nesterov if nesterov is not None else self.nesterov,
            custom_params=custom_params if custom_params is not None else {}
        )
        self.groups.append(group)
        self.use_groups = True


def create_optimizer(model_parameters, optimizer_config: OptimizerConfig, 
                    parameter_groups: Optional[List[Dict]] = None):
    """
    Create an optimizer from configuration with support for grouped parameters.
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_config: Optimizer configuration
        parameter_groups: Optional list of parameter groups (for grouped optimization)
        
    Returns:
        Configured optimizer
    """
    import torch
    
    # If using grouped parameters and parameter_groups is provided
    if optimizer_config.use_groups and parameter_groups is not None:
        # Create parameter groups with different optimizer settings
        grouped_params = []
        for i, (group_params, group_config) in enumerate(zip(parameter_groups, optimizer_config.groups)):
            group_kwargs = optimizer_config.get_group_kwargs(i)
            grouped_params.append({
                'params': group_params,
                **group_kwargs
            })
        
        # Create optimizer with grouped parameters
        if optimizer_config.optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(grouped_params)
        elif optimizer_config.optimizer_type.lower() == "adam":
            return torch.optim.Adam(grouped_params)
        elif optimizer_config.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(grouped_params)
        else:
            # For custom optimizers, try to import and use
            try:
                optimizer_class = getattr(torch.optim, optimizer_config.optimizer_type)
                return optimizer_class(grouped_params)
            except AttributeError:
                raise ValueError(f"Unknown optimizer type: {optimizer_config.optimizer_type}")
    
    else:
        # Use global parameters for all parameters
        optimizer_kwargs = optimizer_config.get_optimizer_kwargs()
        
        if optimizer_config.optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(model_parameters, **optimizer_kwargs)
        elif optimizer_config.optimizer_type.lower() == "adam":
            return torch.optim.Adam(model_parameters, **optimizer_kwargs)
        elif optimizer_config.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(model_parameters, **optimizer_kwargs)
        else:
            # For custom optimizers, try to import and use
            try:
                optimizer_class = getattr(torch.optim, optimizer_config.optimizer_type)
                return optimizer_class(model_parameters, **optimizer_kwargs)
            except AttributeError:
                raise ValueError(f"Unknown optimizer type: {optimizer_config.optimizer_type}")


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration."""
    
    # Training parameters
    batch_size: int = 4
    seq_len: int = 512
    max_steps: int = 200
    
    # Optimizer configuration
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    # Device and memory
    device: str = "cuda"
    gradient_checkpointing: bool = False
    
    # Caching configuration
    cache_dir: str = "./cache"
    cache_mode: str = "stack"  # "stack" or "rack"
    
    # Saving configuration
    save_stacks: bool = True      # Always save individual stacks (default enabled)
    save_rack: bool = False       # Optionally save entire rack (default disabled)
    
    # Mask-diffusion training
    min_mask_fraction: float = 0.15
    max_mask_fraction: float = 0.90
    mask_schedule_type: str = "linear"  # "linear", "exponential", "cosine"
    mask_token_id: Optional[int] = None  # Will be set from tokenizer
    epochs_per_stack: int = 1
    
    # Fine-tuning
    joint_tuning_steps: int = 50
    fine_tune_mode: FineTuneMode = "clm"
    
    # Training strategy: HOW to train
    strategy: str = "progressive"  # "progressive" | "end_to_end"
    # progressive: Build and train stacks one by one
    # end_to_end: Train the entire model at once
    
    # End-to-end training scope: WHAT to train (only used when strategy="end_to_end")
    end_to_end_scope: str = "stackwise"  # "stackwise" | "rackwise"
    # stackwise: Train each stack independently
    # rackwise: Train the entire rack together
    
    # Run identification and organization
    run_id: str = "default_run"
    
    # QLoRA and quantization settings
    qlora: 'QLoRAConfig' = field(default_factory=lambda: QLoRAConfig())
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

    # Progressive training sub-configuration (typed)
    progressive: 'ProgressiveConfig' = field(default_factory=lambda: ProgressiveConfig())

    def validate(self) -> None:
        """Validate training configuration."""
        super().validate()
        
        # Validate optimizer configuration
        self.optimizer.validate()
        
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
        
        # Validate QLoRA and quantization settings
        self.qlora.validate()
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
    
    def set_mask_token_id(self, mask_token_id: int) -> None:
        """Set mask token ID from tokenizer."""
        if mask_token_id < 0:
            raise ValueError("mask_token_id must be non-negative")
        self.mask_token_id = mask_token_id

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Construct TrainingConfig from dict and convert progressive sub-dict to ProgressiveConfig."""
        # Copy to avoid mutating input
        cfg = dict(config_dict or {})
        
        # Handle optimizer configuration
        optimizer = cfg.get('optimizer')
        if isinstance(optimizer, dict):
            # Handle grouped optimizer configuration
            if 'groups' in optimizer and isinstance(optimizer['groups'], list):
                groups = []
                for group_dict in optimizer['groups']:
                    if isinstance(group_dict, dict):
                        groups.append(OptimizerGroupConfig.from_dict(group_dict))
                optimizer['groups'] = groups
            cfg['optimizer'] = OptimizerConfig.from_dict(optimizer)
        
        # Handle QLoRA configuration
        qlora = cfg.get('qlora')
        if isinstance(qlora, dict):
            cfg['qlora'] = QLoRAConfig.from_dict(qlora)
        
        # Handle progressive configuration
        prog = cfg.get('progressive')
        if isinstance(prog, dict):
            cfg['progressive'] = ProgressiveConfig.from_dict(prog)
        

        obj = cls(**cfg)
        return obj



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
    
    def set_mask_token_id(self, mask_token_id: int) -> None:
        """Set mask token ID from tokenizer."""
        self.training.set_mask_token_id(mask_token_id)
    
    def generate_run_name(self) -> str:
        """Generate a unique run name based on architecture configuration."""
        n_stacks = self.model.architecture.n_stacks
        blocks_per_stack = self.model.architecture.blocks_per_stack
        total_blocks = n_stacks * blocks_per_stack
        
        strategy = self.training.strategy
        training_scope = self.training.training_scope if strategy == "end_to_end" else "progressive"
        
        run_name = f"{strategy}_{training_scope}_s{n_stacks}_b{blocks_per_stack}_t{total_blocks}"
        
        # Add progressive-specific info if applicable
        if strategy == "progressive":
            target_stacks = getattr(self.training.progressive, 'target_stacks', n_stacks)
            trunk_strategy = getattr(self.training.progressive, 'trunk_strategy', 'frozen')
            run_name += f"_target{target_stacks}_{trunk_strategy}"
        
        return run_name


@dataclass
class QLoRAConfig(BaseConfig):
    """
    Configuration for QLoRA (Quantized Low-Rank Adaptation) settings.
    
    QLoRA enables efficient fine-tuning by using low-rank adaptation matrices
    instead of updating the full model parameters. This significantly reduces
    memory usage and training time while maintaining performance.
    
    Attributes:
        enabled: Whether to enable QLoRA adapters
        rank: Rank of the low-rank adaptation matrices (higher = more parameters)
        alpha: Scaling factor for the adaptation (typically 2x rank)
        dropout: Dropout rate for QLoRA adapters
        lr: Learning rate for QLoRA parameters
        
        progressive_enabled: Whether to use progressive QLoRA strategies
        progressive_rank: Rank for progressive QLoRA (typically lower than main rank)
        progressive_alpha: Alpha for progressive QLoRA
        
        strategy: QLoRA strategy type:
            - 'simplified': Use same QLoRA config for all stacks
            - 'progressive': Gradually increase/decrease QLoRA parameters across stacks
            - 'variable': Use custom QLoRA configs per stack (defined in configs dict)
        
        rank_pattern: How rank changes across stacks:
            - 'constant': Same rank for all stacks
            - 'increasing': Rank increases with stack depth (early stacks: low rank, later: high rank)
            - 'decreasing': Rank decreases with stack depth (early stacks: high rank, later: low rank)
        
        alpha_pattern: How alpha changes across stacks:
            - 'constant': Same alpha for all stacks
            - 'increasing': Alpha increases with stack depth
            - 'decreasing': Alpha decreases with stack depth
        
        configs: Per-stack custom QLoRA configurations (used when strategy='variable')
                 Format: {stack_idx: {'rank': int, 'alpha': int, 'dropout': float}}
    
    Examples:
        # Simplified QLoRA (same config for all stacks)
        qlora = QLoRAConfig(
            enabled=True,
            rank=16,
            alpha=32,
            strategy='simplified'
        )
        
        # Progressive QLoRA (increasing rank with depth)
        qlora = QLoRAConfig(
            enabled=True,
            rank=8,
            alpha=16,
            strategy='progressive',
            rank_pattern='increasing',
            alpha_pattern='increasing'
        )
        
        # Variable QLoRA (custom config per stack)
        qlora = QLoRAConfig(
            enabled=True,
            strategy='variable',
            configs={
                0: {'rank': 8, 'alpha': 16, 'dropout': 0.1},   # Early stack: low rank
                1: {'rank': 16, 'alpha': 32, 'dropout': 0.1},  # Middle stack: medium rank
                2: {'rank': 32, 'alpha': 64, 'dropout': 0.05}  # Later stack: high rank
            }
        )
    """
    enabled: bool = True
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    lr: float = 1e-5
    
    # Progressive QLoRA settings
    progressive_enabled: bool = False
    progressive_rank: int = 8
    progressive_alpha: int = 16
    strategy: str = 'simplified'  # 'simplified' | 'progressive' | 'variable'
    rank_pattern: str = 'constant'  # 'constant' | 'increasing' | 'decreasing'
    alpha_pattern: str = 'constant'  # 'constant' | 'increasing' | 'decreasing'
    configs: Dict[int, Dict] = field(default_factory=dict)  # Per-stack QLoRA configs
    
    # Mixed precision training (auto-enabled with QLoRA)
    mixed_precision: bool = True  # Frozen trunk in fp16 + adapters in fp32
    
    def validate(self) -> None:
        """Validate QLoRA configuration."""
        super().validate()
        
        if self.rank <= 0:
            raise ValueError("qlora.rank must be positive")
        if self.alpha <= 0:
            raise ValueError("qlora.alpha must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("qlora.dropout must be between 0 and 1")
        if self.lr <= 0:
            raise ValueError("qlora.lr must be positive")
        if self.progressive_rank <= 0:
            raise ValueError("qlora.progressive_rank must be positive")
        if self.progressive_alpha <= 0:
            raise ValueError("qlora.progressive_alpha must be positive")
        if self.strategy not in ['simplified', 'progressive', 'variable']:
            raise ValueError("qlora.strategy must be one of: simplified, progressive, variable")
        if self.rank_pattern not in ['constant', 'increasing', 'decreasing']:
            raise ValueError("qlora.rank_pattern must be one of: constant, increasing, decreasing")
        if self.alpha_pattern not in ['constant', 'increasing', 'decreasing']:
            raise ValueError("qlora.alpha_pattern must be one of: constant, increasing, decreasing")
    
    def get_stack_config(self, stack_idx: int, total_stacks: int) -> Dict[str, Any]:
        """
        Get QLoRA configuration for a specific stack based on strategy and patterns.
        
        Args:
            stack_idx: Index of the stack (0-based)
            total_stacks: Total number of stacks
            
        Returns:
            Dictionary with QLoRA configuration for the stack
        """
        if self.strategy == 'simplified':
            return {
                'rank': self.rank,
                'alpha': self.alpha,
                'dropout': self.dropout,
                'lr': self.lr
            }
        
        elif self.strategy == 'progressive':
            # Calculate rank based on pattern
            if self.rank_pattern == 'constant':
                rank = self.rank
            elif self.rank_pattern == 'increasing':
                # Linear increase from base_rank to rank
                progress = stack_idx / max(1, total_stacks - 1)
                rank = int(self.progressive_rank + progress * (self.rank - self.progressive_rank))
            elif self.rank_pattern == 'decreasing':
                # Linear decrease from rank to base_rank
                progress = stack_idx / max(1, total_stacks - 1)
                rank = int(self.rank - progress * (self.rank - self.progressive_rank))
            else:
                rank = self.rank
            
            # Calculate alpha based on pattern
            if self.alpha_pattern == 'constant':
                alpha = self.alpha
            elif self.alpha_pattern == 'increasing':
                progress = stack_idx / max(1, total_stacks - 1)
                alpha = int(self.progressive_alpha + progress * (self.alpha - self.progressive_alpha))
            elif self.alpha_pattern == 'decreasing':
                progress = stack_idx / max(1, total_stacks - 1)
                alpha = int(self.alpha - progress * (self.alpha - self.progressive_alpha))
            else:
                alpha = self.alpha
            
            return {
                'rank': rank,
                'alpha': alpha,
                'dropout': self.dropout,
                'lr': self.lr
            }
        
        elif self.strategy == 'variable':
            # Use custom config for this stack
            if stack_idx in self.configs:
                config = self.configs[stack_idx].copy()
                # Fill in defaults for missing fields
                config.setdefault('rank', self.rank)
                config.setdefault('alpha', self.alpha)
                config.setdefault('dropout', self.dropout)
                config.setdefault('lr', self.lr)
                return config
            else:
                # Fallback to default config
                return {
                    'rank': self.rank,
                    'alpha': self.alpha,
                    'dropout': self.dropout,
                    'lr': self.lr
                }
        
        else:
            raise ValueError(f"Unknown QLoRA strategy: {self.strategy}")


@dataclass
class ProgressiveConfig(BaseConfig):
    """Configuration for progressive training features."""
    enabled: bool = True
    max_stacks: int = 12
    target_stacks: int = 8          # Number of stacks to build progressively
    time_interpretation: str = 'depth'  # 'depth' or 'input'
    trunk_strategy: str = 'frozen'  # 'frozen' or 'qlora'
    new_stack_precision: str = 'full'
    cache_activations: bool = True
    training_objective: str = 'mlm'

    @classmethod
    def from_dict(cls, d: dict) -> 'ProgressiveConfig':
        return cls(**(d or {}))
    
    def validate(self) -> None:
        """Validate progressive-specific configuration."""
        super().validate()

        if not isinstance(self.enabled, bool):
            raise ValueError("progressive.enabled must be a boolean")
        if self.max_stacks <= 0:
            raise ValueError("progressive.max_stacks must be positive")
        if self.target_stacks <= 0:
            raise ValueError("progressive.target_stacks must be positive")
        if self.target_stacks > self.max_stacks:
            raise ValueError("progressive.target_stacks cannot exceed max_stacks")
        if self.trunk_strategy not in ['frozen', 'qlora']:
            raise ValueError("progressive.trunk_strategy must be 'frozen' or 'qlora'")
        if self.training_objective not in ['mlm', 'clm', 'custom']:
            raise ValueError("progressive.training_objective must be one of: mlm, clm, custom")
