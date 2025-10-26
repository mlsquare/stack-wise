"""
Progressive Rack Builder for progressive training.

This module provides functionality to build Racks progressively by adding
Stacks one by one, with support for different building modes (append/prepend)
and precision management.
"""

import logging
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

from ..model.architecture import Block, Stack, Rack, create_stack, create_stack_from_config
from ..config.base import StackWiseConfig, ProgressiveConfig


class LoRAAdapter(nn.Module):
    """Simple LoRA adapter: down-project -> nonlinearity(opt) -> up-project with scaling."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: Optional[int] = None):
        super().__init__()
        if alpha is None:
            alpha = max(1, rank * 2)
        self.rank = max(1, int(rank))
        self.alpha = int(alpha)
        self.scaling = float(self.alpha) / float(self.rank)

        # Down and up projections
        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=False)

        # Initialize small
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, ..., in_features) -> apply linear elementwise
        # LoRA typically applies to 2D [batch*seq, in_features], but we accept any shape
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])
        down = self.down(flat)
        up = self.up(down)
        out = up.reshape(*orig_shape[:-1], up.shape[-1])
        return out * self.scaling

logger = logging.getLogger(__name__)


class ProgressiveRackBuilder:
    """
    Builder for progressive Rack construction.
    
    Supports:
    - Progressive building (append/prepend modes)
    - Precision management (full, half, bfloat16, QLoRA)
    - Trunk management (frozen, QLoRA fine-tuning)
    - Activation caching
    """
    
    def __init__(self, 
                 config: StackWiseConfig,
                 building_mode: str = "append",
                 default_precision: str = "full"):
        """
        Initialize progressive rack builder (fully config-driven).
        
        Args:
            config: Configuration object containing all architectural parameters
            building_mode: Building mode ("append" or "prepend")
            default_precision: Default precision for new stacks
        """
        self.config = config
        self.building_mode = building_mode
        self.default_precision = default_precision
        
        # Read all parameters from config
        self.vocab_size = config.model.vocab_size
        self.d_model = config.model.d_model
        self.d_ff = config.model.d_ff
        self.n_heads = config.model.n_heads
        # Normalize progressive config early
        prog_cfg = None
        try:
            prog_cfg = self._get_progressive_config()
        except Exception:
            prog_cfg = ProgressiveConfig()
        self.max_stacks = prog_cfg.max_stacks
        
        # Initialize components
        self.embeddings = None
        self.stacks = []
        self.lm_head = None
        self.qlora_adapters = {}  # Per-stack QLoRA adapters
        self.precision_settings = {}  # Per-stack precision settings
        
        # Building state
        self.current_stacks = 0
        self.built_rack = None
    # ensure max_stacks is set (already assigned above)
        
        logger.info(f"Initialized ProgressiveRackBuilder: {building_mode} mode, max_stacks={self.max_stacks}")
    
    def initialize_embeddings_and_head(self):
        """Initialize embeddings and language model head"""
        if self.embeddings is None:
            # Create embeddings (using LexicalKernelManager)
            from ..model.layers import LexicalKernelManager
            # Pull tokenizer/embedding config from global config
            tokenizer_cfg = getattr(self.config.model, 'tokenizer_embedding', {}) or {}
            family = tokenizer_cfg.get('family', 'gpt2')
            embedding_option = tokenizer_cfg.get('embedding_option', 'embed_tokens')
            freeze_embeddings = tokenizer_cfg.get('freeze_embeddings', True)
            adapter_hidden_dim = tokenizer_cfg.get('adapter_hidden_dim', None)

            # Initialize LexicalKernelManager using the expected signature
            self.embeddings = LexicalKernelManager(
                family=family,
                embedding_option=embedding_option,
                freeze_embeddings=freeze_embeddings,
                target_model_dim=self.d_model,
                adapter_hidden_dim=adapter_hidden_dim
            )
        
        if self.lm_head is None:
            # Create language model head
            self.lm_head = nn.Linear(self.d_model, self.vocab_size)
        
        logger.info("Initialized embeddings and language model head")

    def _get_progressive_config(self) -> ProgressiveConfig:
        """Return a ProgressiveConfig instance (accept dict for backwards compatibility)."""
        prog = getattr(self.config.training, 'progressive', None)
        if prog is None:
            return ProgressiveConfig()
        if isinstance(prog, dict):
            return ProgressiveConfig.from_dict(prog)
        if isinstance(prog, ProgressiveConfig):
            return prog
        # Fallback: try to construct from attributes
        try:
            return ProgressiveConfig(**vars(prog))
        except Exception:
            return ProgressiveConfig()
    
    def append_stack(self, 
                     stack: Optional[Stack] = None,
                     n_blocks: int = 4,
                     precision: str = None,
                     qlora_rank: int = 16) -> Stack:
        """
        Append a new stack to the rack.
        
        Args:
            stack: Existing stack to add (if None, creates new stack)
            n_blocks: Number of blocks in the stack
            precision: Precision for the stack
            qlora_rank: QLoRA rank for LoRA adapters
            
        Returns:
            Added stack
        """
        if self.current_stacks >= self.max_stacks:
            raise ValueError(f"Maximum stacks ({self.max_stacks}) reached")
        
        if stack is None:
            # Create new stack using config-driven approach
            stack = create_stack_from_config(
                stack_id=self.current_stacks,
                n_blocks=n_blocks,
                config=self.config
            )
        
        # Set precision
        precision = precision or self.default_precision
        self.precision_settings[self.current_stacks] = precision
        
        # Always add LoRA adapters to each stack
        prog = self._get_progressive_config()
        if getattr(prog, 'qlora_enabled', False):
            # Get LoRA configuration for this specific stack
            lora_config = self._get_qlora_config_for_stack(self.current_stacks)
            if lora_config:
                self._add_qlora_to_stack(stack, self.current_stacks, lora_config['rank'], lora_config.get('alpha'))
                logger.debug(f"Added LoRA adapters to stack {self.current_stacks}: {lora_config}")
            
            # Add additional QLoRA to entire trunk when new stack is added
            if self.current_stacks > 0 and getattr(prog, 'progressive_qlora', False):  # Only if there are existing stacks and progressive_qlora is enabled
                self._add_qlora_to_trunk()
        
        # Add stack
        self.stacks.append(stack)
        self.current_stacks += 1
        
        logger.info(f"Appended stack {self.current_stacks - 1} with precision {precision}")
        return stack
    
    def prepend_stack(self, 
                      stack: Optional[Stack] = None,
                      n_blocks: int = 4,
                      precision: str = None,
                      qlora_rank: int = 16) -> Stack:
        """
        Prepend a new stack to the rack.
        
        Args:
            stack: Existing stack to add (if None, creates new stack)
            n_blocks: Number of blocks in the stack
            precision: Precision for the stack
            qlora_rank: QLoRA rank for LoRA adapters
            
        Returns:
            Added stack
        """
        if self.current_stacks >= self.max_stacks:
            raise ValueError(f"Maximum stacks ({self.max_stacks}) reached")
        
        if stack is None:
            # Create new stack using config-driven approach
            stack = create_stack_from_config(
                stack_id=self.current_stacks,
                n_blocks=n_blocks,
                config=self.config
            )
        
        # Set precision
        precision = precision or self.default_precision
        self.precision_settings[self.current_stacks] = precision
        
        # Always add LoRA adapters to each stack
        prog = self._get_progressive_config()
        if getattr(prog, 'qlora_enabled', False):
            # Get LoRA configuration for this specific stack
            lora_config = self._get_qlora_config_for_stack(self.current_stacks)
            if lora_config:
                self._add_qlora_to_stack(stack, self.current_stacks, lora_config['rank'], lora_config.get('alpha'))
                logger.debug(f"Added LoRA adapters to stack {self.current_stacks}: {lora_config}")
            
            # Add additional QLoRA to entire trunk when new stack is added
            if self.current_stacks > 0 and getattr(prog, 'progressive_qlora', False):  # Only if there are existing stacks and progressive_qlora is enabled
                self._add_qlora_to_trunk()
        
        # Prepend stack (insert at beginning)
        self.stacks.insert(0, stack)
        self.current_stacks += 1
        
        # Update stack IDs
        for i, stack in enumerate(self.stacks):
            stack.stack_id = i
        
        logger.info(f"Prepended stack {self.current_stacks - 1} with precision {precision}")
        return stack
    
    def add_qlora_to_trunk(self, stack_indices: List[int], rank: int = 16):
        """
        Add QLoRA adapters to specified stacks (trunk).
        
        Args:
            stack_indices: List of stack indices to add QLoRA to
            rank: QLoRA rank
        """
        for stack_idx in stack_indices:
            if 0 <= stack_idx < len(self.stacks):
                stack = self.stacks[stack_idx]
                self._add_qlora_adapters(stack, stack_idx, rank)
                self.precision_settings[stack_idx] = "qlora"
                logger.info(f"Added QLoRA adapters to stack {stack_idx}")
    
    def add_qlora_to_all_stacks(self, rank: int = 16, alpha: int = 32):
        """
        Add QLoRA adapters to all stacks (simplified approach).
        
        This is the recommended approach:
        - All stacks get LoRA adapters
        - When trunk is frozen: all params (including LoRA) are frozen
        - When QLoRA trunk: only LoRA adapters are updated
        
        Args:
            rank: LoRA rank
            alpha: LoRA alpha parameter
        """
        for stack_idx, stack in enumerate(self.stacks):
            self._add_qlora_adapters(stack, stack_idx, rank, alpha)
            self.precision_settings[stack_idx] = "qlora"
            logger.info(f"Added QLoRA adapters to stack {stack_idx} (rank={rank}, alpha={alpha})")
        
        logger.info(f"Added QLoRA adapters to all {len(self.stacks)} stacks")
    
    def freeze_trunk(self, stack_indices: List[int]):
        """
        Freeze specified stacks (trunk).
        
        Args:
            stack_indices: List of stack indices to freeze
        """
        for stack_idx in stack_indices:
            if 0 <= stack_idx < len(self.stacks):
                stack = self.stacks[stack_idx]
                self._freeze_stack(stack)
                logger.info(f"Frozen stack {stack_idx}")
    
    def _add_qlora_to_stack(self, stack: Stack, stack_idx: int, rank: int, alpha: int = None):
        """Add QLoRA adapters to every nn.Linear in the given stack.

        This attaches a LoRAAdapter module to each Linear and registers a forward
        hook that adds the adapter output to the original linear output. The
        adapter modules are registered as child modules of the linear so their
        parameters are part of the model parameters.
        """
        if alpha is None:
            alpha = max(1, rank * 2)

        adapters = {}

        for i, module in enumerate(stack.modules()):
            if isinstance(module, nn.Linear):
                # Create adapter and attach as submodule so parameters are found
                adapter_name = f"lora_adapter_{stack_idx}_{i}"
                adapter = LoRAAdapter(module.in_features, module.out_features, rank=rank, alpha=alpha)
                # Attach to the linear module
                module.add_module(adapter_name, adapter)

                # Register forward hook to add adapter contribution
                def _hook(mod, inp, out, adapter=adapter):
                    try:
                        return out + adapter(inp[0])
                    except Exception:
                        # On any unexpected shape issues, fall back to original output
                        return out

                handle = module.register_forward_hook(_hook)

                key = str(id(module))
                adapters[key] = {
                    'adapter': adapter,
                    'handle': handle,
                    'module': module,
                    'name': adapter_name
                }

        # Store metadata
        self.qlora_adapters[stack_idx] = {
            'rank': rank,
            'alpha': alpha,
            'adapters': adapters,
            'enabled': True
        }

        logger.info(f"Added QLoRA adapters to stack {stack_idx}: rank={rank}, alpha={alpha}, adapters={len(adapters)}")
    
    def _add_qlora_to_trunk(self):
        """
        Add QLoRA adapters to the entire trunk when a new stack is added.
        
        NOTE: This is currently a placeholder implementation.
        The actual QLoRA adapters are not added to the trunk parameters.
        
        This implements the dual-LoRA approach:
        1. Each stack gets its own LoRA adapters (added during stack creation)
        2. QLoRA adapters are added to the entire trunk when new stacks are added
        """
        progressive_config = self._get_progressive_config()

        # Get progressive QLoRA configuration
        progressive_qlora_enabled = getattr(progressive_config, 'progressive_qlora', True)
        if not progressive_qlora_enabled:
            return

        # Get progressive QLoRA parameters
        progressive_rank = getattr(progressive_config, 'progressive_qlora_rank', 8)  # Smaller rank for progressive
        progressive_alpha = getattr(progressive_config, 'progressive_qlora_alpha', 16)

        # Add progressive QLoRA adapters to all existing stacks (trunk)
        for stack_idx in range(self.current_stacks):
            # Add progressive QLoRA adapters (additional layer) by reusing _add_qlora_to_stack
            stack = self.stacks[stack_idx]
            # Use a distinct namespace by offsetting rank slightly or simply add another set of adapters
            self._add_qlora_to_stack(stack, stack_idx, progressive_rank, progressive_alpha)
            # Mark these as progressive-type adapters in metadata
            self.qlora_adapters[f"progressive_qlora_{stack_idx}"] = {
                'rank': progressive_rank,
                'alpha': progressive_alpha,
                'adapters': self.qlora_adapters.get(stack_idx, {}).get('adapters', {}),
                'enabled': True,
                'type': 'progressive_qlora'
            }

            logger.debug(f"Added progressive QLoRA adapters to trunk stack {stack_idx}: rank={progressive_rank}, alpha={progressive_alpha}")

        logger.info(f"Added QLoRA adapters to {self.current_stacks} existing trunk stacks")
    
    def _get_qlora_config_for_stack(self, stack_idx: int) -> Optional[Dict]:
        """
        Get QLoRA configuration for a specific stack.
        
        This allows for progressive QLoRA configuration where each stack
        can have different QLoRA parameters.
        
        Args:
            stack_idx: Index of the stack
            
        Returns:
            QLoRA configuration dict or None if no QLoRA for this stack
        """
        progressive_config = getattr(self.config.training, 'progressive', {})
        
        # Check if QLoRA is enabled
        if not getattr(progressive_config, 'qlora_enabled', False):
            return None
        
        # Get QLoRA strategy
        qlora_strategy = getattr(progressive_config, 'qlora_strategy', 'simplified')
        
        if qlora_strategy == 'simplified':
            # All stacks get the same QLoRA configuration
            return {
                'rank': getattr(progressive_config, 'qlora_rank', 16),
                'alpha': getattr(progressive_config, 'qlora_alpha', 32)
            }
        
        elif qlora_strategy == 'progressive':
            # Progressive QLoRA: different configurations per stack
            return self._get_progressive_qlora_config(stack_idx)
        
        elif qlora_strategy == 'variable':
            # Variable QLoRA: custom configuration per stack
            return self._get_variable_qlora_config(stack_idx)
        
        else:
            # Default: no QLoRA for this stack
            return None
    
    def _get_progressive_qlora_config(self, stack_idx: int) -> Dict:
        """
        Get progressive QLoRA configuration.
        
        Progressive strategy: QLoRA parameters change as stacks are added.
        Examples:
        - Increasing rank: [8, 16, 32, 64]
        - Decreasing rank: [64, 32, 16, 8]
        - Variable alpha: [16, 32, 64, 128]
        """
        progressive_config = getattr(self.config.training, 'progressive', {})
        
        # Get base parameters
        base_rank = getattr(progressive_config, 'qlora_rank', 16)
        base_alpha = getattr(progressive_config, 'qlora_alpha', 32)
        
        # Progressive patterns
        rank_pattern = getattr(progressive_config, 'qlora_rank_pattern', 'constant')
        alpha_pattern = getattr(progressive_config, 'qlora_alpha_pattern', 'constant')
        
        # Calculate rank based on pattern
        if rank_pattern == 'increasing':
            rank = base_rank * (2 ** stack_idx)  # 16, 32, 64, 128...
        elif rank_pattern == 'decreasing':
            rank = base_rank // (2 ** stack_idx)  # 16, 8, 4, 2...
        elif rank_pattern == 'linear':
            rank = base_rank + (stack_idx * 8)  # 16, 24, 32, 40...
        else:  # constant
            rank = base_rank
        
        # Calculate alpha based on pattern
        if alpha_pattern == 'increasing':
            alpha = base_alpha * (2 ** stack_idx)
        elif alpha_pattern == 'decreasing':
            alpha = base_alpha // (2 ** stack_idx)
        elif alpha_pattern == 'linear':
            alpha = base_alpha + (stack_idx * 16)
        else:  # constant
            alpha = base_alpha
        
        return {
            'rank': max(1, rank),  # Ensure rank is at least 1
            'alpha': max(1, alpha)  # Ensure alpha is at least 1
        }
    
    def _get_variable_qlora_config(self, stack_idx: int) -> Optional[Dict]:
        """
        Get variable QLoRA configuration.
        
        Variable strategy: Custom configuration per stack using a lookup table.
        """
        progressive_config = getattr(self.config.training, 'progressive', {})
        
        # Get custom QLoRA configurations per stack
        qlora_configs = getattr(progressive_config, 'qlora_configs', {})
        
        # Look up configuration for this stack
        if stack_idx in qlora_configs:
            return qlora_configs[stack_idx]
        
        # Default configuration if not specified
        return {
            'rank': getattr(progressive_config, 'qlora_rank', 16),
            'alpha': getattr(progressive_config, 'qlora_alpha', 32)
        }
    
    def _freeze_stack(self, stack: Stack):
        """Freeze a stack (set requires_grad=False)"""
        for param in stack.parameters():
            param.requires_grad = False
    
    def freeze_all_but_qlora(self, stack_indices: List[int]):
        """
        Freeze all parameters except QLoRA adapters for specified stacks.
        
        This implements the dual-LoRA strategy:
        - All original parameters are frozen
        - Only LoRA adapter parameters are trainable (both stack LoRA + trunk QLoRA)
        
        Args:
            stack_indices: List of stack indices to apply QLoRA freezing to
        """
        for stack_idx in stack_indices:
            if 0 <= stack_idx < len(self.stacks):
                stack = self.stacks[stack_idx]
                self._freeze_single_stack_but_qlora(stack, stack_idx)
                logger.info(f"Applied QLoRA freezing to stack {stack_idx}")
    
    def _freeze_single_stack_but_qlora(self, stack: Stack, stack_idx: int):
        """Freeze all parameters except QLoRA adapters for a single stack"""
        # Freeze all original parameters
        for param in stack.parameters():
            param.requires_grad = False

        # Enable stack LoRA adapter parameters if present
        meta = self.qlora_adapters.get(stack_idx)
        if meta and meta.get('enabled'):
            for key, info in meta.get('adapters', {}).items():
                adapter = info.get('adapter')
                if adapter is not None:
                    for p in adapter.parameters():
                        p.requires_grad = True

        # Also enable trunk/progressive adapters if present
        trunk_key = f"progressive_qlora_{stack_idx}"
        trunk_meta = self.qlora_adapters.get(trunk_key)
        if trunk_meta and trunk_meta.get('enabled'):
            for key, info in trunk_meta.get('adapters', {}).items():
                adapter = info.get('adapter')
                if adapter is not None:
                    for p in adapter.parameters():
                        p.requires_grad = True

        logger.debug(f"All LoRA adapters enabled for stack {stack_idx} (stack LoRA + trunk QLoRA)")
    
    def enable_qlora_training(self, stack_indices: List[int]):
        """
        Enable QLoRA training for specified stacks.
        
        This sets up the training configuration where:
        - Original parameters are frozen
        - Only QLoRA adapters are trainable
        
        Args:
            stack_indices: List of stack indices to enable QLoRA training for
        """
        for stack_idx in stack_indices:
            if 0 <= stack_idx < len(self.stacks):
                if stack_idx in self.qlora_adapters:
                    self.qlora_adapters[stack_idx]['enabled'] = True
                    logger.info(f"Enabled QLoRA training for stack {stack_idx}")
                else:
                    logger.warning(f"No QLoRA adapters found for stack {stack_idx}")
    
    def disable_qlora_training(self, stack_indices: List[int]):
        """
        Disable QLoRA training for specified stacks (freeze everything).
        
        Args:
            stack_indices: List of stack indices to disable QLoRA training for
        """
        for stack_idx in stack_indices:
            if 0 <= stack_idx < len(self.stacks):
                if stack_idx in self.qlora_adapters:
                    self.qlora_adapters[stack_idx]['enabled'] = False
                    # Freeze all parameters including QLoRA
                    self._freeze_stack(self.stacks[stack_idx])
                    logger.info(f"Disabled QLoRA training for stack {stack_idx}")
                else:
                    logger.warning(f"No QLoRA adapters found for stack {stack_idx}")
    
    def build_rack(self) -> Rack:
        """
        Build the complete rack.
        
        Returns:
            Complete Rack instance
        """
        if self.built_rack is not None:
            return self.built_rack
        
        # Ensure embeddings and head are initialized
        self.initialize_embeddings_and_head()
        
        # Create rack
        self.built_rack = Rack(
            embeddings=self.embeddings,
            stacks=self.stacks,
            lm_head=self.lm_head
        )
        
        logger.info(f"Built rack with {self.current_stacks} stacks")
        return self.built_rack
    
    def get_current_depth(self) -> int:
        """Get current number of stacks"""
        return self.current_stacks
    
    def get_stack_info(self, stack_idx: int) -> Dict[str, Any]:
        """Get information about a specific stack"""
        if stack_idx >= len(self.stacks):
            raise ValueError(f"Stack {stack_idx} does not exist")
        
        stack = self.stacks[stack_idx]
        precision = self.precision_settings.get(stack_idx, self.default_precision)
        qlora_info = self.qlora_adapters.get(stack_idx)
        
        return {
            'stack_idx': stack_idx,
            'n_blocks': len(stack.blocks),
            'precision': precision,
            'qlora_rank': qlora_info['rank'] if qlora_info else None,
            'frozen': not any(param.requires_grad for param in stack.parameters())
        }
    
    def get_rack_info(self) -> Dict[str, Any]:
        """Get information about the entire rack"""
        return {
            'current_stacks': self.current_stacks,
            'max_stacks': self.max_stacks,
            'building_mode': self.building_mode,
            'default_precision': self.default_precision,
            'stack_info': [self.get_stack_info(i) for i in range(self.current_stacks)]
        }
    
    def get_model(self) -> Rack:
        """Get the current model (Rack)"""
        return self.build_rack()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model"""
        rack = self.build_rack()
        total_params = sum(p.numel() for p in rack.parameters())
        trainable_params = sum(p.numel() for p in rack.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'num_stacks': len(self.stacks),
            'current_stacks': self.current_stacks,
            'max_stacks': self.max_stacks,
            'building_mode': self.building_mode,
            'precision_settings': self.precision_settings,
            'qlora_adapters': len(self.qlora_adapters)
        }
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get the complete model state for saving/loading"""
        rack = self.build_rack()
        return {
            'model_state_dict': rack.state_dict(),
            'stacks': [stack.state_dict() for stack in self.stacks],
            'current_stacks': self.current_stacks,
            'precision_settings': self.precision_settings,
            'qlora_adapters': self.qlora_adapters,
            'config': self.config.to_dict()
        }
    
    def load_model_state(self, state: Dict[str, Any]):
        """Load model state from saved state"""
        # Load rack builder state
        self.current_stacks = state['current_stacks']
        self.precision_settings = state['precision_settings']
        self.qlora_adapters = state['qlora_adapters']
        
        # Load stack states
        for i, stack_state in enumerate(state['stacks']):
            if i < len(self.stacks):
                self.stacks[i].load_state_dict(stack_state)
        
        # Load model state
        rack = self.build_rack()
        rack.load_state_dict(state['model_state_dict'])
        
        logger.info(f"Loaded model state with {self.current_stacks} stacks")
    
    def get_embeddings(self):
        """Get the embeddings layer"""
        rack = self.build_rack()
        return rack.get_embeddings()
    
    def get_lm_head(self):
        """Get the language model head"""
        rack = self.build_rack()
        return rack.get_lm_head()
    
    def get_stack(self, stack_idx: int) -> Optional[Stack]:
        """Get a specific stack by index"""
        if 0 <= stack_idx < len(self.stacks):
            return self.stacks[stack_idx]
        return None
    
    def get_all_stacks(self) -> List[Stack]:
        """Get all stacks"""
        return self.stacks.copy()
    
    def get_stack_info(self, stack_idx: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific stack"""
        stack = self.get_stack(stack_idx)
        if stack is None:
            return None
        
        total_params = sum(p.numel() for p in stack.parameters())
        trainable_params = sum(p.numel() for p in stack.parameters() if p.requires_grad)
        
        return {
            'stack_idx': stack_idx,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'precision': self.precision_settings.get(stack_idx, self.default_precision),
            'qlora_enabled': stack_idx in self.qlora_adapters,
            'num_blocks': len(stack.blocks)
        }
    
    def save_rack(self, path: str) -> str:
        """
        Save complete rack to file.
        
        Args:
            path: Path to save the rack
            
        Returns:
            Saved file path
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare rack data
        rack_data = {
            'timestamp': datetime.now().isoformat(),
            'current_stacks': self.current_stacks,
            'max_stacks': self.max_stacks,
            'building_mode': self.building_mode,
            'default_precision': self.default_precision,
            'precision_settings': self.precision_settings,
            'qlora_adapters': self.qlora_adapters,
            'config': self.config.to_dict()
        }
        
        # Save stack states
        rack_data['stacks'] = [stack.state_dict() for stack in self.stacks]
        
        # Save embeddings and lm_head if available
        if hasattr(self, 'embeddings') and self.embeddings is not None:
            rack_data['embeddings'] = self.embeddings.state_dict()
        
        if hasattr(self, 'lm_head') and self.lm_head is not None:
            rack_data['lm_head'] = self.lm_head.state_dict()
        
        # Save to file
        torch.save(rack_data, save_path)
        
        logger.info(f"Saved rack to: {save_path}")
        return str(save_path)
    
    def load_rack(self, path: str) -> bool:
        """
        Load complete rack from file.
        
        Args:
            path: Path to load the rack from
            
        Returns:
            True if successful, False otherwise
        """
        load_path = Path(path)
        
        if not load_path.exists():
            logger.warning(f"Rack file not found: {load_path}")
            return False
        
        try:
            rack_data = torch.load(load_path, map_location='cpu')
            
            # Restore basic state
            self.current_stacks = rack_data['current_stacks']
            self.max_stacks = rack_data['max_stacks']
            self.building_mode = rack_data['building_mode']
            self.default_precision = rack_data['default_precision']
            self.precision_settings = rack_data['precision_settings']
            self.qlora_adapters = rack_data['qlora_adapters']
            
            # Restore stacks
            for i, stack_state in enumerate(rack_data['stacks']):
                if i < len(self.stacks):
                    self.stacks[i].load_state_dict(stack_state)
            
            # Restore embeddings and lm_head if available
            if 'embeddings' in rack_data and hasattr(self, 'embeddings') and self.embeddings is not None:
                self.embeddings.load_state_dict(rack_data['embeddings'])
            
            if 'lm_head' in rack_data and hasattr(self, 'lm_head') and self.lm_head is not None:
                self.lm_head.load_state_dict(rack_data['lm_head'])
            
            logger.info(f"Loaded rack from: {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load rack from {load_path}: {e}")
            return False
    
    def save_stack(self, stack_idx: int, path: str) -> str:
        """
        Save individual stack to file.
        
        Args:
            stack_idx: Index of stack to save
            path: Path to save the stack
            
        Returns:
            Saved file path
        """
        if stack_idx >= len(self.stacks):
            raise ValueError(f"Stack {stack_idx} does not exist")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare stack data
        stack_data = {
            'timestamp': datetime.now().isoformat(),
            'stack_idx': stack_idx,
            'stack_state_dict': self.stacks[stack_idx].state_dict(),
            'precision': self.precision_settings.get(stack_idx, self.default_precision),
            'qlora_adapters': self.qlora_adapters.get(stack_idx),
            'config': self.config.to_dict()
        }
        
        # Save to file
        torch.save(stack_data, save_path)
        
        logger.info(f"Saved stack {stack_idx} to: {save_path}")
        return str(save_path)
    
    def load_stack(self, stack_idx: int, path: str) -> bool:
        """
        Load individual stack from file.
        
        Args:
            stack_idx: Index of stack to load
            path: Path to load the stack from
            
        Returns:
            True if successful, False otherwise
        """
        if stack_idx >= len(self.stacks):
            raise ValueError(f"Stack {stack_idx} does not exist")
        
        load_path = Path(path)
        
        if not load_path.exists():
            logger.warning(f"Stack file not found: {load_path}")
            return False
        
        try:
            stack_data = torch.load(load_path, map_location='cpu')
            
            # Restore stack state
            self.stacks[stack_idx].load_state_dict(stack_data['stack_state_dict'])
            
            # Restore precision and QLoRA settings
            if 'precision' in stack_data:
                self.precision_settings[stack_idx] = stack_data['precision']
            
            if 'qlora_adapters' in stack_data:
                self.qlora_adapters[stack_idx] = stack_data['qlora_adapters']
            
            logger.info(f"Loaded stack {stack_idx} from: {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load stack {stack_idx} from {load_path}: {e}")
            return False


class PrecisionManager:
    """Manager for precision settings across stacks"""
    
    def __init__(self):
        self.precision_modes = {
            "full": torch.float32,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "nvfp4": "nvfp4",  # NVIDIA FP4 precision
            "qlora": "qlora"  # Special QLoRA mode
        }
    
    def set_stack_precision(self, stack: Stack, precision: str):
        """Set precision for a specific stack"""
        if precision == "qlora":
            # QLoRA is handled separately
            return
        
        if precision == "nvfp4":
            # NVIDIA FP4 precision - handled separately
            return
        
        if precision in self.precision_modes:
            # Convert stack to specified precision
            stack = stack.to(self.precision_modes[precision])
        else:
            raise ValueError(f"Unknown precision: {precision}")
    
