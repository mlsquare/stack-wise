"""
Configuration validation for the unified trainer.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Configuration validator for training configurations.
    
    Validates training configurations to ensure they are valid and consistent.
    """
    
    def __init__(self):
        """Initialize configuration validator."""
        self.errors = []
        self.warnings = []
    
    def validate(self, config) -> bool:
        """
        Validate training configuration.
        
        Args:
            config: Training configuration
            
        Returns:
            True if valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Validate basic training parameters
        self._validate_basic_params(config)
        
        # Validate time-step masking parameters
        if hasattr(config, 'time_step_masking') and config.time_step_masking:
            self._validate_time_step_masking(config)
        
        # Validate quantization parameters
        if hasattr(config, 'quantization_enabled') and config.quantization_enabled:
            self._validate_quantization(config)
        
        # Validate QLoRA parameters
        if hasattr(config, 'qlora_enabled') and config.qlora_enabled:
            self._validate_qlora(config)
        
        # Validate caching parameters
        self._validate_caching(config)
        
        # Log results
        if self.errors:
            for error in self.errors:
                logger.error(f"Configuration error: {error}")
        
        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"Configuration warning: {warning}")
        
        return len(self.errors) == 0
    
    def _validate_basic_params(self, config):
        """Validate basic training parameters."""
        # Validate training mode
        valid_strategies = ['progressive', 'end_to_end']
        if hasattr(config, 'strategy') and config.strategy not in valid_strategies:
            self.errors.append(f"Invalid training strategy: {config.strategy}. Must be one of {valid_strategies}")
        
        # Validate end-to-end training scope
        valid_scopes = ['stackwise', 'rackwise']
        if hasattr(config, 'end_to_end_scope') and config.end_to_end_scope not in valid_scopes:
            self.errors.append(f"Invalid end-to-end scope: {config.end_to_end_scope}. Must be one of {valid_scopes}")
        
        # Validate block size
        if hasattr(config, 'block_size') and config.block_size < 1:
            self.errors.append(f"Invalid block size: {config.block_size}. Must be >= 1")
        
        # Validate fusion mode
        if hasattr(config, 'fusion_mode'):
            valid_fusion_modes = ['frozen', 'trainable']
            if config.fusion_mode not in valid_fusion_modes:
                self.errors.append(f"Invalid fusion mode: {config.fusion_mode}. Must be one of {valid_fusion_modes}")
        
        # Validate learning rate
        if hasattr(config, 'optimizer') and hasattr(config.optimizer, 'lr') and config.optimizer.lr <= 0:
            self.errors.append(f"Invalid learning rate: {config.optimizer.lr}. Must be > 0")
        
        # Validate batch size
        if hasattr(config, 'batch_size') and config.batch_size < 1:
            self.errors.append(f"Invalid batch size: {config.batch_size}. Must be >= 1")
    
    def _validate_time_step_masking(self, config):
        """Validate time-step masking parameters."""
        # Validate number of time steps
        if hasattr(config, 'num_time_steps') and config.num_time_steps < 1:
            self.errors.append(f"Invalid num_time_steps: {config.num_time_steps}. Must be >= 1")
        
        # Validate time step bins
        if hasattr(config, 'time_step_bins'):
            if not isinstance(config.time_step_bins, list):
                self.errors.append("time_step_bins must be a list")
            elif len(config.time_step_bins) == 0:
                self.errors.append("time_step_bins cannot be empty")
            elif not all(isinstance(t, int) for t in config.time_step_bins):
                self.errors.append("All time_step_bins must be integers")
        
        # Validate mask fractions
        if hasattr(config, 'time_step_mask_fractions'):
            if not isinstance(config.time_step_mask_fractions, dict):
                self.errors.append("time_step_mask_fractions must be a dictionary")
            else:
                for time_t, fraction in config.time_step_mask_fractions.items():
                    if not isinstance(time_t, int):
                        self.errors.append(f"Time step {time_t} must be an integer")
                    if not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
                        self.errors.append(f"Mask fraction for time {time_t} must be between 0 and 1")
    
    def _validate_quantization(self, config):
        """Validate quantization parameters."""
        # Validate quantization type
        valid_types = ['nf_fp8', 'fp16', 'fp32']
        if hasattr(config, 'quantization_type') and config.quantization_type not in valid_types:
            self.errors.append(f"Invalid quantization type: {config.quantization_type}. Must be one of {valid_types}")
        
        # Validate mixed precision settings
        if hasattr(config, 'mixed_precision') and config.mixed_precision:
            if not hasattr(config, 'backbone_quantized') or not hasattr(config, 'adapters_full_precision'):
                self.warnings.append("Mixed precision enabled but backbone_quantized or adapters_full_precision not set")
    
    def _validate_qlora(self, config):
        """Validate QLoRA parameters."""
        # Validate QLoRA rank
        if hasattr(config, 'qlora_rank') and config.qlora_rank < 1:
            self.errors.append(f"Invalid QLoRA rank: {config.qlora_rank}. Must be >= 1")
        
        # Validate QLoRA alpha
        if hasattr(config, 'qlora_alpha') and config.qlora_alpha < 1:
            self.errors.append(f"Invalid QLoRA alpha: {config.qlora_alpha}. Must be >= 1")
        
        # Validate QLoRA dropout
        if hasattr(config, 'qlora_dropout'):
            if not isinstance(config.qlora_dropout, (int, float)) or not 0 <= config.qlora_dropout <= 1:
                self.errors.append(f"Invalid QLoRA dropout: {config.qlora_dropout}. Must be between 0 and 1")
    
    def _validate_caching(self, config):
        """Validate caching parameters."""
        # Validate cache mode
        valid_cache_modes = ['stack', 'rack']
        if hasattr(config, 'cache_mode') and config.cache_mode not in valid_cache_modes:
            self.errors.append(f"Invalid cache mode: {config.cache_mode}. Must be one of {valid_cache_modes}")
        
        # Validate cache directory
        if hasattr(config, 'cache_dir'):
            if not isinstance(config.cache_dir, str):
                self.errors.append("cache_dir must be a string")
        
        # Validate time step cache size
        if hasattr(config, 'time_step_cache_size') and config.time_step_cache_size < 1:
            self.errors.append(f"Invalid time_step_cache_size: {config.time_step_cache_size}. Must be >= 1")
    
    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self.errors.copy()
    
    def get_warnings(self) -> List[str]:
        """Get validation warnings."""
        return self.warnings.copy()
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.errors) == 0
