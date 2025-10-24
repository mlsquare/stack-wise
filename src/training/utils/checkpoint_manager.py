"""
Checkpoint management for the unified trainer.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Checkpoint manager for saving and loading training checkpoints.
    
    Handles:
    - Model checkpoints
    - Training state
    - Configuration snapshots
    - Checkpoint metadata
    """
    
    def __init__(self, config):
        """
        Initialize checkpoint manager.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.checkpoint_dir = Path(config.cache_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint metadata
        self.checkpoints = {}
        self.current_checkpoint = None
        
        logger.info(f"Initialized CheckpointManager: {self.checkpoint_dir}")
    
    def save_checkpoint(self, block_idx: int, model_layers: List[torch.nn.Module], 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       epoch: Optional[int] = None, loss: Optional[float] = None):
        """
        Save checkpoint for a block.
        
        Args:
            block_idx: Block index
            model_layers: Model layers
            optimizer: Optimizer state (optional)
            epoch: Epoch number (optional)
            loss: Loss value (optional)
        """
        checkpoint_id = f"block_{block_idx}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'block_idx': block_idx,
            'model_state_dict': {f"layer_{i}": layer.state_dict() for i, layer in enumerate(model_layers)},
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'loss': loss
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update metadata
        self.checkpoints[checkpoint_id] = {
            'path': str(checkpoint_path),
            'block_idx': block_idx,
            'timestamp': checkpoint_data['timestamp'],
            'epoch': epoch,
            'loss': loss
        }
        
        self.current_checkpoint = checkpoint_id
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, block_idx: int) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for a block.
        
        Args:
            block_idx: Block index
            
        Returns:
            Checkpoint data or None
        """
        checkpoint_id = f"block_{block_idx}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def save_fusion_checkpoint(self, block_idx: int, all_blocks: List[List[torch.nn.Module]],
                              optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Save fusion checkpoint for all blocks up to block_idx.
        
        Args:
            block_idx: Current block index
            all_blocks: All blocks in the model
            optimizer: Optimizer state (optional)
        """
        checkpoint_id = f"fusion_block_{block_idx}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        
        # Prepare fusion checkpoint data
        checkpoint_data = {
            'block_idx': block_idx,
            'fusion_mode': self.config.fusion_mode,
            'model_state_dict': {
                f"block_{i}": {f"layer_{j}": layer.state_dict() 
                              for j, layer in enumerate(block)}
                for i, block in enumerate(all_blocks[:block_idx + 1])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update metadata
        self.checkpoints[checkpoint_id] = {
            'path': str(checkpoint_path),
            'block_idx': block_idx,
            'fusion_mode': self.config.fusion_mode,
            'timestamp': checkpoint_data['timestamp']
        }
        
        logger.info(f"Saved fusion checkpoint: {checkpoint_path}")
    
    def save_final_checkpoints(self):
        """Save final checkpoints and metadata."""
        # Save configuration snapshot
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        # Save checkpoint metadata
        metadata_path = self.checkpoint_dir / "checkpoints.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.checkpoints, f, indent=2)
        
        logger.info(f"Saved final checkpoints and metadata to {self.checkpoint_dir}")
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get checkpoint information."""
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "num_checkpoints": len(self.checkpoints),
            "current_checkpoint": self.current_checkpoint,
            "checkpoints": self.checkpoints
        }
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """
        Cleanup old checkpoints, keeping only the last N.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
        if len(self.checkpoints) <= keep_last:
            return
        
        # Sort checkpoints by timestamp
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        # Remove old checkpoints
        for checkpoint_id, _ in sorted_checkpoints[keep_last:]:
            checkpoint_path = Path(self.checkpoints[checkpoint_id]['path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            del self.checkpoints[checkpoint_id]
        
        logger.info(f"Cleaned up old checkpoints, keeping last {keep_last}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return list(self.checkpoints.values())
