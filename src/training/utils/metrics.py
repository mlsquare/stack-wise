"""
Training metrics for the unified trainer.
"""

import logging
from typing import Dict, List, Any, Optional
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrainingMetrics:
    """
    Training metrics collection and management.
    
    Tracks:
    - Training losses
    - Training times
    - Memory usage
    - Model performance
    """
    
    def __init__(self):
        """Initialize training metrics."""
        self.metrics = defaultdict(list)
        self.start_time = None
        self.current_epoch = 0
        self.current_block = 0
        
        logger.debug("Initialized TrainingMetrics")
    
    def start_training(self):
        """Start training timer."""
        self.start_time = time.time()
        logger.debug("Started training timer")
    
    def end_training(self):
        """End training timer."""
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            self.metrics['total_training_time'].append(total_time)
            logger.info(f"Total training time: {total_time:.2f} seconds")
    
    def record_loss(self, loss: float, block_idx: int, epoch: int):
        """
        Record training loss.
        
        Args:
            loss: Loss value
            block_idx: Block index
            epoch: Epoch number
        """
        self.metrics['losses'].append({
            'loss': loss,
            'block_idx': block_idx,
            'epoch': epoch,
            'timestamp': time.time()
        })
        
        logger.debug(f"Recorded loss: {loss:.4f} for block {block_idx}, epoch {epoch}")
    
    def record_block_time(self, block_idx: int, block_time: float):
        """
        Record block training time.
        
        Args:
            block_idx: Block index
            block_time: Time taken for block
        """
        self.metrics['block_times'].append({
            'block_idx': block_idx,
            'time': block_time,
            'timestamp': time.time()
        })
        
        logger.debug(f"Recorded block {block_idx} time: {block_time:.2f} seconds")
    
    def record_memory_usage(self, memory_mb: float, block_idx: int):
        """
        Record memory usage.
        
        Args:
            memory_mb: Memory usage in MB
            block_idx: Block index
        """
        self.metrics['memory_usage'].append({
            'memory_mb': memory_mb,
            'block_idx': block_idx,
            'timestamp': time.time()
        })
        
        logger.debug(f"Recorded memory usage: {memory_mb:.2f} MB for block {block_idx}")
    
    def record_quantization_info(self, block_idx: int, quantization_type: str, 
                                qlora_enabled: bool, qlora_rank: int):
        """
        Record quantization information.
        
        Args:
            block_idx: Block index
            quantization_type: Type of quantization
            qlora_enabled: Whether QLoRA is enabled
            qlora_rank: QLoRA rank
        """
        self.metrics['quantization_info'].append({
            'block_idx': block_idx,
            'quantization_type': quantization_type,
            'qlora_enabled': qlora_enabled,
            'qlora_rank': qlora_rank,
            'timestamp': time.time()
        })
        
        logger.debug(f"Recorded quantization info for block {block_idx}")
    
    def record_cache_info(self, block_idx: int, cache_size: int, cache_memory_mb: float):
        """
        Record cache information.
        
        Args:
            block_idx: Block index
            cache_size: Cache size
            cache_memory_mb: Cache memory usage in MB
        """
        self.metrics['cache_info'].append({
            'block_idx': block_idx,
            'cache_size': cache_size,
            'cache_memory_mb': cache_memory_mb,
            'timestamp': time.time()
        })
        
        logger.debug(f"Recorded cache info for block {block_idx}")
    
    def get_average_loss(self, block_idx: Optional[int] = None) -> float:
        """
        Get average loss for a block or overall.
        
        Args:
            block_idx: Block index (optional)
            
        Returns:
            Average loss
        """
        losses = self.metrics['losses']
        
        if block_idx is not None:
            losses = [l for l in losses if l['block_idx'] == block_idx]
        
        if not losses:
            return 0.0
        
        return sum(l['loss'] for l in losses) / len(losses)
    
    def get_block_times(self) -> Dict[int, float]:
        """Get training times for each block."""
        block_times = {}
        for entry in self.metrics['block_times']:
            block_idx = entry['block_idx']
            if block_idx not in block_times:
                block_times[block_idx] = 0.0
            block_times[block_idx] += entry['time']
        
        return block_times
    
    def get_memory_usage(self) -> Dict[int, float]:
        """Get memory usage for each block."""
        memory_usage = {}
        for entry in self.metrics['memory_usage']:
            block_idx = entry['block_idx']
            if block_idx not in memory_usage:
                memory_usage[block_idx] = 0.0
            memory_usage[block_idx] = max(memory_usage[block_idx], entry['memory_mb'])
        
        return memory_usage
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            'total_training_time': self.metrics['total_training_time'][-1] if self.metrics['total_training_time'] else 0.0,
            'average_loss': self.get_average_loss(),
            'block_times': self.get_block_times(),
            'memory_usage': self.get_memory_usage(),
            'num_losses': len(self.metrics['losses']),
            'num_blocks': len(set(l['block_idx'] for l in self.metrics['losses']))
        }
    
    def log_final_metrics(self):
        """Log final training metrics."""
        metrics = self.get_metrics()
        
        logger.info("=== Final Training Metrics ===")
        logger.info(f"Total training time: {metrics['total_training_time']:.2f} seconds")
        logger.info(f"Average loss: {metrics['average_loss']:.4f}")
        logger.info(f"Number of blocks: {metrics['num_blocks']}")
        logger.info(f"Number of loss recordings: {metrics['num_losses']}")
        
        if metrics['block_times']:
            logger.info("Block training times:")
            for block_idx, time_taken in metrics['block_times'].items():
                logger.info(f"  Block {block_idx}: {time_taken:.2f} seconds")
        
        if metrics['memory_usage']:
            logger.info("Peak memory usage:")
            for block_idx, memory_mb in metrics['memory_usage'].items():
                logger.info(f"  Block {block_idx}: {memory_mb:.2f} MB")
    
    def clear_metrics(self):
        """Clear all metrics."""
        self.metrics.clear()
        self.start_time = None
        self.current_epoch = 0
        self.current_block = 0
        
        logger.debug("Cleared all metrics")
