"""
Training Machine: Orchestrator for unified time-based training.

The TrainingMachine is the core orchestrator that binds together the three
planes (Data, Model, Optimizer) across a schedule/curriculum.

At each time step t, it:
1. Converts step to normalized time t ∈ [0,1]
2. Determines the active phase from the schedule
3. Assembles the complete training state (Data_t, Model_t, Optimizer_t)
4. Performs training steps
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
import torch
import torch.nn as nn
import logging
from pathlib import Path

from .schedule import PhaseSchedule, TimeController, Phase
try:
    from ..framework.adapters import DataAdapter, ModelAdapter, OptimizerAdapter
    from ..framework.factories import (
        build_data_loader, build_model, build_optimizer,
        validate_training_state
    )
    from ..framework.specs import BatchSpec, OptimizerSpec
except ImportError:
    # Handle import when running from examples
    from framework.adapters import DataAdapter, ModelAdapter, OptimizerAdapter
    from framework.factories import (
        build_data_loader, build_model, build_optimizer,
        validate_training_state
    )
    from framework.specs import BatchSpec, OptimizerSpec

logger = logging.getLogger(__name__)


class TrainingMachine:
    """
    Orchestrator for time-based training across data, model, and optimizer planes.
    
    The training machine:
    - Maintains a time controller and phase schedule
    - Assembles (Data, Model, Optimizer) triplets at each step
    - Coordinates training loops with phase transitions
    - Supports curriculum learning and gradual model/data/optimizer evolution
    """
    
    def __init__(self,
                 config_or_schedule: Dict,
                 total_steps: int,
                 start_step: int = 0):
        """
        Initialize training machine.
        
        Args:
            config_or_schedule: Configuration dict or phase schedule
            total_steps: Total number of training steps
            start_step: Starting step (for resuming)
        """
        self.total_steps = total_steps
        
        # Create time controller
        self.time_controller = TimeController(total_steps, start_step)
        
        # Create or extract phase schedule
        if 'schedule' in config_or_schedule:
            schedule_dict = config_or_schedule['schedule']
            if isinstance(schedule_dict, list):
                # List of phase dicts
                from .schedule import create_simple_schedule
                self.schedule, _ = create_simple_schedule(total_steps, schedule_dict)
            elif isinstance(schedule_dict, dict) and 'phases' in schedule_dict:
                # Schedule dict with phases
                self.schedule = PhaseSchedule.from_dict(schedule_dict)
            else:
                raise ValueError("Invalid schedule format in config")
        else:
            # Create single-phase schedule from config
            self.schedule = PhaseSchedule.single_phase(
                task=config_or_schedule.get('task', 'mlm'),
                data_config=config_or_schedule.get('data', {}),
                model_config=config_or_schedule.get('model', {}),
                optimizer_config=config_or_schedule.get('training', {}).get('optimizer', {})
            )
        
        # Store base config
        self.base_config = config_or_schedule
        
        # Current training state
        self.current_phase: Optional[Phase] = None
        self.current_data: Optional[DataAdapter] = None
        self.current_model: Optional[ModelAdapter] = None
        self.current_optimizer: Optional[OptimizerAdapter] = None
        
        # Phase transition tracking
        self.phase_transitions: List[int] = []
        self.phase_history: List[Dict[str, Any]] = []
        
        # Training metrics
        self.step_losses: List[float] = []
        self.step_metrics: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized TrainingMachine with {len(self.schedule.phases)} phases over {total_steps} steps")
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current training state (Data_t, Model_t, Optimizer_t).
        
        Returns:
            Dictionary with current phase and components
        """
        t = self.time_controller.get_current_time()
        phase = self.schedule.at(t)
        
        return {
            'step': self.time_controller.current_step,
            't': t,
            'phase': phase,
            'data': self.current_data,
            'model': self.current_model,
            'optimizer': self.current_optimizer
        }
    
    def assemble_planes(self) -> Tuple[DataAdapter, ModelAdapter, OptimizerAdapter]:
        """
        Assemble the three planes at the current time.
        
        This is the core method that creates the causal path:
        1. Get normalized time t
        2. Determine active phase
        3. Build or adapt data, model, optimizer according to phase config
        
        Returns:
            Tuple of (data_adapter, model_adapter, optimizer_adapter)
        """
        t = self.time_controller.get_current_time()
        phase = self.schedule.at(t)
        
        logger.debug(f"Assembling planes at step={self.time_controller.current_step}, t={t:.4f}, phase={phase.name}")
        
        # Check if we've transitioned to a new phase
        phase_changed = (self.current_phase != phase)
        
        if phase_changed:
            logger.info(f"Phase transition: {self.current_phase.name if self.current_phase else 'None'} -> {phase.name}")
            self.current_phase = phase
            self.phase_transitions.append(self.time_controller.current_step)
            
            # Build new components for this phase
            data = self._build_data(phase)
            model = self._build_model(phase)
            optimizer = self._build_optimizer(model, phase)
            
            self.current_data = data
            self.current_model = model
            self.current_optimizer = optimizer
            
            # Validate compatibility
            if not validate_training_state(data, model, optimizer):
                logger.warning("Training state validation failed, continuing anyway")
        else:
            # Reuse existing components
            data = self.current_data
            model = self.current_model
            optimizer = self.current_optimizer
        
        return data, model, optimizer
    
    def _build_data(self, phase: Phase) -> DataAdapter:
        """Build data adapter for current phase."""
        # Merge base config with phase-specific data config
        base_data_config = self.base_config.get('data', {})
        phase_data_config = phase.data_config
        
        # If phase has data config, use it; otherwise use base
        if phase_data_config:
            merged_config = {**base_data_config, **phase_data_config}
        else:
            merged_config = base_data_config
        
        # Build data loader
        # Pass base config as config_or_data, and phase config as override
        data = build_data_loader(self.base_config, phase_data_config)
        
        return data
    
    def _build_model(self, phase: Phase) -> ModelAdapter:
        """Build model adapter for current phase."""
        # Merge base config with phase-specific model config
        base_model_config = self.base_config.get('model', {})
        phase_model_config = phase.model_config
        
        # If phase has model config, use it; otherwise use base
        if phase_model_config:
            merged_config = {**base_model_config, **phase_model_config}
        else:
            merged_config = base_model_config
        
        # Build model
        # Pass base config as config_or_model, and phase config as override
        model = build_model(self.base_config, {'model': phase_model_config})
        
        return model
    
    def _build_optimizer(self, model: ModelAdapter, phase: Phase) -> OptimizerAdapter:
        """Build optimizer adapter for current phase."""
        # Merge base config with phase-specific optimizer config
        base_optimizer_config = self.base_config.get('training', {}).get('optimizer', {})
        phase_optimizer_config = phase.optimizer_config
        
        # Build optimizer with phase config
        optimizer = build_optimizer(
            {'training': {'optimizer': phase_optimizer_config or base_optimizer_config}},
            model,
            phase_optimizer_config
        )
        
        return optimizer
    
    def train_step(self, loss_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Perform a single training step.
        
        Args:
            loss_fn: Optional loss function (takes batch, outputs, returns loss)
            
        Returns:
            Dictionary with metrics for this step
        """
        # Assemble planes
        data_adapter, model_adapter, optimizer_adapter = self.assemble_planes()
        
        # Get batch
        try:
            batch_iter = iter(data_adapter)
            batch = next(batch_iter)
        except StopIteration:
            # Reset data loader
            batch_iter = iter(data_adapter)
            batch = next(batch_iter)
        
        # Forward pass
        input_ids = batch.get('input_ids', batch.get('input'))
        if input_ids is None:
            raise ValueError("Batch must contain 'input_ids' or 'input'")
        
        outputs = model_adapter.forward(input_ids)
        
        # Compute loss
        if loss_fn is not None:
            loss = loss_fn(batch, outputs)
        else:
            # Default loss: cross-entropy if targets available
            if 'targets' in batch or 'labels' in batch:
                targets = batch.get('targets', batch.get('labels'))
                
                # Flatten for cross-entropy
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.shape[-1]),
                    targets.view(-1)
                )
            else:
                raise ValueError("No loss function provided and no targets in batch")
        
        # Backward pass
        optimizer_adapter.zero_grad()
        loss.backward()
        optimizer_adapter.step(loss)
        
        # Record metrics
        step_metrics = {
            'step': self.time_controller.current_step,
            't': self.time_controller.get_current_time(),
            'phase': self.current_phase.name if self.current_phase else 'unknown',
            'loss': loss.item()
        }
        
        self.step_losses.append(loss.item())
        self.step_metrics.append(step_metrics)
        
        # Advance time
        self.time_controller.advance(1)
        
        return step_metrics
    
    def train(self,
              loss_fn: Optional[Callable] = None,
              log_interval: int = 10,
              save_interval: Optional[int] = None,
              save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete training loop.
        
        Args:
            loss_fn: Optional loss function
            log_interval: Log every N steps
            save_interval: Save checkpoint every N steps (optional)
            save_dir: Directory to save checkpoints (optional)
            
        Returns:
            Dictionary with training summary
        """
        logger.info("Starting training loop")
        
        # Create save directory if needed
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Training loop
        while not self.time_controller.is_complete():
            # Training step
            metrics = self.train_step(loss_fn)
            
            # Logging
            if self.time_controller.current_step % log_interval == 0:
                logger.info(
                    f"Step {metrics['step']}/{self.total_steps} "
                    f"(t={metrics['t']:.4f}, phase={metrics['phase']}): "
                    f"loss={metrics['loss']:.4f}"
                )
            
            # Checkpointing
            if save_interval is not None and save_dir is not None:
                if self.time_controller.current_step % save_interval == 0:
                    self.save_checkpoint(save_dir / f"checkpoint_step_{self.time_controller.current_step}.pt")
        
        logger.info("Training completed")
        
        # Return summary
        return {
            'total_steps': self.total_steps,
            'phase_transitions': self.phase_transitions,
            'final_loss': self.step_losses[-1] if self.step_losses else None,
            'avg_loss': sum(self.step_losses) / len(self.step_losses) if self.step_losses else None,
            'metrics': self.step_metrics
        }
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'time_controller': {
                'total_steps': self.time_controller.total_steps,
                'current_step': self.time_controller.current_step
            },
            'schedule': self.schedule.to_dict(),
            'base_config': self.base_config,
            'current_phase': self.current_phase.name if self.current_phase else None,
            'phase_transitions': self.phase_transitions,
            'metrics': self.step_metrics[-1000:] if self.step_metrics else []  # Last 1000 steps
        }
        
        # Save model state if available
        if self.current_model is not None and self.current_model.rack is not None:
            checkpoint['model_state'] = self.current_model.rack.state_dict()
        
        # Save optimizer state if available
        if self.current_optimizer is not None:
            checkpoint['optimizer_state'] = self.current_optimizer.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Restore time controller
        tc_data = checkpoint['time_controller']
        self.time_controller = TimeController(tc_data['total_steps'], tc_data['current_step'])
        
        # Restore schedule
        self.schedule = PhaseSchedule.from_dict(checkpoint['schedule'])
        
        # Restore phase transitions and metrics
        self.phase_transitions = checkpoint.get('phase_transitions', [])
        self.step_metrics = checkpoint.get('metrics', [])
        self.step_losses = [m['loss'] for m in self.step_metrics]
        
        logger.info(f"Loaded checkpoint from {path}, resumed at step {self.time_controller.current_step}")

