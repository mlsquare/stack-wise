"""
Time scheduling and phase management for StackWise training.

This module provides:
- Normalized time t ∈ [0,1] and step conversion
- Phase schedule defining training phases
- Time controller for managing time evolution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class Phase:
    """A training phase in the curriculum."""
    
    start_t: float  # Normalized start time [0, 1]
    end_t: float    # Normalized end time [0, 1]
    
    # Phase-specific configuration
    task: str = "mlm"  # Training task/objective
    
    # Per-plane configurations (optional overrides)
    data_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    
    name: str = ""  # Human-readable phase name
    
    def __post_init__(self):
        """Validate phase."""
        if not 0 <= self.start_t <= 1:
            raise ValueError(f"start_t must be in [0, 1], got {self.start_t}")
        if not 0 <= self.end_t <= 1:
            raise ValueError(f"end_t must be in [0, 1], got {self.end_t}")
        if self.start_t >= self.end_t:
            raise ValueError(f"start_t must be < end_t, got [{self.start_t}, {self.end_t})")
    
    def contains(self, t: float) -> bool:
        """
        Check if phase contains normalized time t.
        
        Args:
            t: Normalized time [0, 1]
            
        Returns:
            True if t is in [start_t, end_t)
        """
        return self.start_t <= t < self.end_t
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert phase to dictionary."""
        return {
            'start_t': self.start_t,
            'end_t': self.end_t,
            'task': self.task,
            'data_config': self.data_config,
            'model_config': self.model_config,
            'optimizer_config': self.optimizer_config,
            'name': self.name
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Phase':
        """Create phase from dictionary."""
        return cls(**d)


class PhaseSchedule:
    """Schedule of training phases over normalized time."""
    
    def __init__(self, phases: List[Phase]):
        """
        Initialize phase schedule.
        
        Args:
            phases: List of phases (must cover [0, 1] without gaps)
        """
        if not phases:
            raise ValueError("Phase schedule must have at least one phase")
        
        # Sort phases by start time
        sorted_phases = sorted(phases, key=lambda p: p.start_t)
        
        # Validate coverage
        if sorted_phases[0].start_t != 0.0:
            raise ValueError(f"First phase must start at 0.0, got {sorted_phases[0].start_t}")
        
        if sorted_phases[-1].end_t != 1.0:
            raise ValueError(f"Last phase must end at 1.0, got {sorted_phases[-1].end_t}")
        
        # Validate no gaps or overlaps
        for i in range(len(sorted_phases) - 1):
            if sorted_phases[i].end_t != sorted_phases[i + 1].start_t:
                raise ValueError(
                    f"Phases must be contiguous. "
                    f"Phase {i} ends at {sorted_phases[i].end_t}, "
                    f"phase {i+1} starts at {sorted_phases[i+1].start_t}"
                )
        
        self.phases = sorted_phases
        
        logger.info(f"Initialized PhaseSchedule with {len(self.phases)} phases")
        for i, phase in enumerate(self.phases):
            logger.info(f"  Phase {i}: {phase.name} [{phase.start_t:.3f}, {phase.end_t:.3f})")
    
    def at(self, t: float) -> Phase:
        """
        Get the phase at normalized time t.
        
        Args:
            t: Normalized time [0, 1]
            
        Returns:
            Phase active at time t
            
        Raises:
            ValueError: If t is outside [0, 1]
        """
        if not 0 <= t <= 1:
            raise ValueError(f"Normalized time must be in [0, 1], got {t}")
        
        # Special case for t = 1.0 (belongs to last phase)
        if t == 1.0:
            return self.phases[-1]
        
        # Binary search for phase containing t
        for phase in self.phases:
            if phase.contains(t):
                return phase
        
        # Should never reach here if phases are valid
        raise RuntimeError(f"No phase found for t={t}")
    
    def get_phase_transitions(self) -> List[float]:
        """
        Get list of phase transition times.
        
        Returns:
            List of normalized times where phases transition
        """
        transitions = [0.0]  # Start
        transitions.extend([p.end_t for p in self.phases[:-1]])  # Internal transitions
        transitions.append(1.0)  # End
        return sorted(set(transitions))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schedule to dictionary."""
        return {
            'phases': [phase.to_dict() for phase in self.phases]
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PhaseSchedule':
        """Create schedule from dictionary."""
        phases = [Phase.from_dict(p) for p in d.get('phases', [])]
        return cls(phases)
    
    @classmethod
    def single_phase(cls, 
                    task: str = "mlm",
                    data_config: Optional[Dict] = None,
                    model_config: Optional[Dict] = None,
                    optimizer_config: Optional[Dict] = None,
                    name: str = "single_phase") -> 'PhaseSchedule':
        """
        Create a single-phase schedule (no curriculum).
        
        Args:
            task: Training task
            data_config: Optional data config
            model_config: Optional model config
            optimizer_config: Optional optimizer config
            name: Phase name
            
        Returns:
            Single-phase schedule
        """
        phase = Phase(
            start_t=0.0,
            end_t=1.0,
            task=task,
            data_config=data_config or {},
            model_config=model_config or {},
            optimizer_config=optimizer_config or {},
            name=name
        )
        return cls([phase])


class TimeController:
    """Controller for managing normalized time and step conversion."""
    
    def __init__(self, total_steps: int, start_step: int = 0):
        """
        Initialize time controller.
        
        Args:
            total_steps: Total number of training steps
            start_step: Starting step (for resuming)
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {start_step}")
        
        self.total_steps = total_steps
        self.current_step = start_step
        
        logger.info(f"Initialized TimeController: {total_steps} steps starting at {start_step}")
    
    def normalized(self, step: Optional[int] = None) -> float:
        """
        Convert step to normalized time t ∈ [0, 1].
        
        Args:
            step: Training step (None uses current_step)
            
        Returns:
            Normalized time in [0, 1]
        """
        if step is None:
            step = self.current_step
        
        # Clip to [0, total_steps]
        step = max(0, min(step, self.total_steps))
        
        # Convert to normalized time
        t = step / self.total_steps
        
        return float(t)
    
    def step_from_normalized(self, t: float) -> int:
        """
        Convert normalized time to step.
        
        Args:
            t: Normalized time [0, 1]
            
        Returns:
            Training step
        """
        if not 0 <= t <= 1:
            raise ValueError(f"Normalized time must be in [0, 1], got {t}")
        
        step = int(t * self.total_steps)
        # Ensure step is in valid range
        step = max(0, min(step, self.total_steps))
        
        return step
    
    def advance(self, steps: int = 1):
        """
        Advance time by specified number of steps.
        
        Args:
            steps: Number of steps to advance
        """
        self.current_step += steps
    
    def set_step(self, step: int):
        """
        Set current step.
        
        Args:
            step: Step to set
        """
        if not 0 <= step <= self.total_steps:
            raise ValueError(f"Step must be in [0, {self.total_steps}], got {step}")
        self.current_step = step
    
    def get_current_time(self) -> float:
        """Get current normalized time."""
        return self.normalized(self.current_step)
    
    def get_progress(self) -> float:
        """Get training progress as a fraction."""
        return self.get_current_time()
    
    def is_complete(self) -> bool:
        """Check if training is complete."""
        return self.current_step >= self.total_steps
    
    def remaining_steps(self) -> int:
        """Get number of remaining steps."""
        return max(0, self.total_steps - self.current_step)


def create_simple_schedule(total_steps: int,
                          phases: List[Dict[str, Any]]) -> Tuple[PhaseSchedule, TimeController]:
    """
    Create a simple phase schedule and time controller.
    
    Args:
        total_steps: Total number of training steps
        phases: List of phase dictionaries with keys:
            - start_t: Start time [0, 1] (or start_step as int)
            - end_t: End time [0, 1] (or end_step as int)
            - task: Task name
            - data_config, model_config, optimizer_config: Optional configs
            - name: Phase name
            
    Returns:
        Tuple of (PhaseSchedule, TimeController)
        
    Examples:
        >>> # Two phases: 0-0.5 and 0.5-1.0
        >>> phases = [
        ...     {'start_t': 0.0, 'end_t': 0.5, 'name': 'phase1'},
        ...     {'start_t': 0.5, 'end_t': 1.0, 'name': 'phase2'}
        ... ]
        >>> schedule, controller = create_simple_schedule(1000, phases)
        
        >>> # Steps-based phases (converted to normalized time)
        >>> phases = [
        ...     {'start_step': 0, 'end_step': 500, 'name': 'phase1'},
        ...     {'start_step': 500, 'end_step': 1000, 'name': 'phase2'}
        ... ]
        >>> schedule, controller = create_simple_schedule(1000, phases)
    """
    # Convert step-based phases to normalized time if needed
    converted_phases = []
    for phase in phases:
        phase_dict = dict(phase)
        
        # Convert start_step/end_step to start_t/end_t if present
        if 'start_step' in phase_dict:
            phase_dict['start_t'] = phase_dict['start_step'] / total_steps
            del phase_dict['start_step']
        
        if 'end_step' in phase_dict:
            phase_dict['end_t'] = phase_dict['end_step'] / total_steps
            del phase_dict['end_step']
        
        converted_phases.append(Phase.from_dict(phase_dict))
    
    schedule = PhaseSchedule(converted_phases)
    controller = TimeController(total_steps)
    
    return schedule, controller

