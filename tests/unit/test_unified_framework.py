"""
Unit tests for the unified time-based training framework.

Tests the core components:
- Spec dataclasses
- Adapters
- Factories
- Schedule and time controller
- Training machine
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from framework.specs import (
    BatchSpec, EmbeddingSpec, RackSpec, HeadSpec,
    OptimizerSpec, DataSpec, validate_batch_model_compatibility
)
from framework.adapters import DataAdapter, ModelAdapter, OptimizerAdapter
from training.schedule import Phase, PhaseSchedule, TimeController
from training.training_machine import TrainingMachine


class TestSpecs(unittest.TestCase):
    """Test specification dataclasses."""
    
    def test_batch_spec(self):
        """Test BatchSpec creation and validation."""
        spec = BatchSpec(
            batch_size=4,
            seq_len=128,
            vocab_size=1000
        )
        
        self.assertEqual(spec.batch_size, 4)
        self.assertEqual(spec.seq_len, 128)
        self.assertEqual(spec.vocab_size, 1000)
        
        # Test serialization
        d = spec.to_dict()
        spec2 = BatchSpec.from_dict(d)
        self.assertEqual(spec.batch_size, spec2.batch_size)
    
    def test_embedding_spec(self):
        """Test EmbeddingSpec."""
        spec = EmbeddingSpec(vocab_size=1000, d_model=256)
        self.assertEqual(spec.vocab_size, 1000)
        self.assertEqual(spec.d_model, 256)
    
    def test_rack_spec(self):
        """Test RackSpec."""
        spec = RackSpec(n_stacks=2, d_model=256)
        self.assertEqual(spec.n_stacks, 2)
        self.assertEqual(spec.d_model, 256)
    
    def test_optimizer_spec(self):
        """Test OptimizerSpec."""
        spec = OptimizerSpec(
            optimizer_type="AdamW",
            lr=1e-4,
            betas=(0.9, 0.95)
        )
        self.assertEqual(spec.optimizer_type, "AdamW")
        self.assertEqual(spec.lr, 1e-4)
        self.assertEqual(spec.betas, (0.9, 0.95))
    
    def test_compatibility_validation(self):
        """Test batch-model compatibility validation."""
        batch_spec = BatchSpec(batch_size=4, seq_len=128, vocab_size=1000)
        embedding_spec = EmbeddingSpec(vocab_size=1000, d_model=256)
        head_spec = HeadSpec(d_model=256, output_dim=1000)
        
        # Should be compatible
        self.assertTrue(
            validate_batch_model_compatibility(batch_spec, embedding_spec, head_spec)
        )
        
        # Incompatible vocab size
        head_spec2 = HeadSpec(d_model=256, output_dim=500)
        self.assertFalse(
            validate_batch_model_compatibility(batch_spec, embedding_spec, head_spec2)
        )


class TestSchedule(unittest.TestCase):
    """Test schedule and time controller."""
    
    def test_phase_creation(self):
        """Test Phase creation and validation."""
        phase = Phase(
            start_t=0.0,
            end_t=1.0,
            task='mlm',
            name='test_phase'
        )
        
        self.assertEqual(phase.start_t, 0.0)
        self.assertEqual(phase.end_t, 1.0)
        self.assertEqual(phase.task, 'mlm')
        
        # Test contains
        self.assertTrue(phase.contains(0.5))
        self.assertFalse(phase.contains(1.5))
    
    def test_phase_schedule(self):
        """Test PhaseSchedule creation."""
        phases = [
            Phase(start_t=0.0, end_t=0.5, name='phase1'),
            Phase(start_t=0.5, end_t=1.0, name='phase2')
        ]
        
        schedule = PhaseSchedule(phases)
        self.assertEqual(len(schedule.phases), 2)
        
        # Test phase lookup
        self.assertEqual(schedule.at(0.25).name, 'phase1')
        self.assertEqual(schedule.at(0.75).name, 'phase2')
    
    def test_time_controller(self):
        """Test TimeController."""
        controller = TimeController(total_steps=1000)
        
        # Test conversion
        t = controller.normalized(500)
        self.assertAlmostEqual(t, 0.5)
        
        step = controller.step_from_normalized(0.5)
        self.assertEqual(step, 500)
        
        # Test advancement
        controller.advance(100)
        self.assertEqual(controller.current_step, 100)


class TestAdapters(unittest.TestCase):
    """Test adapters."""
    
    def test_data_adapter(self):
        """Test DataAdapter."""
        # Create dummy dataloader
        data = torch.randint(0, 1000, (100, 128))
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # Create spec
        spec = BatchSpec(batch_size=4, seq_len=128, vocab_size=1000)
        
        # Create adapter
        adapter = DataAdapter(dataloader, spec)
        
        # Test iteration
        batch = next(iter(adapter))
        self.assertIn('input_ids', batch)
    
    def test_model_adapter(self):
        """Test ModelAdapter."""
        # Create simple model
        model = nn.Sequential(
            nn.Embedding(1000, 256),
            nn.Linear(256, 1000)
        )
        
        adapter = ModelAdapter(rack=model)
        
        # Test forward
        input_ids = torch.randint(0, 1000, (4, 128))
        outputs = adapter.forward(input_ids)
        
        self.assertEqual(outputs.shape, (4, 128, 1000))


class TestTrainingMachine(unittest.TestCase):
    """Test TrainingMachine orchestrator."""
    
    def test_single_phase(self):
        """Test single-phase training."""
        config = {
            'model': {
                'vocab_size': 1000,
                'd_model': 128,
                'n_heads': 4,
                'n_kv_heads': 2,
                'd_ff': 512,
                'architecture': {
                    'n_stacks': 1,
                    'blocks_per_stack': 1
                }
            },
            'training': {
                'batch_size': 2,
                'seq_len': 32,
                'optimizer': {
                    'optimizer_type': 'AdamW',
                    'lr': 1e-4
                }
            },
            'data': {
                'use_dummy_data': True,
                'num_samples': 10
            },
            'task': 'mlm'
        }
        
        machine = TrainingMachine(config, total_steps=10)
        
        # Check initial state
        state = machine.get_current_state()
        self.assertEqual(state['step'], 0)
        self.assertAlmostEqual(state['t'], 0.0)
        
        # Run a few steps
        for i in range(3):
            metrics = machine.train_step()
            self.assertIn('loss', metrics)
            self.assertIn('step', metrics)
            self.assertIn('phase', metrics)
    
    def test_two_phase_curriculum(self):
        """Test two-phase curriculum."""
        config = {
            'model': {
                'vocab_size': 1000,
                'd_model': 128,
                'n_heads': 4,
                'n_kv_heads': 2,
                'd_ff': 512,
                'architecture': {
                    'n_stacks': 1,
                    'blocks_per_stack': 1
                }
            },
            'training': {
                'batch_size': 2,
                'seq_len': 32,
                'optimizer': {
                    'optimizer_type': 'AdamW',
                    'lr': 1e-4
                }
            },
            'data': {
                'use_dummy_data': True,
                'num_samples': 10
            },
            'schedule': [
                {
                    'start_t': 0.0,
                    'end_t': 0.5,
                    'name': 'phase1'
                },
                {
                    'start_t': 0.5,
                    'end_t': 1.0,
                    'name': 'phase2'
                }
            ]
        }
        
        machine = TrainingMachine(config, total_steps=100)
        
        # Check phases
        self.assertEqual(len(machine.schedule.phases), 2)
        
        # Verify phase transitions
        phase1 = machine.schedule.at(0.0)
        self.assertEqual(phase1.name, 'phase1')
        
        phase2 = machine.schedule.at(0.5)
        self.assertEqual(phase2.name, 'phase2')


if __name__ == '__main__':
    unittest.main()

