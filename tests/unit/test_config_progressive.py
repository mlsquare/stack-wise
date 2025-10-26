import pytest
from src.config.base import TrainingConfig, ProgressiveConfig


def test_progressive_config_defaults_and_validate():
    prog = ProgressiveConfig()
    # Should not raise
    prog.validate()
    assert prog.enabled is True
    assert prog.max_stacks >= 1


def test_training_config_from_dict_converts_progressive_and_lr_alias():
    data = {
        'lr': 0.001,
        'batch_size': 8,
        'progressive': {
            'enabled': True,
            'max_stacks': 5,
            'qlora_enabled': False,
            'trunk_strategy': 'frozen'
        }
    }

    tc = TrainingConfig.from_dict(data)
    # lr should be in optimizer config
    assert hasattr(tc.optimizer, 'lr')
    assert tc.optimizer.lr == pytest.approx(0.001)
    # progressive should be ProgressiveConfig
    assert isinstance(tc.progressive, ProgressiveConfig)
    assert tc.progressive.max_stacks == 5


def test_progressive_config_validation_errors():
    # invalid strategy
    with pytest.raises(ValueError):
        prog = ProgressiveConfig(qlora_strategy='unknown')
        prog.validate()

    # negative max_stacks
    with pytest.raises(ValueError):
        prog = ProgressiveConfig(max_stacks=0)
        prog.validate()
