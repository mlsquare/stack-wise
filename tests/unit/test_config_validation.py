"""Unit tests for configuration validation logic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DataConfig, ModelConfig


class TestModelConfigValidation:
    """Tests covering the validation constraints on ``ModelConfig``."""

    def test_mask_fraction_range_must_be_valid(self) -> None:
        """``mask_fraction_min`` must be strictly smaller than ``mask_fraction_max``."""

        config = ModelConfig(mask_fraction_min=0.5, mask_fraction_max=0.4)

        with pytest.raises(ValueError, match="mask_fraction_min must be less than mask_fraction_max"):
            config.validate()

    def test_dropout_range_is_enforced(self) -> None:
        """Dropout values outside of [0, 1] should raise an error."""

        config = ModelConfig(dropout=1.5)

        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            config.validate()

    def test_special_mask_id_must_be_non_negative(self) -> None:
        """Negative special mask identifiers are rejected."""

        config = ModelConfig(special_mask_id=-3)

        with pytest.raises(ValueError, match="special_mask_id must be non-negative"):
            config.validate()


class TestDataConfigValidation:
    """Tests covering the validation constraints on ``DataConfig``."""

    def test_dataset_path_required_when_not_using_dummy_data(self) -> None:
        """An explicit dataset path is required when dummy data is disabled."""

        config = DataConfig(use_dummy_data=False, dataset_path=None)

        with pytest.raises(ValueError, match="dataset_path must be provided when use_dummy_data is False"):
            config.validate()
