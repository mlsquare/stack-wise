"""Validation tests for the attention module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model.attention.attention import CoreAttention
from src.model.attention.builder import AttentionBuilder


class TestCoreAttentionValidation:
    """Ensure ``CoreAttention`` guards against invalid hyper-parameters."""

    def test_rejects_invalid_dropout_probability(self) -> None:
        """Dropout must lie inside the closed interval [0, 1]."""

        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            CoreAttention(d_model=16, n_heads=4, dropout=1.2)

    def test_requires_grouped_heads_to_divide_evenly(self) -> None:
        """Grouped attention requires the number of KV heads to divide the query heads."""

        with pytest.raises(ValueError, match="n_heads must be divisible"):
            CoreAttention(d_model=16, n_heads=4, n_kv_heads=3)

    def test_disallows_more_kv_heads_than_query_heads(self) -> None:
        """The grouped attention path cannot allocate more KV heads than query heads."""

        with pytest.raises(ValueError, match="n_kv_heads cannot exceed n_heads"):
            CoreAttention(d_model=16, n_heads=4, n_kv_heads=8)

    def test_requires_positive_kernel_dim_for_random_features(self) -> None:
        """Kernel attention variants must provide a positive kernel dimension."""

        with pytest.raises(ValueError, match="kernel_dim must be positive"):
            CoreAttention(d_model=16, n_heads=4, kernel_type="gaussian", kernel_dim=0)


class TestAttentionBuilderValidation:
    """Validate convenience checks inside ``AttentionBuilder``."""

    def test_builder_enforces_valid_gqa_configuration(self) -> None:
        """The builder should propagate the grouped attention divisibility requirement."""

        builder = AttentionBuilder(d_model=16, n_heads=4)

        with pytest.raises(ValueError, match="must be divisible"):
            builder.with_gqa(3)

