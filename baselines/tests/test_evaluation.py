"""
Tests for the evaluation module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from src.evaluation import UnifiedEvaluator, BenchmarkConfig, EvaluationResult
from src.evaluation.metrics import NLUMetrics, NLGMetrics
from src.evaluation.task_loaders import TaskLoader, GLUETaskLoader


class TestBenchmarkConfig:
    """Test BenchmarkConfig class."""
    
    def test_benchmark_config_creation(self):
        """Test creating a BenchmarkConfig."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="Test benchmark",
            version="1.0",
            tasks=[{"name": "test_task", "metric": "accuracy"}],
            dataset={"name": "test_dataset"},
            evaluation={"batch_size": 32},
            metrics=["accuracy"],
            baseline={"model": "test_model"},
            target_scores={"test_task": 0.9}
        )
        
        assert config.name == "test_benchmark"
        assert config.description == "Test benchmark"
        assert config.version == "1.0"
        assert len(config.tasks) == 1
        assert config.tasks[0]["name"] == "test_task"
        assert "accuracy" in config.metrics
        assert config.baseline["model"] == "test_model"
        assert config.target_scores["test_task"] == 0.9


class TestEvaluationResult:
    """Test EvaluationResult class."""
    
    def test_evaluation_result_creation(self):
        """Test creating an EvaluationResult."""
        result = EvaluationResult(
            task="test_task",
            metric="accuracy",
            score=0.85,
            target_score=0.90,
            improvement=-0.05,
            confidence_interval=(0.80, 0.90)
        )
        
        assert result.task == "test_task"
        assert result.metric == "accuracy"
        assert result.score == 0.85
        assert result.target_score == 0.90
        assert result.improvement == -0.05
        assert result.confidence_interval == (0.80, 0.90)


class TestNLUMetrics:
    """Test NLU metrics computation."""
    
    def test_accuracy_computation(self):
        """Test accuracy computation."""
        metrics = NLUMetrics()
        
        predictions = [0, 1, 1, 0, 1]
        labels = [0, 1, 0, 0, 1]
        
        accuracy = metrics._compute_accuracy(predictions, labels)
        expected_accuracy = 0.6  # 3 out of 5 correct
        
        assert abs(accuracy - expected_accuracy) < 1e-6
    
    def test_f1_score_computation(self):
        """Test F1 score computation."""
        metrics = NLUMetrics()
        
        predictions = [0, 1, 1, 0, 1]
        labels = [0, 1, 0, 0, 1]
        
        f1 = metrics._compute_f1_score(predictions, labels)
        assert 0 <= f1 <= 1
    
    def test_matthews_correlation_computation(self):
        """Test Matthews correlation computation."""
        metrics = NLUMetrics()
        
        predictions = [0, 1, 1, 0, 1]
        labels = [0, 1, 0, 0, 1]
        
        mcc = metrics._compute_matthews_correlation(predictions, labels)
        assert -1 <= mcc <= 1


class TestNLGMetrics:
    """Test NLG metrics computation."""
    
    def test_accuracy_computation(self):
        """Test accuracy computation for NLG tasks."""
        metrics = NLGMetrics()
        
        predictions = [0, 1, 1, 0, 1]
        labels = [0, 1, 0, 0, 1]
        
        accuracy = metrics._compute_accuracy(predictions, labels)
        expected_accuracy = 0.6  # 3 out of 5 correct
        
        assert abs(accuracy - expected_accuracy) < 1e-6
    
    def test_perplexity_computation(self):
        """Test perplexity computation."""
        metrics = NLGMetrics()
        
        # Create dummy logits and labels
        logits = torch.randn(5, 10)  # 5 samples, 10 classes
        labels = torch.randint(0, 10, (5,))
        
        # Convert to lists for the method
        predictions = [logits[i] for i in range(len(logits))]
        labels_list = labels.tolist()
        
        perplexity = metrics._compute_perplexity(predictions, labels_list)
        
        assert perplexity > 0
        assert not np.isnan(perplexity)
        assert not np.isinf(perplexity)


class TestUnifiedEvaluator:
    """Test UnifiedEvaluator class."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="Test benchmark",
            version="1.0",
            tasks=[{"name": "cola", "metric": "matthews_correlation"}],
            dataset={"name": "glue"},
            evaluation={"batch_size": 32},
            metrics=["matthews_correlation"]
        )
        
        evaluator = UnifiedEvaluator(config)
        
        assert evaluator.config == config
        assert "cola" in evaluator.task_loaders
        assert "matthews_correlation" in evaluator.metric_computers
    
    def test_get_target_score(self):
        """Test getting target score."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="Test benchmark",
            version="1.0",
            tasks=[{"name": "cola", "metric": "matthews_correlation"}],
            dataset={"name": "glue"},
            evaluation={"batch_size": 32},
            metrics=["matthews_correlation"],
            target_scores={"cola_matthews_correlation": 0.5}
        )
        
        evaluator = UnifiedEvaluator(config)
        
        # Test with target_scores
        score = evaluator._get_target_score("cola", "matthews_correlation")
        assert score == 0.5
        
        # Test with baseline scores
        config.target_scores = None
        config.baseline = {"scores": {"cola": 0.6}}
        evaluator = UnifiedEvaluator(config)
        
        score = evaluator._get_target_score("cola", "matthews_correlation")
        assert score == 0.6
        
        # Test with no target score
        config.baseline = None
        evaluator = UnifiedEvaluator(config)
        
        score = evaluator._get_target_score("cola", "matthews_correlation")
        assert score is None
    
    def test_compute_improvement(self):
        """Test improvement computation."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="Test benchmark",
            version="1.0",
            tasks=[],
            dataset={},
            evaluation={},
            metrics=[]
        )
        
        evaluator = UnifiedEvaluator(config)
        
        # Test with target score
        improvement = evaluator._compute_improvement(0.8, 0.7)
        assert improvement == 0.1
        
        # Test with no target score
        improvement = evaluator._compute_improvement(0.8, None)
        assert improvement is None


class TestTaskLoader:
    """Test TaskLoader base class."""
    
    def test_task_loader_initialization(self):
        """Test task loader initialization."""
        loader = TaskLoader("test_task", {"name": "test_dataset"})
        
        assert loader.task_name == "test_task"
        assert loader.dataset_config == {"name": "test_dataset"}
        assert loader.tokenizer is None


if __name__ == "__main__":
    pytest.main([__file__])
