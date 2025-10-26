"""
Metric computation modules for evaluation.
"""

import torch
import numpy as np
from typing import List, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    pearsonr, spearmanr
)
import math


class MetricComputer:
    """Base class for metric computation."""
    
    def compute(self, predictions: List[Any], labels: List[Any], 
                task_name: Optional[str] = None) -> float:
        """Compute metric score."""
        raise NotImplementedError


class NLUMetrics(MetricComputer):
    """Natural Language Understanding metrics."""
    
    def compute(self, predictions: List[Any], labels: List[Any], 
                task_name: Optional[str] = None) -> float:
        """Compute NLU metric based on task."""
        if task_name in ["cola"]:
            return self._compute_matthews_correlation(predictions, labels)
        elif task_name in ["sst2", "mnli", "qnli", "rte", "wnli"]:
            return self._compute_accuracy(predictions, labels)
        elif task_name in ["mrpc", "qqp"]:
            return self._compute_f1_score(predictions, labels)
        elif task_name in ["stsb"]:
            return self._compute_pearson_correlation(predictions, labels)
        else:
            # Default to accuracy
            return self._compute_accuracy(predictions, labels)
    
    def _compute_accuracy(self, predictions: List[Any], labels: List[Any]) -> float:
        """Compute accuracy score."""
        predictions = np.array(predictions)
        labels = np.array(labels)
        return accuracy_score(labels, predictions)
    
    def _compute_f1_score(self, predictions: List[Any], labels: List[Any]) -> float:
        """Compute F1 score (macro average)."""
        predictions = np.array(predictions)
        labels = np.array(labels)
        return f1_score(labels, predictions, average='macro')
    
    def _compute_matthews_correlation(self, predictions: List[Any], labels: List[Any]) -> float:
        """Compute Matthews correlation coefficient."""
        predictions = np.array(predictions)
        labels = np.array(labels)
        return matthews_corrcoef(labels, predictions)
    
    def _compute_pearson_correlation(self, predictions: List[Any], labels: List[Any]) -> float:
        """Compute Pearson correlation coefficient."""
        predictions = np.array(predictions)
        labels = np.array(labels)
        corr, _ = pearsonr(predictions, labels)
        return corr
    
    def _compute_spearman_correlation(self, predictions: List[Any], labels: List[Any]) -> float:
        """Compute Spearman correlation coefficient."""
        predictions = np.array(predictions)
        labels = np.array(labels)
        corr, _ = spearmanr(predictions, labels)
        return corr


class NLGMetrics(MetricComputer):
    """Natural Language Generation metrics."""
    
    def compute(self, predictions: List[Any], labels: List[Any], 
                task_name: Optional[str] = None) -> float:
        """Compute NLG metric based on task."""
        if task_name in ["wikitext103", "ptb"]:
            return self._compute_perplexity(predictions, labels)
        elif task_name in ["lambada", "hellaswag", "piqa", "winogrande"]:
            return self._compute_accuracy(predictions, labels)
        else:
            # Default to perplexity for language modeling
            return self._compute_perplexity(predictions, labels)
    
    def _compute_accuracy(self, predictions: List[Any], labels: List[Any]) -> float:
        """Compute accuracy score."""
        predictions = np.array(predictions)
        labels = np.array(labels)
        return accuracy_score(labels, predictions)
    
    def _compute_perplexity(self, predictions: List[Any], labels: List[Any]) -> float:
        """Compute perplexity from logits."""
        if isinstance(predictions[0], (list, tuple)):
            # Handle batched predictions
            all_logits = []
            all_labels = []
            for pred, label in zip(predictions, labels):
                all_logits.extend(pred)
                all_labels.extend(label)
            predictions = all_logits
            labels = all_labels
        
        # Convert to tensors if needed
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        
        # Compute cross-entropy loss
        if predictions.dim() > 1:
            # Logits format
            loss = torch.nn.functional.cross_entropy(predictions, labels, reduction='mean')
        else:
            # Already computed loss
            loss = predictions.mean()
        
        # Convert to perplexity
        perplexity = torch.exp(loss).item()
        return perplexity
    
    def _compute_bleu_score(self, predictions: List[Any], labels: List[Any]) -> float:
        """Compute BLEU score."""
        # This would require additional dependencies like nltk or sacrebleu
        # For now, return a placeholder
        return 0.0
    
    def _compute_rouge_score(self, predictions: List[Any], labels: List[Any]) -> float:
        """Compute ROUGE score."""
        # This would require additional dependencies like rouge-score
        # For now, return a placeholder
        return 0.0
