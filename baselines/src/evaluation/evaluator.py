"""
Unified evaluation harness for all model families and benchmark tasks.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import numpy as np
from omegaconf import DictConfig

from .metrics import MetricComputer, NLUMetrics, NLGMetrics
from .task_loaders import TaskLoader, GLUETaskLoader, LanguageModelingTaskLoader

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    name: str
    description: str
    version: str
    tasks: List[Dict[str, Any]]
    dataset: Dict[str, Any]
    evaluation: Dict[str, Any]
    metrics: List[str]
    baseline: Optional[Dict[str, Any]] = None
    target_scores: Optional[Dict[str, float]] = None


@dataclass
class EvaluationResult:
    """Results from benchmark evaluation."""
    task: str
    metric: str
    score: float
    target_score: Optional[float] = None
    improvement: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    predictions: Optional[List[Any]] = None
    labels: Optional[List[Any]] = None


class UnifiedEvaluator:
    """Unified evaluation harness for all model families."""
    
    def __init__(self, config: Union[BenchmarkConfig, DictConfig]):
        """Initialize evaluator with benchmark configuration."""
        self.config = config
        self.task_loaders = self._load_task_loaders()
        self.metric_computers = self._load_metric_computers()
        
    def _load_task_loaders(self) -> Dict[str, TaskLoader]:
        """Load task-specific data loaders."""
        loaders = {}
        
        for task_config in self.config.tasks:
            task_name = task_config["name"]
            
            if task_name in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"]:
                loaders[task_name] = GLUETaskLoader(task_name, self.config.dataset)
            elif task_name in ["wikitext103", "ptb", "lambada", "hellaswag", "piqa", "winogrande"]:
                loaders[task_name] = LanguageModelingTaskLoader(task_name, self.config.dataset)
            else:
                logger.warning(f"Unknown task: {task_name}, using generic loader")
                loaders[task_name] = TaskLoader(task_name, self.config.dataset)
                
        return loaders
    
    def _load_metric_computers(self) -> Dict[str, MetricComputer]:
        """Load metric computation modules."""
        computers = {}
        
        for metric in self.config.metrics:
            if metric in ["accuracy", "f1", "matthews_correlation", "pearson_correlation", "spearman_correlation"]:
                computers[metric] = NLUMetrics()
            elif metric in ["perplexity", "bleu", "rouge"]:
                computers[metric] = NLGMetrics()
            else:
                logger.warning(f"Unknown metric: {metric}, using generic computer")
                computers[metric] = MetricComputer()
                
        return computers
    
    def evaluate_model(self, model: torch.nn.Module, tokenizer: Any, 
                      device: str = "cuda") -> Dict[str, EvaluationResult]:
        """Evaluate model on all configured tasks."""
        results = {}
        model.eval()
        
        with torch.no_grad():
            for task_config in self.config.tasks:
                task_name = task_config["name"]
                logger.info(f"Evaluating task: {task_name}")
                
                try:
                    # Load task data
                    task_loader = self.task_loaders[task_name]
                    task_data = task_loader.load_data()
                    
                    # Run inference
                    predictions = self._run_inference(model, tokenizer, task_data, device)
                    
                    # Compute metrics
                    for metric in self.config.metrics:
                        if metric in self.metric_computers:
                            score = self.metric_computers[metric].compute(
                                predictions, task_data.labels, task_name
                            )
                            
                            target_score = self._get_target_score(task_name, metric)
                            improvement = self._compute_improvement(score, target_score)
                            
                            results[f"{task_name}_{metric}"] = EvaluationResult(
                                task=task_name,
                                metric=metric,
                                score=score,
                                target_score=target_score,
                                improvement=improvement,
                                predictions=predictions,
                                labels=task_data.labels
                            )
                            
                            logger.info(f"{task_name}_{metric}: {score:.4f} (target: {target_score})")
                            
                except Exception as e:
                    logger.error(f"Error evaluating task {task_name}: {e}")
                    continue
                    
        return results
    
    def _run_inference(self, model: torch.nn.Module, tokenizer: Any, 
                      task_data: Any, device: str) -> List[Any]:
        """Run inference on task data."""
        predictions = []
        
        for batch in task_data.dataloader:
            # Move batch to device
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            
            # Get model inputs
            if hasattr(batch, 'input_ids'):
                input_ids = batch.input_ids
                attention_mask = getattr(batch, 'attention_mask', None)
            else:
                # Handle different batch formats
                input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
                attention_mask = None
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Extract predictions
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            
            # Convert to predictions
            if logits.dim() > 2:
                # Language modeling: use last token logits
                logits = logits[:, -1, :]
            
            if logits.dim() == 2 and logits.size(1) > 1:
                # Classification: get predicted class
                preds = torch.argmax(logits, dim=-1)
            else:
                # Regression or single value
                preds = logits.squeeze(-1) if logits.dim() > 1 else logits
            
            predictions.extend(preds.cpu().numpy())
            
        return predictions
    
    def _get_target_score(self, task_name: str, metric: str) -> Optional[float]:
        """Get target score for task and metric."""
        if self.config.target_scores:
            key = f"{task_name}_{metric}"
            return self.config.target_scores.get(key)
        
        if self.config.baseline and "scores" in self.config.baseline:
            return self.config.baseline["scores"].get(task_name)
            
        return None
    
    def _compute_improvement(self, score: float, target_score: Optional[float]) -> Optional[float]:
        """Compute improvement over target score."""
        if target_score is None:
            return None
            
        return score - target_score
    
    def save_results(self, results: Dict[str, EvaluationResult], 
                    output_dir: Union[str, Path]) -> None:
        """Save evaluation results to file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        results_dict = {}
        for key, result in results.items():
            results_dict[key] = {
                "task": result.task,
                "metric": result.metric,
                "score": result.score,
                "target_score": result.target_score,
                "improvement": result.improvement,
                "confidence_interval": result.confidence_interval,
            }
        
        # Save results
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def generate_report(self, results: Dict[str, EvaluationResult]) -> str:
        """Generate markdown report from results."""
        report = ["# Evaluation Report", ""]
        
        # Summary table
        report.append("## Summary")
        report.append("| Task | Metric | Score | Target | Improvement |")
        report.append("|------|--------|-------|--------|-------------|")
        
        for key, result in results.items():
            target_str = f"{result.target_score:.4f}" if result.target_score else "N/A"
            improvement_str = f"{result.improvement:+.4f}" if result.improvement else "N/A"
            report.append(f"| {result.task} | {result.metric} | {result.score:.4f} | {target_str} | {improvement_str} |")
        
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        for key, result in results.items():
            report.append(f"### {result.task} - {result.metric}")
            report.append(f"- **Score**: {result.score:.4f}")
            if result.target_score:
                report.append(f"- **Target**: {result.target_score:.4f}")
            if result.improvement:
                report.append(f"- **Improvement**: {result.improvement:+.4f}")
            report.append("")
        
        return "\n".join(report)
