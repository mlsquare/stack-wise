"""
Task-specific data loaders for evaluation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any, Dict, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TaskData:
    """Container for task data."""
    dataset: Dataset
    dataloader: DataLoader
    labels: List[Any]
    task_name: str
    num_samples: int


class TaskLoader(ABC):
    """Base class for task data loaders."""
    
    def __init__(self, task_name: str, dataset_config: Dict[str, Any]):
        self.task_name = task_name
        self.dataset_config = dataset_config
        self.tokenizer = None
        
    @abstractmethod
    def load_data(self) -> TaskData:
        """Load task data."""
        pass
    
    def _load_tokenizer(self, model_name: str) -> None:
        """Load tokenizer for the task."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Could not load tokenizer {model_name}: {e}")
            self.tokenizer = None


class GLUETaskLoader(TaskLoader):
    """Data loader for GLUE tasks."""
    
    def __init__(self, task_name: str, dataset_config: Dict[str, Any]):
        super().__init__(task_name, dataset_config)
        self.task_name = task_name
        
    def load_data(self) -> TaskData:
        """Load GLUE task data."""
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("glue", self.task_name, split="validation")
            
            # Load tokenizer
            model_name = self.dataset_config.get("model_name", "bert-base-uncased")
            self._load_tokenizer(model_name)
            
            if self.tokenizer is None:
                raise ValueError(f"Could not load tokenizer for {self.task_name}")
            
            # Preprocess data
            def preprocess_function(examples):
                # Get text columns based on task
                if self.task_name in ["cola"]:
                    texts = examples["sentence"]
                elif self.task_name in ["sst2"]:
                    texts = examples["sentence"]
                elif self.task_name in ["mrpc"]:
                    texts = (examples["sentence1"], examples["sentence2"])
                elif self.task_name in ["stsb"]:
                    texts = (examples["sentence1"], examples["sentence2"])
                elif self.task_name in ["qqp"]:
                    texts = (examples["question1"], examples["question2"])
                elif self.task_name in ["mnli"]:
                    texts = (examples["premise"], examples["hypothesis"])
                elif self.task_name in ["qnli"]:
                    texts = (examples["question"], examples["sentence"])
                elif self.task_name in ["rte"]:
                    texts = (examples["sentence1"], examples["sentence2"])
                elif self.task_name in ["wnli"]:
                    texts = (examples["sentence1"], examples["sentence2"])
                else:
                    texts = examples["sentence"]
                
                # Tokenize
                if isinstance(texts, tuple):
                    # Sentence pair tasks
                    tokenized = self.tokenizer(
                        texts[0], texts[1],
                        truncation=True,
                        padding=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                else:
                    # Single sentence tasks
                    tokenized = self.tokenizer(
                        texts,
                        truncation=True,
                        padding=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                
                return tokenized
            
            # Apply preprocessing
            processed_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Create PyTorch dataset
            class GLUEDataset(Dataset):
                def __init__(self, processed_data):
                    self.data = processed_data
                    
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    item = self.data[idx]
                    return {
                        'input_ids': item['input_ids'],
                        'attention_mask': item['attention_mask'],
                        'labels': item['labels'] if 'labels' in item else 0
                    }
            
            torch_dataset = GLUEDataset(processed_dataset)
            
            # Create data loader
            dataloader = DataLoader(
                torch_dataset,
                batch_size=self.dataset_config.get("batch_size", 32),
                shuffle=False,
                num_workers=self.dataset_config.get("num_workers", 4)
            )
            
            # Extract labels
            labels = [item["labels"] for item in processed_dataset]
            
            return TaskData(
                dataset=torch_dataset,
                dataloader=dataloader,
                labels=labels,
                task_name=self.task_name,
                num_samples=len(torch_dataset)
            )
            
        except Exception as e:
            logger.error(f"Error loading GLUE task {self.task_name}: {e}")
            raise


class LanguageModelingTaskLoader(TaskLoader):
    """Data loader for language modeling tasks."""
    
    def __init__(self, task_name: str, dataset_config: Dict[str, Any]):
        super().__init__(task_name, dataset_config)
        self.task_name = task_name
        
    def load_data(self) -> TaskData:
        """Load language modeling task data."""
        try:
            # Load dataset based on task
            if self.task_name == "wikitext103":
                dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            elif self.task_name == "ptb":
                dataset = load_dataset("ptb_text_only", split="test")
            else:
                # For other tasks, try to load from a generic source
                dataset = load_dataset(self.task_name, split="test")
            
            # Load tokenizer
            model_name = self.dataset_config.get("model_name", "gpt2")
            self._load_tokenizer(model_name)
            
            if self.tokenizer is None:
                raise ValueError(f"Could not load tokenizer for {self.task_name}")
            
            # Preprocess data
            def preprocess_function(examples):
                # Get text column
                text_column = self.dataset_config.get("text_column", "text")
                texts = examples[text_column]
                
                # Tokenize
                tokenized = self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=self.dataset_config.get("max_length", 1024),
                    return_tensors="pt"
                )
                
                # For language modeling, labels are the same as input_ids
                tokenized["labels"] = tokenized["input_ids"].clone()
                
                return tokenized
            
            # Apply preprocessing
            processed_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Create PyTorch dataset
            class LanguageModelingDataset(Dataset):
                def __init__(self, processed_data):
                    self.data = processed_data
                    
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    item = self.data[idx]
                    return {
                        'input_ids': item['input_ids'],
                        'attention_mask': item['attention_mask'],
                        'labels': item['labels']
                    }
            
            torch_dataset = LanguageModelingDataset(processed_dataset)
            
            # Create data loader
            dataloader = DataLoader(
                torch_dataset,
                batch_size=self.dataset_config.get("batch_size", 16),
                shuffle=False,
                num_workers=self.dataset_config.get("num_workers", 4)
            )
            
            # Extract labels
            labels = [item["labels"] for item in processed_dataset]
            
            return TaskData(
                dataset=torch_dataset,
                dataloader=dataloader,
                labels=labels,
                task_name=self.task_name,
                num_samples=len(torch_dataset)
            )
            
        except Exception as e:
            logger.error(f"Error loading language modeling task {self.task_name}: {e}")
            raise
