# 🦙 TinyLLaMA - CLM Example

A minimal LLaMA-style model demonstrating **Causal Language Modeling (CLM)** with progressive training using the Stack-Wise framework.

## 🎯 Overview

TinyLLaMA contrasts with TinyBERT by using:
- **CLM (Causal Language Modeling)** instead of MLM (Masked Language Modeling)
- **GPT-style causal attention** instead of BERT-style bidirectional attention
- **Next-token prediction** instead of masked token prediction
- **Left padding** for autoregressive generation
- **GQA (Grouped Query Attention)** for efficiency

## 🏗️ Architecture

- **Model Size**: 256 d_model, 6 stacks, 1 block per stack
- **Attention**: 8 heads with 4 key-value heads (2:1 GQA ratio)
- **Feed-Forward**: 1024 d_ff with SwiGLU activation
- **Vocabulary**: 32,000 tokens
- **Sequence Length**: 128 tokens

## 🚀 Quick Start

### 1. Train the Model
```bash
cd examples/tiny_llama
python train_tiny_llama.py
```

### 2. Evaluate the Model
```bash
python evaluate_tiny_llama.py --model tiny_llama_outputs/tiny_llama_rack.pt --config tiny_llama_config.yaml
```

### 3. Generate Text
```bash
# Single generation
python use_tiny_llama.py --model tiny_llama_outputs/tiny_llama_rack.pt --config tiny_llama_config.yaml --prompt "The quick brown fox"

# Interactive mode
python use_tiny_llama.py --model tiny_llama_outputs/tiny_llama_rack.pt --config tiny_llama_config.yaml --interactive
```

## 📊 Training Configuration

### Progressive Training
- **Strategy**: Progressive with frozen trunk
- **Target Stacks**: 6 stacks
- **Training Objective**: CLM (next-token prediction)
- **Activation Caching**: Enabled for efficiency

### CLM-Specific Settings
- **No Masking**: `min_mask_fraction: 0.0`, `max_mask_fraction: 0.0`
- **Left Padding**: For causal attention patterns
- **Causal Attention**: GPT-style autoregressive attention
- **Next-Token Loss**: Cross-entropy on shifted targets

## 🔧 Key Differences from TinyBERT

| Aspect | TinyBERT (MLM) | TinyLLaMA (CLM) |
|--------|----------------|-----------------|
| **Task** | Masked Language Modeling | Causal Language Modeling |
| **Attention** | Bidirectional (BERT-style) | Causal (GPT-style) |
| **Masking** | Random token masking | No masking |
| **Padding** | Right padding | Left padding |
| **Loss** | Masked token prediction | Next-token prediction |
| **Generation** | Fill-in-the-blank | Autoregressive text generation |
| **GQA** | No (4:4 heads) | Yes (8:4 heads) |
| **Model Size** | 128 d_model, 4 stacks | 256 d_model, 6 stacks |

## 📈 Expected Results

### Training Progress
- **Stack 0**: Initial learning of basic patterns
- **Stack 1-5**: Progressive refinement of language modeling
- **Loss Decrease**: Each stack should show improved next-token prediction

### Evaluation Metrics
- **Perplexity**: Lower is better (measures prediction uncertainty)
- **Next-Token Accuracy**: Percentage of correct next-token predictions
- **Generation Quality**: Coherence of generated text continuations

## 🎮 Interactive Usage

The interactive mode supports various generation parameters:

```bash
# Basic generation
Enter prompt: Hello world

# With parameters
Enter prompt: The future of AI is with max_length=30, temperature=0.8

# Help
Enter prompt: help
```

### Generation Parameters
- **max_length**: Maximum tokens to generate (default: 50)
- **temperature**: Randomness level 0.1-2.0 (default: 1.0)
- **top_k**: Keep only top k tokens (optional)
- **top_p**: Nucleus sampling threshold (optional)

## 📁 File Structure

```
tiny_llama/
├── tiny_llama_config.yaml    # Configuration file
├── train_tiny_llama.py       # Training script
├── evaluate_tiny_llama.py    # Evaluation script
├── use_tiny_llama.py         # Text generation script
├── toy_dataset.py            # CLM dataset generator
└── README.md                 # This file
```

## 🔬 Technical Details

### CLM Dataset Generation
- **Input**: Random token sequences
- **Target**: Input shifted by 1 position (next-token prediction)
- **No Masking**: All tokens are used for training
- **Left Padding**: Maintains causal attention patterns

### Progressive Training Benefits
- **Efficiency**: Each stack builds on frozen previous knowledge
- **Stability**: Gradual complexity increase
- **Memory**: Activation caching reduces recomputation
- **Quality**: Better convergence than end-to-end training

### GQA Implementation
- **8 Query Heads**: Full attention computation
- **4 Key-Value Heads**: Shared across query heads
- **2:1 Ratio**: Balances quality and efficiency
- **Memory Savings**: ~50% reduction in attention parameters

## 🎯 Use Cases

1. **Language Modeling**: Next-token prediction for text generation
2. **Text Completion**: Autoregressive text continuation
3. **Progressive Training**: Demonstrating stack-wise learning
4. **CLM vs MLM**: Comparing causal vs masked language modeling
5. **GQA Efficiency**: Testing grouped query attention benefits

## 🚨 Limitations

- **Simple Tokenization**: Uses hash-based tokenization for demo
- **Small Model**: Limited capacity for complex language understanding
- **Toy Dataset**: Random sequences, not real text
- **No Pretraining**: Starts from scratch, not from pretrained weights

## 🔄 Comparison with TinyBERT

Run both examples to see the differences:

```bash
# Train TinyBERT (MLM)
cd ../tiny_bert
python train_tiny_bert.py

# Train TinyLLaMA (CLM)
cd ../tiny_llama
python train_tiny_llama.py

# Compare results
echo "TinyBERT (MLM) vs TinyLLaMA (CLM) comparison"
```

This demonstrates the flexibility of the Stack-Wise framework for different language modeling objectives!
