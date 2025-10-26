# Hydra Visual Guide for StackWise Baselines

## 🎯 **What is Hydra?**

Hydra is a configuration management framework that helps you:
- **Organize** your settings in files
- **Combine** different configs together
- **Override** values from command line
- **Run** multiple experiments easily

Think of it like a recipe book where you can mix and match ingredients!

## 📁 **How Configs are Organized**

```
baselines/configs/
├── config.yaml              # 🏠 Main recipe (defaults)
├── model/                   # 🧠 Model ingredients
│   ├── encoder/
│   │   └── bert_family/
│   │       ├── tiny.yaml    # Small model recipe
│   │       ├── base.yaml    # Medium model recipe
│   │       └── large.yaml   # Big model recipe
│   └── decoder/
│       └── gpt2_family/
│           ├── small.yaml   # Small GPT recipe
│           └── medium.yaml  # Medium GPT recipe
├── training/                # 🏃 Training ingredients
│   ├── classical.yaml       # Normal training recipe
│   └── depth_time.yaml      # Special training recipe
├── benchmarks/              # 📊 Evaluation ingredients
│   ├── nlu/
│   │   └── glue.yaml        # GLUE tasks recipe
│   └── nlg/
│       └── language_modeling.yaml  # Language modeling recipe
└── experiments/             # 🧪 Complete recipes
    ├── bert_reproduction/
    │   └── bert_base_glue.yaml  # Complete BERT experiment
    └── depth_time_ablation/
        └── bert_depth_time_vs_classical.yaml  # Comparison experiment
```

## 🔄 **How Hydra Works**

### **Step 1: Main Config (config.yaml)**
```yaml
# @package _global_

defaults:
  - model: encoder/bert_family/base    # Use BERT base model
  - training: classical                # Use classical training
  - benchmark: nlu/glue               # Use GLUE benchmark
  - _self_

# Global settings
experiment:
  name: "my_experiment"
```

**What this means:**
- Load BERT base model config
- Load classical training config  
- Load GLUE benchmark config
- Add any global settings

### **Step 2: Individual Configs**

**Model Config (model/encoder/bert_family/base.yaml):**
```yaml
# @package model

d_model: 768
n_heads: 12
n_layers: 12
vocab_size: 30522
```

**Training Config (training/classical.yaml):**
```yaml
# @package training

batch_size: 16
learning_rate: 2e-5
max_steps: 10000
```

**Benchmark Config (benchmarks/nlu/glue.yaml):**
```yaml
# @package benchmark

name: "glue"
tasks: [cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli]
```

### **Step 3: Hydra Combines Everything**

When you run:
```bash
python scripts/train.py --config-name=bert_base_glue
```

Hydra creates a final config like this:
```yaml
experiment:
  name: "my_experiment"

model:
  d_model: 768
  n_heads: 12
  n_layers: 12
  vocab_size: 30522

training:
  batch_size: 16
  learning_rate: 2e-5
  max_steps: 10000

benchmark:
  name: "glue"
  tasks: [cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli]
```

## 🎮 **How to Use It**

### **Method 1: Use Complete Experiment**
```bash
# Use a pre-made complete experiment
python scripts/train.py --config-name=experiments/bert_reproduction/bert_base_glue
```

### **Method 2: Mix and Match**
```bash
# Use BERT base model with depth-time training
python scripts/train.py model=encoder/bert_family/base training=depth_time

# Use GPT-2 small with classical training
python scripts/train.py model=decoder/gpt2_family/small training=classical
```

### **Method 3: Override Values**
```bash
# Change specific values
python scripts/train.py model=encoder/bert_family/base model.d_model=512

# Change multiple values
python scripts/train.py model=encoder/bert_family/base model.d_model=512 training.batch_size=32
```

### **Method 4: Run Multiple Experiments**
```bash
# Try different model sizes
python scripts/train.py --multirun model=encoder/bert_family/tiny,encoder/bert_family/base,encoder/bert_family/large

# Try different learning rates
python scripts/train.py --multirun training.learning_rate=1e-5,2e-5,5e-5
```

## 🧪 **Practical Examples**

### **Example 1: Quick Test**
```bash
# Train a tiny model quickly
python scripts/train.py model=encoder/bert_family/tiny training.batch_size=4 training.max_steps=100
```

### **Example 2: Compare Training Methods**
```bash
# Compare classical vs depth-time training
python scripts/train.py --multirun training=classical,depth_time
```

### **Example 3: Custom Experiment**
```bash
# Create your own experiment by mixing components
python scripts/train.py \
  model=encoder/bert_family/base \
  training=classical \
  benchmark=nlu/glue \
  training.batch_size=8 \
  training.learning_rate=1e-4
```

## 🔍 **Understanding the Commands**

### **Command Structure:**
```bash
python scripts/train.py [OPTIONS]
```

### **Options:**
- `--config-name=NAME` - Use a complete experiment config
- `model=MODEL_NAME` - Use a specific model config
- `training=TRAINING_NAME` - Use a specific training config
- `benchmark=BENCHMARK_NAME` - Use a specific benchmark config
- `KEY=VALUE` - Override any config value
- `--multirun` - Run multiple experiments

### **Examples:**
```bash
# Use complete experiment
--config-name=experiments/bert_reproduction/bert_base_glue

# Mix components
model=encoder/bert_family/base training=classical

# Override values
model.d_model=512 training.batch_size=16

# Multiple runs
--multirun model=encoder/bert_family/tiny,encoder/bert_family/base
```

## 🎯 **Key Concepts**

### **1. Defaults List**
```yaml
defaults:
  - model: encoder/bert_family/base
  - training: classical
  - benchmark: nlu/glue
```
**Means:** Load these configs and combine them

### **2. Package Directives**
```yaml
# @package model
d_model: 768
```
**Means:** Put this config under the "model" section

### **3. Command Line Overrides**
```bash
model.d_model=512
```
**Means:** Change model.d_model to 512

### **4. Multi-run**
```bash
--multirun model=encoder/bert_family/tiny,encoder/bert_family/base
```
**Means:** Run the experiment twice, once with each model

## 🚀 **Getting Started**

### **Step 1: Look at the Configs**
```bash
# See what's available
ls baselines/configs/
ls baselines/configs/model/
ls baselines/configs/training/
ls baselines/configs/experiments/
```

### **Step 2: Try Simple Commands**
```bash
# Start with a simple model
python scripts/train.py model=encoder/bert_family/tiny

# Try different training
python scripts/train.py model=encoder/bert_family/tiny training=depth_time

# Override some values
python scripts/train.py model=encoder/bert_family/tiny training.batch_size=4
```

### **Step 3: Use Complete Experiments**
```bash
# Use a pre-made experiment
python scripts/train.py --config-name=experiments/bert_reproduction/bert_base_glue
```

### **Step 4: Create Your Own**
```bash
# Mix and match to create your own experiment
python scripts/train.py \
  model=encoder/bert_family/base \
  training=depth_time \
  benchmark=nlu/glue \
  training.batch_size=8
```

## 💡 **Tips**

1. **Start Simple**: Begin with basic commands
2. **Look at Examples**: Check the experiment configs
3. **Use Overrides**: Change values with `key=value`
4. **Try Multi-run**: Run multiple experiments at once
5. **Check Output**: Look at the generated configs in the output directory

## 🆘 **Common Issues**

### **Problem**: "Config not found"
**Solution**: Check the path exists in `baselines/configs/`

### **Problem**: "Invalid override"
**Solution**: Check the key exists in the config

### **Problem**: "Module not found"
**Solution**: Make sure you're in the right directory and have activated the virtual environment

Remember: Hydra is just a way to organize and manage your experiment settings! 🎉
