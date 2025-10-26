# Hydra Quick Reference for StackWise Baselines

## 🎯 **What is Hydra?**
Hydra is a configuration management tool that helps you organize and manage experiment settings using YAML files instead of hardcoding them in Python.

## 🚀 **Quick Start Commands**

### **Basic Usage**
```bash
# Use a complete experiment
python scripts/train.py --config-name=experiments/bert_reproduction/bert_base_glue

# Mix and match components
python scripts/train.py model=encoder/bert_family/base training=classical

# Override specific values
python scripts/train.py model=encoder/bert_family/base model.d_model=512

# Run multiple experiments
python scripts/train.py --multirun model=encoder/bert_family/tiny,encoder/bert_family/base
```

### **Common Patterns**
```bash
# Try different models
python scripts/train.py model=encoder/bert_family/tiny
python scripts/train.py model=encoder/bert_family/base
python scripts/train.py model=encoder/bert_family/large

# Try different training methods
python scripts/train.py training=classical
python scripts/train.py training=depth_time

# Try different benchmarks
python scripts/train.py benchmark=nlu/glue
python scripts/train.py benchmark=nlg/language_modeling

# Combine components
python scripts/train.py model=encoder/bert_family/base training=depth_time benchmark=nlu/glue
```

## 📁 **Config Structure**

```
baselines/configs/
├── config.yaml                    # Main config
├── model/
│   ├── encoder/bert_family/       # BERT models
│   └── decoder/gpt2_family/       # GPT-2 models
├── training/
│   ├── classical.yaml            # Normal training
│   └── depth_time.yaml           # Special training
├── benchmarks/
│   ├── nlu/glue.yaml             # GLUE tasks
│   └── nlg/language_modeling.yaml # Language modeling
└── experiments/
    ├── bert_reproduction/         # BERT experiments
    └── depth_time_ablation/       # Comparison experiments
```

## 🎮 **Command Line Overrides**

### **Override Model Settings**
```bash
model.d_model=512
model.n_heads=8
model.n_layers=6
model.vocab_size=50000
```

### **Override Training Settings**
```bash
training.batch_size=32
training.learning_rate=1e-4
training.max_steps=5000
training.optimizer=AdamW
```

### **Override Benchmark Settings**
```bash
benchmark.tasks=[cola,sst2,mnli]
benchmark.dataset.max_length=256
```

### **Override Multiple Values**
```bash
model.d_model=512 training.batch_size=32 benchmark.tasks=[cola,sst2]
```

## 🔄 **Multi-Run Experiments**

### **Model Size Sweep**
```bash
python scripts/train.py --multirun model=encoder/bert_family/tiny,encoder/bert_family/base,encoder/bert_family/large
```

### **Learning Rate Sweep**
```bash
python scripts/train.py --multirun training.learning_rate=1e-5,2e-5,5e-5
```

### **Training Method Comparison**
```bash
python scripts/train.py --multirun training=classical,depth_time
```

### **Combined Sweep**
```bash
python scripts/train.py --multirun model=encoder/bert_family/tiny,encoder/bert_family/base training=classical,depth_time
```

## 📋 **Config File Examples**

### **Main Config (config.yaml)**
```yaml
defaults:
  - model: encoder/bert_family/base
  - training: classical
  - benchmark: nlu/glue
  - _self_

experiment:
  name: "default_experiment"
```

### **Model Config (model/encoder/bert_family/tiny.yaml)**
```yaml
# @package model

d_model: 312
n_heads: 12
n_layers: 4
vocab_size: 30522
```

### **Training Config (training/classical.yaml)**
```yaml
# @package training

batch_size: 16
learning_rate: 2e-5
max_steps: 10000
optimizer: "AdamW"
```

## 🎯 **Key Concepts**

### **1. Defaults List**
```yaml
defaults:
  - model: encoder/bert_family/base
  - training: classical
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

## 🛠️ **Debugging Commands**

### **Print Resolved Config**
```bash
python scripts/train.py --config-name=bert_base_glue --cfg job
```

### **List Available Configs**
```bash
python scripts/train.py --config-path=configs --config-name=config --help
```

### **Dry Run (no execution)**
```bash
python scripts/train.py --config-name=bert_base_glue --cfg job --resolve
```

## 💡 **Tips and Tricks**

### **1. Start Simple**
```bash
# Start with basic commands
python scripts/train.py model=encoder/bert_family/tiny
```

### **2. Look at Configs**
```bash
# Explore available configs
ls baselines/configs/
cat baselines/configs/config.yaml
cat baselines/configs/model/encoder/bert_family/tiny.yaml
```

### **3. Use Overrides**
```bash
# Change values without editing files
python scripts/train.py model=encoder/bert_family/tiny training.batch_size=4
```

### **4. Run Multiple Experiments**
```bash
# Compare different settings
python scripts/train.py --multirun model=encoder/bert_family/tiny,encoder/bert_family/base
```

### **5. Create Your Own**
```bash
# Mix and match to create custom experiments
python scripts/train.py model=encoder/bert_family/base training=depth_time benchmark=nlu/glue
```

## 🆘 **Common Issues**

### **Problem**: "Config not found"
**Solution**: Check the path exists in `baselines/configs/`

### **Problem**: "Invalid override"
**Solution**: Check the key exists in the config

### **Problem**: "Module not found"
**Solution**: Make sure you're in the right directory and have activated the virtual environment

### **Problem**: "No such file or directory"
**Solution**: Run from the `baselines/` directory

## 🎉 **Remember**

- **Hydra organizes** your settings in YAML files
- **Command line overrides** let you change values easily
- **Multi-run** lets you run multiple experiments
- **Start simple** and build up complexity
- **Look at examples** to understand patterns

Hydra makes managing complex experiments easy! 🚀
