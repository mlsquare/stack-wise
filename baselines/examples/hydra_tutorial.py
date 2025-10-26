#!/usr/bin/env python3
"""
Hydra Framework Tutorial for StackWise Baselines

This tutorial explains how to use Hydra step by step with simple examples.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def tutorial_1_basic_concepts():
    """Tutorial 1: Basic Hydra Concepts"""
    print("=" * 60)
    print("TUTORIAL 1: Basic Hydra Concepts")
    print("=" * 60)
    
    print("""
Hydra is a configuration management framework. Here's what you need to know:

1. CONFIGURATION FILES (.yaml)
   - Store settings in YAML files
   - Organized in a hierarchy
   - Can reference other configs

2. DEFAULTS LIST
   - Tells Hydra which configs to load
   - Example: defaults: [model: bert, training: classical]

3. PACKAGE DIRECTIVES (@package)
   - Tells Hydra where to put config sections
   - Example: @package model

4. COMMAND LINE OVERRIDES
   - Override any config value from command line
   - Example: model.d_model=512

Let's see this in action...
""")

def tutorial_2_simple_config():
    """Tutorial 2: Create a Simple Config"""
    print("=" * 60)
    print("TUTORIAL 2: Create a Simple Config")
    print("=" * 60)
    
    # Create a simple config file
    simple_config = """
# Simple model configuration
model:
  name: "my_model"
  d_model: 128
  n_heads: 4
  n_layers: 2

training:
  batch_size: 8
  learning_rate: 0.001
  max_steps: 100

data:
  dataset_name: "my_dataset"
  num_samples: 1000
"""
    
    config_path = Path(__file__).parent / "simple_config.yaml"
    with open(config_path, 'w') as f:
        f.write(simple_config)
    
    print(f"Created simple config: {config_path}")
    print("""
This config has:
- model: architecture settings
- training: training parameters  
- data: dataset settings

Now let's load it with Hydra...
""")

def tutorial_3_load_config():
    """Tutorial 3: Load Config with Hydra"""
    print("=" * 60)
    print("TUTORIAL 3: Load Config with Hydra")
    print("=" * 60)
    
    try:
        from omegaconf import OmegaConf
        
        # Load the simple config
        config_path = Path(__file__).parent / "simple_config.yaml"
        cfg = OmegaConf.load(config_path)
        
        print("✅ Successfully loaded config!")
        print("\nConfig contents:")
        print(OmegaConf.to_yaml(cfg))
        
        print("\nAccessing values:")
        print(f"Model name: {cfg.model.name}")
        print(f"Model d_model: {cfg.model.d_model}")
        print(f"Training batch_size: {cfg.training.batch_size}")
        print(f"Data num_samples: {cfg.data.num_samples}")
        
        return True
        
    except ImportError:
        print("❌ omegaconf not available")
        print("Install with: pip install omegaconf")
        return False
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def tutorial_4_override_values():
    """Tutorial 4: Override Values"""
    print("=" * 60)
    print("TUTORIAL 4: Override Values")
    print("=" * 60)
    
    try:
        from omegaconf import OmegaConf
        
        config_path = Path(__file__).parent / "simple_config.yaml"
        cfg = OmegaConf.load(config_path)
        
        print("Original config:")
        print(f"  d_model: {cfg.model.d_model}")
        print(f"  batch_size: {cfg.training.batch_size}")
        
        # Override values programmatically
        cfg.model.d_model = 256
        cfg.training.batch_size = 16
        
        print("\nAfter overriding:")
        print(f"  d_model: {cfg.model.d_model}")
        print(f"  batch_size: {cfg.training.batch_size}")
        
        print("""
In Hydra, you can override values from command line:
  python script.py model.d_model=256 training.batch_size=16

Or merge with another config:
  python script.py --config-path=configs --config-name=my_config
""")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def tutorial_5_hierarchical_configs():
    """Tutorial 5: Hierarchical Configs"""
    print("=" * 60)
    print("TUTORIAL 5: Hierarchical Configs")
    print("=" * 60)
    
    # Create a main config that references others
    main_config = """
# @package _global_

defaults:
  - model: bert_tiny
  - training: fast
  - data: small_dataset

# Global settings
experiment:
  name: "my_experiment"
  description: "Learning Hydra"
"""
    
    # Create model config
    model_config = """
# @package model

d_model: 64
n_heads: 2
n_layers: 2
vocab_size: 1000
"""
    
    # Create training config
    training_config = """
# @package training

batch_size: 4
learning_rate: 0.001
max_steps: 50
"""
    
    # Create data config
    data_config = """
# @package data

dataset_name: "toy_data"
num_samples: 100
max_length: 32
"""
    
    # Save configs
    config_dir = Path(__file__).parent / "hierarchical_configs"
    config_dir.mkdir(exist_ok=True)
    
    (config_dir / "config.yaml").write_text(main_config)
    (config_dir / "model" / "bert_tiny.yaml").write_text(model_config)
    (config_dir / "training" / "fast.yaml").write_text(training_config)
    (config_dir / "data" / "small_dataset.yaml").write_text(data_config)
    
    print(f"Created hierarchical configs in: {config_dir}")
    print("""
Structure:
hierarchical_configs/
├── config.yaml          # Main config
├── model/
│   └── bert_tiny.yaml   # Model config
├── training/
│   └── fast.yaml        # Training config
└── data/
    └── small_dataset.yaml # Data config

The main config uses 'defaults' to load the others:
  defaults:
    - model: bert_tiny      # Loads model/bert_tiny.yaml
    - training: fast        # Loads training/fast.yaml
    - data: small_dataset   # Loads data/small_dataset.yaml
""")

def tutorial_6_use_with_hydra():
    """Tutorial 6: Use with @hydra.main"""
    print("=" * 60)
    print("TUTORIAL 6: Use with @hydra.main")
    print("=" * 60)
    
    hydra_script = '''
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="hierarchical_configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Hydra loaded config:")
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Model d_model: {cfg.model.d_model}")
    print(f"Training batch_size: {cfg.training.batch_size}")
    print(f"Data num_samples: {cfg.data.num_samples}")

if __name__ == "__main__":
    main()
'''
    
    script_path = Path(__file__).parent / "hydra_example.py"
    with open(script_path, 'w') as f:
        f.write(hydra_script)
    
    print(f"Created Hydra script: {script_path}")
    print("""
This script uses @hydra.main decorator which:
1. Automatically loads the config
2. Handles command line overrides
3. Sets up logging
4. Manages output directories

Run it with:
  python hydra_example.py
  python hydra_example.py model.d_model=128
  python hydra_example.py training.batch_size=8
""")

def tutorial_7_baselines_usage():
    """Tutorial 7: How to Use with StackWise Baselines"""
    print("=" * 60)
    print("TUTORIAL 7: StackWise Baselines Usage")
    print("=" * 60)
    
    print("""
The StackWise Baselines framework uses Hydra like this:

1. CONFIG STRUCTURE:
   baselines/configs/
   ├── config.yaml              # Main config
   ├── model/
   │   ├── encoder/
   │   │   └── bert_family/
   │   │       ├── tiny.yaml
   │   │       ├── base.yaml
   │   │       └── large.yaml
   │   └── decoder/
   │       └── gpt2_family/
   │           ├── small.yaml
   │           └── medium.yaml
   ├── training/
   │   ├── classical.yaml
   │   └── depth_time.yaml
   ├── benchmarks/
   │   ├── nlu/
   │   │   └── glue.yaml
   │   └── nlg/
   │       └── language_modeling.yaml
   └── experiments/
       ├── bert_reproduction/
       │   └── bert_base_glue.yaml
       └── depth_time_ablation/
           └── bert_depth_time_vs_classical.yaml

2. BASIC USAGE:
   # Use a complete experiment config
   python scripts/train.py --config-name=experiments/bert_reproduction/bert_base_glue
   
   # Mix and match components
   python scripts/train.py model=encoder/bert_family/base training=classical
   
   # Override specific values
   python scripts/train.py model=encoder/bert_family/base model.d_model=512
   
   # Run multiple experiments
   python scripts/train.py --multirun model=encoder/bert_family/base,encoder/bert_family/large

3. COMMAND LINE OVERRIDES:
   # Change model size
   python scripts/train.py model.d_model=512 model.n_heads=8
   
   # Change training parameters
   python scripts/train.py training.batch_size=16 training.learning_rate=2e-5
   
   # Change benchmark tasks
   python scripts/train.py benchmark.tasks=[cola,sst2,mnli]

4. MULTI-RUN EXPERIMENTS:
   # Learning rate sweep
   python scripts/train.py --multirun training.learning_rate=1e-5,2e-5,5e-5
   
   # Model size sweep
   python scripts/train.py --multirun model=encoder/bert_family/tiny,encoder/bert_family/base
   
   # Training regime comparison
   python scripts/train.py --multirun training=classical,depth_time
""")

def tutorial_8_practical_examples():
    """Tutorial 8: Practical Examples"""
    print("=" * 60)
    print("TUTORIAL 8: Practical Examples")
    print("=" * 60)
    
    print("""
Here are practical examples you can run:

1. QUICK START (if you have the baselines set up):
   cd baselines
   source .venv/bin/activate
   
   # Train a tiny BERT model
   uv run python scripts/train.py model=encoder/bert_family/tiny
   
   # Train with different settings
   uv run python scripts/train.py model=encoder/bert_family/tiny training.batch_size=4
   
   # Run a complete experiment
   uv run python scripts/train.py --config-name=experiments/bert_reproduction/bert_base_glue

2. WITHOUT SETUP (using the tutorial files):
   cd baselines/examples
   
   # Load and explore configs
   python hydra_tutorial.py
   
   # Run the simple Hydra example
   python hydra_example.py
   python hydra_example.py model.d_model=256
   python hydra_example.py training.batch_size=16

3. UNDERSTANDING CONFIG STRUCTURE:
   # Look at the config files
   cat baselines/configs/config.yaml
   cat baselines/configs/model/encoder/bert_family/tiny.yaml
   cat baselines/configs/training/classical.yaml
   
   # See how they reference each other
   grep -r "defaults:" baselines/configs/
   grep -r "@package" baselines/configs/

4. DEBUGGING:
   # Print the resolved config
   python scripts/train.py --config-name=bert_base_glue --cfg job
   
   # See what configs are available
   python scripts/train.py --config-path=configs --config-name=config --help
""")

def main():
    """Run all tutorials"""
    print("🎓 HYDRA FRAMEWORK TUTORIAL")
    print("Learn how to use Hydra with StackWise Baselines")
    print()
    
    tutorials = [
        tutorial_1_basic_concepts,
        tutorial_2_simple_config,
        tutorial_3_load_config,
        tutorial_4_override_values,
        tutorial_5_hierarchical_configs,
        tutorial_6_use_with_hydra,
        tutorial_7_baselines_usage,
        tutorial_8_practical_examples,
    ]
    
    for i, tutorial in enumerate(tutorials, 1):
        try:
            tutorial()
            print("\n" + "✅" * 20 + f" TUTORIAL {i} COMPLETE " + "✅" * 20 + "\n")
        except Exception as e:
            print(f"❌ Tutorial {i} failed: {e}")
            print()
    
    print("🎉 ALL TUTORIALS COMPLETE!")
    print("""
Next steps:
1. Try the practical examples above
2. Look at the config files in baselines/configs/
3. Run some experiments with different settings
4. Create your own config files

Remember: Hydra is all about organizing and managing configurations!
""")

if __name__ == "__main__":
    main()
