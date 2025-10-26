#!/usr/bin/env python3
"""
Simple Hydra Demo - Run This to Learn!

This is a hands-on demo that shows exactly how Hydra works.
You can run this without any setup to understand the concepts.
"""

import os
import sys
from pathlib import Path

def create_demo_configs():
    """Create simple demo configs to show how Hydra works."""
    
    # Create demo directory
    demo_dir = Path(__file__).parent / "hydra_demo"
    demo_dir.mkdir(exist_ok=True)
    
    # Main config
    main_config = """
# @package _global_

defaults:
  - model: tiny_bert
  - training: fast
  - data: small

# Global settings
experiment:
  name: "hydra_demo"
  description: "Learning how Hydra works"
"""
    
    # Model config
    model_config = """
# @package model

name: "tiny_bert"
d_model: 64
n_heads: 2
n_layers: 2
vocab_size: 1000
"""
    
    # Training config
    training_config = """
# @package training

batch_size: 4
learning_rate: 0.001
max_steps: 50
optimizer: "AdamW"
"""
    
    # Data config
    data_config = """
# @package data

dataset_name: "demo_data"
num_samples: 100
max_length: 32
"""
    
    # Save all configs
    (demo_dir / "config.yaml").write_text(main_config)
    (demo_dir / "model" / "tiny_bert.yaml").write_text(model_config)
    (demo_dir / "training" / "fast.yaml").write_text(training_config)
    (demo_dir / "data" / "small.yaml").write_text(data_config)
    
    print(f"✅ Created demo configs in: {demo_dir}")
    return demo_dir

def demo_1_load_config():
    """Demo 1: Load a config with Hydra"""
    print("\n" + "="*50)
    print("DEMO 1: Loading Config with Hydra")
    print("="*50)
    
    try:
        from omegaconf import OmegaConf
        
        # Load the main config
        demo_dir = create_demo_configs()
        config_path = demo_dir / "config.yaml"
        cfg = OmegaConf.load(config_path)
        
        print("✅ Successfully loaded config!")
        print("\nConfig contents:")
        print(OmegaConf.to_yaml(cfg))
        
        print("\nAccessing values:")
        print(f"  Experiment name: {cfg.experiment.name}")
        print(f"  Model name: {cfg.model.name}")
        print(f"  Model d_model: {cfg.model.d_model}")
        print(f"  Training batch_size: {cfg.training.batch_size}")
        print(f"  Data num_samples: {cfg.data.num_samples}")
        
        return True
        
    except ImportError:
        print("❌ omegaconf not available")
        print("Install with: pip install omegaconf")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_2_override_values():
    """Demo 2: Override values"""
    print("\n" + "="*50)
    print("DEMO 2: Overriding Values")
    print("="*50)
    
    try:
        from omegaconf import OmegaConf
        
        demo_dir = create_demo_configs()
        config_path = demo_dir / "config.yaml"
        cfg = OmegaConf.load(config_path)
        
        print("Original values:")
        print(f"  Model d_model: {cfg.model.d_model}")
        print(f"  Training batch_size: {cfg.training.batch_size}")
        print(f"  Training learning_rate: {cfg.training.learning_rate}")
        
        # Override values
        cfg.model.d_model = 128
        cfg.training.batch_size = 8
        cfg.training.learning_rate = 0.002
        
        print("\nAfter overriding:")
        print(f"  Model d_model: {cfg.model.d_model}")
        print(f"  Training batch_size: {cfg.training.batch_size}")
        print(f"  Training learning_rate: {cfg.training.learning_rate}")
        
        print("\n💡 In Hydra, you can override from command line:")
        print("   python script.py model.d_model=128 training.batch_size=8")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_3_hydra_main():
    """Demo 3: Use @hydra.main decorator"""
    print("\n" + "="*50)
    print("DEMO 3: Using @hydra.main Decorator")
    print("="*50)
    
    # Create a simple Hydra script
    hydra_script = '''
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="hydra_demo", config_name="config")
def main(cfg: DictConfig) -> None:
    print("🎉 Hydra loaded the config!")
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Model: {cfg.model.name} (d_model={cfg.model.d_model})")
    print(f"Training: batch_size={cfg.training.batch_size}, lr={cfg.training.learning_rate}")
    print(f"Data: {cfg.data.dataset_name} ({cfg.data.num_samples} samples)")

if __name__ == "__main__":
    main()
'''
    
    script_path = Path(__file__).parent / "hydra_demo_script.py"
    with open(script_path, 'w') as f:
        f.write(hydra_script)
    
    print(f"✅ Created Hydra script: {script_path}")
    print("\nThis script uses @hydra.main which:")
    print("  1. Automatically loads the config")
    print("  2. Handles command line overrides")
    print("  3. Sets up logging")
    print("  4. Manages output directories")
    
    print("\nTry running it:")
    print(f"  python {script_path}")
    print(f"  python {script_path} model.d_model=256")
    print(f"  python {script_path} training.batch_size=16")
    
    return True

def demo_4_baselines_example():
    """Demo 4: How it works in StackWise Baselines"""
    print("\n" + "="*50)
    print("DEMO 4: StackWise Baselines Example")
    print("="*50)
    
    print("In the StackWise Baselines framework, Hydra works like this:")
    print()
    
    print("1. 📁 CONFIG STRUCTURE:")
    print("   baselines/configs/")
    print("   ├── config.yaml              # Main config")
    print("   ├── model/")
    print("   │   └── encoder/")
    print("   │       └── bert_family/")
    print("   │           ├── tiny.yaml    # Small BERT")
    print("   │           ├── base.yaml    # Medium BERT")
    print("   │           └── large.yaml   # Large BERT")
    print("   ├── training/")
    print("   │   ├── classical.yaml       # Normal training")
    print("   │   └── depth_time.yaml      # Special training")
    print("   └── experiments/")
    print("       └── bert_reproduction/")
    print("           └── bert_base_glue.yaml  # Complete experiment")
    print()
    
    print("2. 🚀 USAGE EXAMPLES:")
    print("   # Use a complete experiment")
    print("   python scripts/train.py --config-name=experiments/bert_reproduction/bert_base_glue")
    print()
    print("   # Mix and match components")
    print("   python scripts/train.py model=encoder/bert_family/base training=classical")
    print()
    print("   # Override specific values")
    print("   python scripts/train.py model=encoder/bert_family/base model.d_model=512")
    print()
    print("   # Run multiple experiments")
    print("   python scripts/train.py --multirun model=encoder/bert_family/tiny,encoder/bert_family/base")
    print()
    
    print("3. 🎯 KEY CONCEPTS:")
    print("   • defaults: Tells Hydra which configs to load")
    print("   • @package: Tells Hydra where to put config sections")
    print("   • Overrides: Change values with key=value")
    print("   • Multi-run: Run multiple experiments with --multirun")
    print()
    
    return True

def demo_5_hands_on():
    """Demo 5: Hands-on practice"""
    print("\n" + "="*50)
    print("DEMO 5: Hands-on Practice")
    print("="*50)
    
    print("Now let's practice! Try these commands:")
    print()
    
    # Create demo configs if they don't exist
    demo_dir = create_demo_configs()
    
    print("1. 📖 Look at the config files:")
    print(f"   cat {demo_dir}/config.yaml")
    print(f"   cat {demo_dir}/model/tiny_bert.yaml")
    print(f"   cat {demo_dir}/training/fast.yaml")
    print()
    
    print("2. 🐍 Run the Python demos:")
    print("   python hydra_tutorial.py")
    print("   python hydra_demo_script.py")
    print()
    
    print("3. 🎮 Try the StackWise Baselines (if set up):")
    print("   cd baselines")
    print("   source .venv/bin/activate")
    print("   python scripts/train.py model=encoder/bert_family/tiny")
    print()
    
    print("4. 🔍 Explore the configs:")
    print("   ls baselines/configs/")
    print("   ls baselines/configs/model/")
    print("   ls baselines/configs/experiments/")
    print()
    
    print("💡 Remember: Hydra is just a way to organize and manage your settings!")
    print("   Think of it like a recipe book where you can mix and match ingredients.")
    
    return True

def main():
    """Run all demos"""
    print("🎓 HYDRA HANDS-ON DEMO")
    print("Learn Hydra by doing!")
    print()
    
    demos = [
        demo_1_load_config,
        demo_2_override_values,
        demo_3_hydra_main,
        demo_4_baselines_example,
        demo_5_hands_on,
    ]
    
    for i, demo in enumerate(demos, 1):
        try:
            demo()
            print(f"\n✅ Demo {i} complete!")
        except Exception as e:
            print(f"❌ Demo {i} failed: {e}")
        print()
    
    print("🎉 ALL DEMOS COMPLETE!")
    print()
    print("Next steps:")
    print("1. Run the demos above")
    print("2. Look at the generated config files")
    print("3. Try the StackWise Baselines examples")
    print("4. Create your own configs!")
    print()
    print("Remember: Hydra makes managing complex experiments easy! 🚀")

if __name__ == "__main__":
    main()
