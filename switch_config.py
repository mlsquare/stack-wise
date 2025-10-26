#!/usr/bin/env python3
"""
Config Switcher for StackWise

This script helps switch between different configuration files:
- config.yaml: Small model for fast testing
- config_standard.yaml: Standard model compatible with common tokenizers

Usage:
    python switch_config.py small    # Use small config
    python switch_config.py standard # Use standard config
"""

import sys
import shutil
from pathlib import Path

def switch_config(config_type):
    """Switch to the specified configuration"""
    config_dir = Path(".")
    
    if config_type == "small":
        source = config_dir / "config_small.yaml"
        target = config_dir / "config.yaml"
        print("🔄 Switching to SMALL model configuration...")
    elif config_type == "standard":
        source = config_dir / "config_standard.yaml"
        target = config_dir / "config.yaml"
        print("🔄 Switching to STANDARD model configuration...")
    else:
        print(f"❌ Unknown config type: {config_type}")
        print("Available types: small, standard")
        return False
    
    if not source.exists():
        print(f"❌ Source config file not found: {source}")
        return False
    
    # Backup current config
    if target.exists():
        backup = target.with_suffix(".yaml.backup")
        shutil.copy2(target, backup)
        print(f"📦 Backed up current config to: {backup}")
    
    # Copy new config
    shutil.copy2(source, target)
    print(f"✅ Switched to {config_type} configuration")
    print(f"   Source: {source}")
    print(f"   Target: {target}")
    
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python switch_config.py <small|standard>")
        print("\nConfigurations:")
        print("  small     - Very small model (d_model=128, vocab=1000) for fast testing")
        print("  standard  - Standard model (d_model=768, vocab=50257) for real tokenizers")
        sys.exit(1)
    
    config_type = sys.argv[1].lower()
    success = switch_config(config_type)
    
    if success:
        print(f"\n🎯 Now using {config_type} configuration!")
        if config_type == "small":
            print("   - Fast testing and development")
            print("   - Small memory footprint")
            print("   - Limited vocabulary")
        else:
            print("   - Compatible with standard tokenizers")
            print("   - Realistic model sizes")
            print("   - Production-ready parameters")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
