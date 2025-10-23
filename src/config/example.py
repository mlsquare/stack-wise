"""
Example usage of the StackWise configuration system.
"""

from src.config import StackWiseConfig, ModelConfig, TrainingConfig, DataConfig


def example_basic_usage():
    """Example of basic configuration usage."""
    
    # Create configuration from YAML file
    config = StackWiseConfig.from_yaml("config.yaml")
    
    # Set vocabulary size from tokenizer (example)
    config.model.set_vocab_size(50000)  # This would come from actual tokenizer
    
    # Validate configuration
    config.validate()
    
    # Access sub-configurations
    print(f"Model: {config.model.d_model}D, {config.model.n_layers} layers")
    print(f"Vocabulary: {config.model.vocab_size} tokens")
    print(f"Attention: {config.model.attention_type} ({config.model.attention_mode})")
    print(f"Training: {config.training.lr} lr, {config.training.batch_size} batch size")
    
    # Save configuration
    config.to_yaml("config_backup.yaml")


def example_custom_config():
    """Example of creating custom configuration."""
    
    # Create custom model configuration
    model_config = ModelConfig(
        d_model=2048,
        n_layers=12,
        attention_type="gqa",
        attention_mode="bidirectional",
        use_rope=True
    )
    
    # Create custom training configuration
    training_config = TrainingConfig(
        lr=2e-4,
        batch_size=8,
        max_steps=500,
        fine_tune_mode="mlm"
    )
    
    # Create custom data configuration
    data_config = DataConfig(
        use_dummy_data=False,
        dataset_path="/path/to/dataset",
        num_samples=1000
    )
    
    # Combine into main configuration
    config = StackWiseConfig(
        model=model_config,
        training=training_config,
        data=data_config
    )
    
    # Validate and save
    config.validate()
    config.to_yaml("custom_config.yaml")


def example_configuration_validation():
    """Example of configuration validation."""
    
    try:
        # This will raise an error due to invalid parameters
        invalid_config = ModelConfig(
            d_model=-1,  # Invalid: negative dimension
            n_heads=5,  # Invalid: not divisible by n_kv_heads for GQA
            attention_type="gqa",
            n_kv_heads=3
        )
        invalid_config.validate()
    except ValueError as e:
        print(f"Validation error: {e}")


if __name__ == "__main__":
    print("StackWise Configuration System Examples")
    print("=" * 50)
    
    print("\n1. Basic Usage:")
    example_basic_usage()
    
    print("\n2. Custom Configuration:")
    example_custom_config()
    
    print("\n3. Configuration Validation:")
    example_configuration_validation()
    
    print("\nConfiguration system ready!")
