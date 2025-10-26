#!/usr/bin/env python3
"""
Setup script for StackWise Baselines module.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent.parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')

setup(
    name="stackwise-baselines",
    version="0.1.0",
    description="Comprehensive benchmarking framework for encoder-decoder model families",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="StackWise Team",
    author_email="team@stackwise.ai",
    url="https://github.com/stackwise/stackwise",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "advanced": [
            "accelerate>=0.20.0",
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stackwise-train=baselines.scripts.train:main",
            "stackwise-evaluate=baselines.scripts.evaluate:main",
            "stackwise-benchmark=baselines.scripts.benchmark:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="transformer language-model benchmarking evaluation hydra",
)
