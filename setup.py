#!/usr/bin/env python3
"""
Configuration d'installation pour EvaDentalAI
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lire le README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Lire les requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="evadental-ai",
    version="1.0.0",
    author="EvaDentalAI Team",
    author_email="contact@evadental.ai",
    description="Détection d'anomalies dentaires avec YOLO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/votre-username/EvaDentalAI_Yolo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "evadental-prepare=scripts.prepare_dataset:main",
            "evadental-train=scripts.train_model:main",
            "evadental-predict=scripts.predict:main",
            "evadental-export=scripts.export_model:main",
            "evadental-api=api.main:main",
            "evadental-demo=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "docker/*", "docs/*.md"],
    },
    keywords="dental, yolo, computer-vision, medical-imaging, ai, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/votre-username/EvaDentalAI_Yolo/issues",
        "Source": "https://github.com/votre-username/EvaDentalAI_Yolo",
        "Documentation": "https://github.com/votre-username/EvaDentalAI_Yolo/blob/main/README.md",
    },
)
