#!/usr/bin/env python3
"""
Setup script for Sanskrit Tutor RAG system.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)

setup(
    name="sanskrit-tutor",
    version="1.0.0",
    author="Sanskrit Tutor Team",
    author_email="contact@sanskrittutor.com",
    description="RAG-powered Sanskrit tutoring system with user-supplied data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sanskrit-tutor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.21.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "gradio>=4.0.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "local": ["llama-cpp-python>=0.2.20"],
        "gpu": ["faiss-gpu>=1.7.0", "torch>=1.13.0"],
        "audio": ["librosa>=0.9.0", "soundfile>=0.10.0", "torchaudio>=0.13.0"],
        "finetune": [
            "transformers>=4.30.0",
            "peft>=0.4.0",
            "bitsandbytes>=0.39.0",
            "accelerate>=0.20.0",
            "datasets>=2.12.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sanskrit-tutor=ui_gradio:main",
            "sanskrit-tutor-validate=utils.config_validator:main",
            "sanskrit-tutor-ingest=ingest:main",
            "sanskrit-tutor-index=embed_index:main",
            "sanskrit-tutor-rag=rag:main",
            "sanskrit-tutor-llm=llm_backends:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    zip_safe=False,
)
