#!/usr/bin/env python3
"""
LLM Council - Setup Script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-council",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Open-source multi-model LLM consensus system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-council",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": ["black", "isort", "mypy", "ruff", "pre-commit"],
        "test": ["pytest", "pytest-asyncio", "pytest-cov", "hypothesis"],
        "vllm": ["vllm>=0.2.0"],
        "llamacpp": ["llama-cpp-python>=0.2.0"],
    },
    entry_points={
        "console_scripts": [
            "llm-council=main:app",
        ],
    },
)
