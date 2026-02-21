"""Setup configuration for AutoPerfPy package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

setup(
    name="autoperfpy",
    version="1.0",
    author="Hamna Nimra",
    description="Performance analysis and benchmarking toolkit for NVIDIA GPU platforms and LLM workloads",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HamnaNimra/AutoPerfPy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: System :: Performance/Monitoring",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "pyyaml>=5.3.0",
        "requests>=2.25.0",
        "matplotlib",
        "plotly",
        "streamlit>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12.0",
            "pytest-mock>=3.6.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.0",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.12.0",
            "pytest-mock>=3.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "autoperfpy=autoperfpy.cli:main",
            "trackiq-compare=trackiq_compare.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
