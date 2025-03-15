# Getting Started with PRGminer

## Introduction
PRGminer is a state-of-the-art bioinformatics tool that uses deep learning to predict and classify plant resistance genes (R-genes). This guide will help you get started with PRGminer quickly.

## Prerequisites
Before using PRGminer, ensure you have:
- Python 3.7 or higher installed
- At least 4GB RAM
- 2GB free disk space
- Linux or macOS operating system

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/navduhan/PRGminer.git
cd PRGminer

# Create and activate conda environment
conda env create -f environment.yml
conda activate PRGminer

# Install the package
pip install .
```

### 2. Basic Usage
```bash
# Run Phase 1 prediction (R-gene vs non-R-gene)
PRGminer -i your_proteins.fasta -od results_phase1 -l Phase1

# Run Phase 2 prediction (detailed classification)
PRGminer -i your_proteins.fasta -od results_phase2 -l Phase2
```

### 3. Understanding Results
- Phase 1 output provides binary classification (R-gene vs non-R-gene)
- Phase 2 output classifies R-genes into eight categories (CNL, KIN, LYK, etc.)
- Results are saved in tab-separated text files

## Next Steps
- Read the [Installation Guide](installation.md) for detailed setup instructions
- Check the [Usage Tutorial](usage.md) for advanced features
- See [Examples](../examples/basic_usage.md) for practical use cases
- Review [Input/Output Formats](io_formats.md) for file specifications

## Getting Help
- Check the [Troubleshooting Guide](troubleshooting.md)
- Submit issues on [GitHub](https://github.com/navduhan/PRGminer/issues)
- Contact support at naveen.duhan@usu.edu 