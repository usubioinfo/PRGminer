# PRGminer: Deep Neural Network-Based Plant Resistance Gene Prediction

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Bioinformatics](https://img.shields.io/badge/Bioinformatics-Tool-brightgreen.svg)](https://kaabil.net/PRGminer/)

## Overview

PRGminer is a state-of-the-art bioinformatics tool that employs deep learning to predict and classify plant resistance genes (R-genes). The tool implements a two-phase prediction approach:

1. **Phase 1**: Binary classification of sequences as R-genes or non-R-genes
2. **Phase 2**: Detailed classification of R-genes into eight distinct categories:
   - CNL (Coiled-coil NBS-LRR)
   - KIN (Kinase)
   - LYK (Lysin Motif Kinase)
   - LECRK (Lectin Receptor Kinase)
   - RLK (Receptor-like Kinase)
   - RLP (Receptor-like Protein)
   - TIR (Toll/Interleukin-1 Receptor)
   - TNL (TIR-NBS-LRR)

## Features

- 🧬 Advanced deep learning models for accurate R-gene prediction
- 🔄 Two-phase prediction pipeline
- 📊 Detailed probability scores for each prediction
- 🚀 Fast and efficient processing
- 💻 User-friendly command-line interface
- 📝 Comprehensive output reports

## Requirements

### System Requirements
- Linux or macOS operating system
- Python 3.7 or higher
- 4GB RAM (minimum)
- 2GB free disk space

### Dependencies
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Biopython
- Scikit-learn

## Installation

Choose one of the following installation methods:

### 1. Using Git and Conda (Recommended)

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

### 2. Using Miniconda

```bash
# Download PRGminer
wget https://kaabil.net/PRGminer/download/PRGminer.tar.gz
tar -xvzf PRGminer.tar.gz
cd PRGminer

# Create and activate environment
conda env create -f environment.yml
conda activate PRGminer
pip install .
```

### 3. Using System Python

```bash
# Download and extract
wget https://kaabil.net/PRGminer/download/PRGminer.tar.gz
tar -xvzf PRGminer.tar.gz
cd PRGminer

# Install
pip install .
```

## Usage

### Basic Usage

```bash
PRGminer -i <input.fasta> -od <output_directory> -l <prediction_level>
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-i, --fasta_file` | Input protein sequences in FASTA format | Required |
| `-od, --output_dir` | Output directory for results | PRGminer_results |
| `-l, --level` | Prediction level (Phase1 or Phase2) | Phase2 |
| `-o, --output_file` | Output file name | PRGminer_results.txt |

### Example

```bash
# Phase 1 prediction (R-gene vs non-R-gene)
PRGminer -i proteins.fasta -od results_phase1 -l Phase1

# Phase 2 prediction (detailed classification)
PRGminer -i proteins.fasta -od results_phase2 -l Phase2
```

## Input Format

PRGminer accepts protein sequences in FASTA format only. Example:

```fasta
>protein1
MAEGEQVQSGEDLGSPVAQVLQKAREQGAQAAVLVVPPGEEQVQSAEDLGSPVAQVLQKA
>protein2
MTKFTILLFFLSVALASNAQPGCNQSQTLSPNWQNVFGASAASSCP
```

⚠️ Note: Nucleotide sequences are not supported.

## Output Structure

PRGminer generates a comprehensive output directory with the following structure:

```
output_directory/
├── PRGminer.log                 # Detailed execution log
├── prediction_summary.txt       # Summary of prediction results
├── PRGminer_results.txt        # Final consolidated results
└── intermediate_files/
    ├── Phase1_predictions.tsv  # Phase 1 detailed predictions
    ├── phase2_input.fasta     # R-genes identified for Phase 2
    └── Phase2_predictions.tsv  # Phase 2 detailed predictions
```

### Output File Formats

#### 1. PRGminer_results.txt (Final Results)
```
SampleID    Prediction    Probability    Additional_Info
seq1        Rgene        0.9234         CNL
seq2        Non-Rgene    0.8567         -
seq3        Rgene        0.9876         RLK
```

#### 2. prediction_summary.txt
```
PRGminer Prediction Summary
========================

Prediction Level: Phase1
Total sequences analyzed: 100
Class Distribution:
-------------------
Rgene: 35 sequences (35.0000%)
Non-Rgene: 65 sequences (65.0000%)

=================================================

Prediction Level: Phase2
Total sequences analyzed: 35
Class Distribution:
-------------------
CNL: 10 sequences (28.5714%)
RLK: 8 sequences (22.8571%)
TNL: 6 sequences (17.1429%)
...
```

#### 3. Phase1_predictions.tsv
```
SampleID    Prediction    Rgene        Non-Rgene
seq1        Rgene        0.9234       0.0766
seq2        Non-Rgene    0.1433       0.8567
seq3        Rgene        0.9876       0.0124
```

#### 4. Phase2_predictions.tsv
```
SampleID    Prediction    CNL         KIN         LYK         LECRK       RLK         RLP         TIR         TNL
seq1        CNL          0.8234      0.0234      0.0156      0.0145      0.0567      0.0234      0.0230      0.0200
seq3        RLK          0.0234      0.0567      0.0145      0.0234      0.7234      0.0890      0.0456      0.0240
```

### Understanding the Output

1. **Probability Scores**
   - All probabilities are reported with 4 decimal places
   - Values range from 0.0000 to 1.0000 (or 0% to 100%)
   - Higher values indicate stronger predictions

2. **Prediction Confidence**
   - High confidence: > 0.8000 (80%)
   - Medium confidence: 0.6000-0.8000 (60-80%)
   - Low confidence: < 0.6000 (60%)

3. **Phase-specific Information**
   - Phase1: Binary classification (Rgene vs Non-Rgene)
   - Phase2: Multi-class classification into 8 R-gene categories
   - Each phase includes detailed probability distributions

4. **Log File Details**
   - Timestamp for each prediction
   - Processing parameters used
   - Any warnings or errors encountered
   - Performance metrics

### Interpreting Results

1. **For Phase1:**
   - Sequences with Rgene probability > 0.5000 are classified as R-genes
   - Higher probabilities indicate stronger R-gene characteristics

2. **For Phase2:**
   - The highest probability among the 8 classes determines the final prediction
   - Probability distribution shows relative confidence for each class
   - Close probabilities may indicate hybrid or novel R-gene types

## Performance Considerations

- Processing time depends on:
  - Number of input sequences
  - Sequence lengths
  - Available computational resources
- Recommended batch size: < 1000 sequences
- For large datasets, consider splitting into smaller batches

## Troubleshooting

Common issues and solutions:

1. **Invalid sequence format**
   - Ensure sequences are in proper FASTA format
   - Verify sequences contain valid amino acids only

2. **Memory errors**
   - Reduce batch size
   - Close unnecessary applications
   - Increase system swap space

3. **Installation issues**
   - Verify Python version compatibility
   - Check for conflicting dependencies
   - Ensure proper environment activation

## Citation

If you use PRGminer in your research, please cite:

<!-- ```bibtex
@article{PRGminer2023,
    author = {Duhan, Naveen and Kaundal, Rakesh},
    title = {PRGminer: Deep learning-based prediction and classification of plant resistance genes},
   
}
``` -->

## Support

### Technical Support
For bugs and technical issues:
- Create an issue on GitHub
- Email: naveen.duhan@usu.edu

### Scientific Inquiries
For questions about the methodology:
- Dr. Rakesh Kaundal: rkaundal@usu.edu
- Naveen Duhan: naveen.duhan@usu.edu

## License

PRGminer is released under the [GNU General Public License v3](LICENSE).

## Acknowledgments

This work was supported by the Kaundal Bioinformatics Lab at Utah State University.

---
© 2023 Kaundal Bioinformatics Lab, Utah State University



