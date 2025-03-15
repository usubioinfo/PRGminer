#!/usr/bin/python

"""
PRGminer Utilities Module

This module provides utility functions for the PRGminer tool, including:
- Sequence feature extraction (DPC, NMBroto)
- FASTA file processing
- Command line argument parsing

The module implements various protein sequence analysis methods:
- Dipeptide Composition (DPC)
- Normalized Moreau-Broto autocorrelation (NMBroto)
- Hybrid feature extraction

Author: Naveen Duhan
Lab: KAABiL (Kaundal Artificial Intelligence & Advanced Bioinformatics Lab)
Version: 0.1

Dependencies:
    - numpy
    - biopython
    - argparse
"""

import numpy as np
import re
import argparse
import os
import logging
import json
from typing import List, Dict, Union, Tuple, Any
from Bio import SeqIO
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

AA_DICT = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}

# Amino acid index data for NMBroto calculation
AA_INDEX = np.array([
    [ 2.000e-02, -4.200e-01, -7.700e-01, -1.040e+00,  7.700e-01, -1.100e+00, -1.140e+00, -8.000e-01,  2.600e-01,  1.810e+00, 1.140e+00, -4.100e-01,  1.000e+00,  1.350e+00, -9.000e-02, -9.700e-01, -7.700e-01,  1.710e+00,  1.110e+00,  1.130e+00],
    [ 3.570e-01,  5.290e-01,  4.630e-01,  5.110e-01,  3.460e-01,  4.930e-01,  4.970e-01,  5.440e-01,  3.230e-01,  4.620e-01, 3.650e-01,  4.660e-01,  2.950e-01,  3.140e-01,  5.090e-01,  5.070e-01,  4.440e-01,  3.050e-01,  4.200e-01,  3.860e-01],
    [ 4.600e-02,  2.910e-01,  1.340e-01,  1.050e-01,  1.280e-01,  1.800e-01,  1.510e-01,  0.000e+00,  2.300e-01,  1.860e-01, 1.860e-01,  2.190e-01,  2.210e-01,  2.900e-01,  1.310e-01,  6.200e-02,  1.080e-01,  4.090e-01,  2.980e-01,  1.400e-01],
    [-3.680e-01, -1.030e+00,  0.000e+00,  2.060e+00,  4.530e+00,  7.310e-01,  1.770e+00, -5.250e-01,  0.000e+00,  7.910e-01, 1.070e+00,  0.000e+00,  6.560e-01,  1.060e+00, -2.240e+00, -5.240e-01,  0.000e+00,  1.600e+00,  4.910e+00,  4.010e-01],
    [ 1.150e+02,  2.250e+02,  1.600e+02,  1.500e+02,  1.350e+02,  1.800e+02,  1.900e+02,  7.500e+01,  1.950e+02,  1.750e+02, 1.700e+02,  2.000e+02,  1.850e+02,  2.100e+02,  1.450e+02,  1.150e+02,  1.400e+02,  2.550e+02,  2.300e+02,  1.550e+02],
    [ 5.260e+01,  1.091e+02,  7.570e+01,  6.840e+01,  6.830e+01,  8.970e+01,  8.470e+01,  3.630e+01,  9.190e+01,  1.020e+02, 1.020e+02,  1.051e+02,  9.770e+01,  1.139e+02,  7.360e+01,  5.490e+01,  7.120e+01,  1.354e+02,  1.162e+02,  8.510e+01],
    [ 5.200e-01,  6.800e-01,  7.600e-01,  7.600e-01,  6.200e-01,  6.800e-01,  6.800e-01,  0.000e+00,  7.000e-01,  1.020e+00, 9.800e-01,  6.800e-01,  7.800e-01,  7.000e-01,  3.600e-01,  5.300e-01,  5.000e-01,  7.000e-01,  7.000e-01,  7.600e-01],
    [ 1.000e+02,  6.500e+01,  1.340e+02,  1.060e+02,  2.000e+01,  9.300e+01,  1.020e+02,  4.900e+01,  6.600e+01,  9.600e+01, 4.000e+01,  5.600e+01,  9.400e+01,  4.100e+01,  5.600e+01,  1.200e+02,  9.700e+01,  1.800e+01,  4.100e+01,  7.400e+01]
])

class SequenceError(Exception):
    """Custom exception for sequence-related errors."""
    pass

def validate_sequence(seq: str) -> str:
    """
    Validates and cleans protein sequence.

    Args:
        seq (str): Input protein sequence

    Returns:
        str: Cleaned sequence containing only valid amino acids

    Raises:
        SequenceError: If sequence is invalid or empty
    """
    if not seq:
        raise SequenceError("Empty sequence provided")
    
    clean_seq = re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(seq).upper())
    if not clean_seq:
        raise SequenceError("No valid amino acids found in sequence")
    
    return clean_seq

def DPC(seq: str) -> List[float]:
    """
    Calculates Dipeptide Composition (DPC) for a protein sequence.

    Args:
        seq (str): Input protein sequence

    Returns:
        List[float]: List of DPC values (400 features)

    Raises:
        SequenceError: If sequence is invalid
    """
    try:
        seq = validate_sequence(seq)
        N = len(seq)
        if N < 2:
            raise SequenceError("Sequence too short for DPC calculation (minimum 2 residues required)")

        dpc = []
        for i in AMINO_ACIDS:
            for j in AMINO_ACIDS:
                dp = i + j
                dp_count = seq.count(dp)
                dp_freq = round(float(dp_count) / (N - 1) * 100, 2)
                dpc.append(dp_freq)
        
        return dpc
    except Exception as e:
        raise SequenceError(f"DPC calculation failed: {str(e)}")

def NMBroto(seq: str, nlag: int = 30) -> List[float]:
    """
    Calculates Normalized Moreau-Broto autocorrelation for a protein sequence.

    Args:
        seq (str): Input protein sequence
        nlag (int): Maximum lag (default: 30)

    Returns:
        List[float]: List of NMBroto values

    Raises:
        SequenceError: If sequence is invalid
    """
    try:
        seq = validate_sequence(seq)
        if len(seq) <= nlag:
            raise SequenceError(f"Sequence length ({len(seq)}) must be greater than nlag ({nlag})")

        # Normalize AA index
        aa_idx = AA_INDEX.copy()
        pstd = np.std(aa_idx, axis=1)
        pmean = np.mean(aa_idx, axis=1)
        aa_idx = (aa_idx - pmean[:, np.newaxis]) / pstd[:, np.newaxis]

        code = []
        N = len(seq)
        
        for prop in range(8):
            for n in range(1, nlag + 1):
                rn = sum(aa_idx[prop][AA_DICT.get(seq[j], 0)] * 
                        aa_idx[prop][AA_DICT.get(seq[j + n], 0)]
                        for j in range(N - n)) / (N - n)
                code.append(float(rn))
        
        return code
    except Exception as e:
        raise SequenceError(f"NMBroto calculation failed: {str(e)}")

def hybrid(seq: str, f1: callable, f2: callable) -> List[float]:
    """
    Combines features from two different feature extraction methods.

    Args:
        seq (str): Input protein sequence
        f1 (callable): First feature extraction function
        f2 (callable): Second feature extraction function

    Returns:
        List[float]: Combined feature list

    Raises:
        SequenceError: If feature extraction fails
    """
    try:
        features = []
        features.extend(f1(seq))
        features.extend(f2(seq))
        return features
    except Exception as e:
        raise SequenceError(f"Hybrid feature extraction failed: {str(e)}")

def preprocess(file: str, feat: str, class_label: List[int], size: int) -> Dict[str, Union[np.ndarray, List]]:
    """
    Preprocesses FASTA file and extracts features.

    Args:
        file (str): Input FASTA file path
        feat (str): Feature type ('DPC' or 'hybrid')
        class_label (List[int]): Class labels
        size (int): Feature size

    Returns:
        Dict: Dictionary containing:
            - Samples: numpy array of features
            - Labels: list of class labels
            - SeqID: list of sequence IDs

    Raises:
        FileNotFoundError: If FASTA file not found
        ValueError: If invalid feature type specified
    """
    try:
        if not os.path.exists(file):
            raise FileNotFoundError(f"FASTA file not found: {file}")

        if feat not in ["DPC", "hybrid"]:
            raise ValueError(f"Invalid feature type: {feat}. Must be 'DPC' or 'hybrid'")

        sequences = list(SeqIO.parse(file, 'fasta'))
        if not sequences:
            raise ValueError("No valid sequences found in FASTA file")

        features = []
        class_labels = []
        seq_ids = []

        for seq_record in sequences:
            try:
                seq_str = validate_sequence(str(seq_record.seq))
                
                if feat == "DPC":
                    feat_vector = DPC(seq_str)
                else:  # hybrid
                    feat_vector = hybrid(seq_str, DPC, NMBroto)
                
                features.append(feat_vector)
                class_labels.append(class_label)
                seq_ids.append(seq_record.id)
                
            except Exception as e:
                logger.warning(f"Skipping sequence {seq_record.id}: {str(e)}")
                continue

        if not features:
            raise ValueError("No valid features extracted from sequences")

        # Convert to numpy array and resize
        feature_array = np.zeros((len(features), size))
        for i, feat in enumerate(features):
            feature_array[i] = feat[:size]  # Truncate or pad to size

        return {
            'Samples': feature_array,
            'Labels': class_labels,
            'SeqID': seq_ids
        }

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

def fasta_process(fasta_file: str, result_file: str, wanted: List[str]) -> None:
    """
    Extracts specific sequences from a FASTA file.

    Args:
        fasta_file (str): Input FASTA file path
        result_file (str): Output FASTA file path
        wanted (List[str]): List of sequence IDs to extract

    Raises:
        FileNotFoundError: If input file not found
        IOError: If writing output fails
    """
    try:
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"Input FASTA file not found: {fasta_file}")

        wanted_set = set(wanted)  # Convert to set for O(1) lookup
        output_dir = os.path.dirname(result_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(result_file, "w") as out_handle:
            for seq_record in SeqIO.parse(fasta_file, 'fasta'):
                if seq_record.id in wanted_set:
                    SeqIO.write([seq_record], out_handle, "fasta")
                    wanted_set.remove(seq_record.id)

        if wanted_set:
            logger.warning(f"Some sequences were not found: {', '.join(wanted_set)}")

    except Exception as e:
        logger.error(f"FASTA processing failed: {str(e)}")
        raise

def argument_parser(version: str = None) -> argparse.ArgumentParser:
    """
    Creates argument parser for PRGminer command line interface.

    Args:
        version (str, optional): Version string to display

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="PRGminer: Plant resistance gene prediction tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-od', '--output_dir',
                       default='PRGminer_results',
                       help="Output directory")
    
    parser.add_argument('-o', '--output_file',
                       default="PRGminer_results.txt",
                       help="Output file")
    
    parser.add_argument('-i', '--fasta_file',
                       required=True,
                       help="Input FASTA file")
    
    parser.add_argument('-l', '--level',
                       default='Phase2',
                       choices=['Phase1', 'Phase2'],
                       help="Choose level for prediction")

    if version:
        parser.add_argument('--version',
                          action='version',
                          version=f'%(prog)s {version}')
    
    return parser

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dict containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    try:
        with open(config_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file: {e.msg}", e.doc, e.pos)

def save_predictions(predictions: Dict, output_file: Union[str, Path]) -> None:
    """Save predictions to file.
    
    Args:
        predictions: Dictionary of predictions
        output_file: Path to output file
        
    Raises:
        ValueError: If predictions format is invalid
        OSError: If file cannot be written
    """
    if not isinstance(predictions, dict):
        raise ValueError("Predictions must be a dictionary")
        
    output_file = Path(output_file)
    try:
        with open(output_file, 'w') as f:
            for seq_id, pred in predictions.items():
                f.write(f"Sequence: {seq_id}\n")
                f.write(f"Class: {pred['class']}\n")
                f.write("Probabilities:\n")
                for cls, prob in pred['probabilities'].items():
                    f.write(f"  {cls}: {prob:.4f}\n")
                f.write("\n")
    except OSError as e:
        raise OSError(f"Error writing predictions to file: {e}")

def validate_input_file(file_path: Union[str, Path]) -> bool:
    """Validate input FASTA file.
    
    Args:
        file_path: Path to input file
        
    Returns:
        True if file is valid
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is invalid
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if file_path.stat().st_size == 0:
        raise ValueError("Empty file")
        
    try:
        with open(file_path) as f:
            records = list(SeqIO.parse(f, "fasta"))
            if not records:
                raise ValueError("No valid FASTA sequences found")
        return True
    except Exception as e:
        raise ValueError(f"Invalid FASTA file: {e}")

def setup_logging(log_file: Union[str, Path], level: str = "INFO") -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
        
    Raises:
        ValueError: If invalid logging level
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate logging level
    level = level.upper()
    if level not in logging._nameToLevel:
        raise ValueError(f"Invalid logging level: {level}")
    
    # Configure logger
    logger = logging.getLogger("PRGminer")
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger

def get_version() -> str:
    """Get PRGminer version.
    
    Returns:
        Version string
    """
    return "0.1.0"  # This should match setup.py version
