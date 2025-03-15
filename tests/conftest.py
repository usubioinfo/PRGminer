"""
PRGminer Test Configuration
Author: Naveen Duhan (naveen.duhan@usu.edu)
"""

import os
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Fixture to provide test data directory path."""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_sequence():
    """Fixture to provide a sample protein sequence."""
    return {
        "seq1": "MAEGEQVQSGEDLGSPVAQVLQKAREQGAQAAVLVVPPGEEQVQSAEDLGSPVAQVLQKA",
        "seq2": "MTKFTILLFFLSVALASNAQPGCNQSQTLSPNWQNVFGASAASSCP"
    }

@pytest.fixture
def sample_fasta(test_data_dir):
    """Fixture to provide path to sample FASTA file."""
    fasta_path = test_data_dir / "sample.fasta"
    
    # Create test data directory if it doesn't exist
    test_data_dir.mkdir(exist_ok=True)
    
    # Create a sample FASTA file
    with open(fasta_path, "w") as f:
        f.write(">seq1\nMAEGEQVQSGEDLGSPVAQVLQKAREQGAQAAVLVVPPGEEQVQSAEDLGSPVAQVLQKA\n")
        f.write(">seq2\nMTKFTILLFFLSVALASNAQPGCNQSQTLSPNWQNVFGASAASSCP\n")
    
    return fasta_path

@pytest.fixture
def mock_model_output():
    """Fixture to provide mock model prediction output."""
    return {
        'seq1': {
            'class': 'Rgene',
            'probabilities': {
                'CNL': 0.8,
                'KIN': 0.05,
                'LYK': 0.03,
                'LECRK': 0.02,
                'RLK': 0.03,
                'RLP': 0.02,
                'TIR': 0.03,
                'TNL': 0.02
            }
        }
    }

@pytest.fixture
def cleanup_test_files(test_data_dir):
    """Fixture to clean up test files after tests."""
    yield
    # Clean up test files after tests
    if test_data_dir.exists():
        for file in test_data_dir.glob("*"):
            file.unlink()
        test_data_dir.rmdir() 