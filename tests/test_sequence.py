"""
Tests for sequence processing functionality
Author: Naveen Duhan (naveen.duhan@usu.edu)
"""

import pytest
import numpy as np
from PRGminer.sequence import SequenceProcessor
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def test_sequence_validation(sample_sequence):
    """Test sequence validation functionality."""
    processor = SequenceProcessor()
    
    # Test valid sequence
    assert processor.validate_sequence(sample_sequence["seq1"]) == True
    
    # Test invalid sequence with numbers
    with pytest.raises(ValueError):
        processor.validate_sequence("MAE123GEQ")
    
    # Test invalid sequence with special characters
    with pytest.raises(ValueError):
        processor.validate_sequence("MAE@GEQ")
    
    # Test empty sequence
    with pytest.raises(ValueError):
        processor.validate_sequence("")

def test_sequence_encoding(sample_sequence):
    """Test sequence encoding functionality."""
    processor = SequenceProcessor()
    
    # Test encoding of valid sequence
    encoded = processor.encode_sequence(sample_sequence["seq1"])
    assert encoded.shape == (1, processor.max_length, len(processor.amino_acids))
    assert encoded.dtype == "float32"
    
    # Test encoding of sequence shorter than max_length
    encoded_short = processor.encode_sequence(sample_sequence["seq2"])
    assert encoded_short.shape == (1, processor.max_length, len(processor.amino_acids))
    
    # Test padding
    assert encoded_short[0][-1].sum() == 0  # Check if padding is zeros

def test_fasta_loading(sample_fasta):
    """Test FASTA file loading functionality."""
    processor = SequenceProcessor()
    
    # Test loading valid FASTA file
    sequences = processor.load_fasta(sample_fasta)
    assert len(sequences) == 2
    assert "seq1" in sequences
    assert "seq2" in sequences
    
    # Test loading non-existent file
    with pytest.raises(FileNotFoundError):
        processor.load_fasta("nonexistent.fasta")
    
    # Test loading invalid FASTA file
    with open(sample_fasta.parent / "invalid.fasta", "w") as f:
        f.write("Invalid FASTA format")
    
    with pytest.raises(ValueError):
        processor.load_fasta(sample_fasta.parent / "invalid.fasta")

def test_sequence_preprocessing(sample_sequence):
    """Test sequence preprocessing pipeline."""
    processor = SequenceProcessor()
    
    # Test complete preprocessing pipeline
    processed = processor.preprocess(sample_sequence["seq1"])
    assert processed is not None
    assert len(processed.shape) == 3  # (batch_size, sequence_length, num_features)
    assert processed.shape == (1, processor.max_length, len(processor.amino_acids))
    
    # Test preprocessing with invalid sequence
    with pytest.raises(ValueError):
        processor.preprocess("INVALID@123")

def test_batch_processing(sample_sequence):
    """Test batch processing of sequences."""
    processor = SequenceProcessor()
    
    # Test processing multiple sequences
    batch_results = processor.process_batch(sample_sequence)
    assert len(batch_results) == 2
    assert all(result.shape == (1, processor.max_length, len(processor.amino_acids)) 
              for result in batch_results)
    
    # Test empty batch
    with pytest.raises(ValueError):
        processor.process_batch({})
    
    # Test batch with invalid sequence
    invalid_batch = {"seq1": "INVALID@123"}
    with pytest.raises(ValueError):
        processor.process_batch(invalid_batch)

def test_sequence_output_formatting(sample_sequence, mock_model_output):
    """Test sequence output formatting."""
    processor = SequenceProcessor()
    
    # Test formatting prediction output
    formatted = processor.format_output(mock_model_output)
    assert isinstance(formatted, dict)
    assert "seq1" in formatted
    assert "class" in formatted["seq1"]
    assert "probabilities" in formatted["seq1"]
    
    # Test formatting empty output
    with pytest.raises(ValueError):
        processor.format_output({})
    
    # Test formatting invalid output structure
    with pytest.raises(ValueError):
        processor.format_output({"seq1": "invalid"}) 