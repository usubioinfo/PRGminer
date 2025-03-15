"""
Tests for PRGminer utility functions
"""

import pytest
import numpy as np
import os
import json
from pathlib import Path
from PRGminer.utils import (
    load_config,
    save_predictions,
    validate_input_file,
    setup_logging,
    get_version,
    argument_parser,
    AA_DICT,
    AMINO_ACIDS
)

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path

def test_config_loading():
    """Test configuration loading functionality."""
    # Test loading valid config
    config = {
        "model_params": {
            "max_length": 2000,
            "embedding_dim": 128
        }
    }
    
    config_path = Path("test_data") / "config.json"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config, f)
    
    loaded_config = load_config(config_path)
    assert loaded_config["model_params"]["max_length"] == 2000
    
    # Test loading non-existent config
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.json")
    
    # Test loading invalid JSON
    invalid_config = Path("test_data") / "invalid_config.json"
    with open(invalid_config, "w") as f:
        f.write("invalid json")
    
    with pytest.raises(json.JSONDecodeError):
        load_config(invalid_config)

def test_prediction_saving(mock_model_output, test_data_dir):
    """Test prediction saving functionality."""
    output_file = test_data_dir / "predictions.txt"
    
    # Test saving predictions
    save_predictions(mock_model_output, output_file)
    assert output_file.exists()
    
    # Verify content format
    with open(output_file) as f:
        content = f.read()
        assert "seq1" in content
        assert "Rgene" in content
    
    # Test saving to invalid path
    with pytest.raises(OSError):
        save_predictions(mock_model_output, "/invalid/path/predictions.txt")
    
    # Test saving invalid format
    with pytest.raises(ValueError):
        save_predictions("invalid format", output_file)

def test_input_validation(sample_fasta):
    """Test input file validation."""
    # Test valid FASTA file
    assert validate_input_file(sample_fasta) == True
    
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        validate_input_file("nonexistent.fasta")
    
    # Test invalid file format
    invalid_file = Path("test_data") / "invalid.txt"
    invalid_file.write_text("invalid content")
    
    with pytest.raises(ValueError):
        validate_input_file(invalid_file)
    
    # Test empty file
    empty_file = Path("test_data") / "empty.fasta"
    empty_file.touch()
    
    with pytest.raises(ValueError):
        validate_input_file(empty_file)

def test_logging_setup(test_data_dir):
    """Test logging setup functionality."""
    log_file = test_data_dir / "test.log"
    
    # Test basic logging setup
    logger = setup_logging(log_file)
    assert logger is not None
    assert log_file.exists()
    
    # Test logging levels
    logger = setup_logging(log_file, level="DEBUG")
    assert logger.level == 10  # DEBUG level
    
    logger = setup_logging(log_file, level="ERROR")
    assert logger.level == 40  # ERROR level
    
    # Test invalid logging level
    with pytest.raises(ValueError):
        setup_logging(log_file, level="INVALID")

def test_version_info():
    """Test version information retrieval."""
    version = get_version()
    
    # Test version format
    assert isinstance(version, str)
    assert len(version.split(".")) >= 2
    
    # Test version components
    major, minor = map(int, version.split(".")[:2])
    assert major >= 0
    assert minor >= 0

def test_file_handling(test_data_dir):
    """Test file handling utilities."""
    # Test directory creation
    new_dir = test_data_dir / "new_directory"
    assert not new_dir.exists()
    new_dir.mkdir()
    assert new_dir.exists()
    
    # Test file writing
    test_file = new_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
    assert test_file.read_text() == "test content"
    
    # Test file deletion
    test_file.unlink()
    assert not test_file.exists()
    
    # Clean up
    new_dir.rmdir()
    assert not new_dir.exists()

def test_argument_parser():
    """Test command line argument parser."""
    parser = argument_parser(version="0.1.0")
    
    # Test default values
    args = parser.parse_args(['-i', 'test.fasta'])
    assert args.output_dir == 'PRGminer_results'
    assert args.output_file == 'PRGminer_results.txt'
    assert args.level == 'Phase2'
    assert args.fasta_file == 'test.fasta'
    
    # Test custom values
    args = parser.parse_args([
        '-i', 'input.fasta',
        '-od', 'custom_output',
        '-o', 'results.txt',
        '-l', 'Phase1'
    ])
    assert args.output_dir == 'custom_output'
    assert args.output_file == 'results.txt'
    assert args.level == 'Phase1'
    assert args.fasta_file == 'input.fasta'
    
    # Test invalid prediction level
    with pytest.raises(SystemExit):
        parser.parse_args(['-i', 'test.fasta', '-l', 'invalid'])

def test_setup_logging(temp_dir):
    """Test logging configuration."""
    log_file = temp_dir / "test.log"
    
    # Test with default level
    logger = setup_logging(log_file)
    assert logger.level == logger.getEffectiveLevel()
    assert len(logger.handlers) == 2  # File and console handlers
    
    # Test with custom level
    logger = setup_logging(log_file, level="DEBUG")
    assert logger.level == logger.getEffectiveLevel()
    
    # Test with invalid level
    with pytest.raises(ValueError):
        setup_logging(log_file, level="INVALID")
    
    # Test log file creation
    assert log_file.exists()
    
    # Test parent directory creation
    deep_log = temp_dir / "logs" / "deep" / "test.log"
    logger = setup_logging(deep_log)
    assert deep_log.parent.exists()

def test_get_version():
    """Test version retrieval."""
    version = get_version()
    assert isinstance(version, str)
    assert version == "0.1.0"  # Update this when version changes

def test_amino_acid_constants():
    """Test amino acid constants."""
    # Test AMINO_ACIDS list
    assert len(AMINO_ACIDS) == 20
    assert all(isinstance(aa, str) for aa in AMINO_ACIDS)
    assert all(len(aa) == 1 for aa in AMINO_ACIDS)
    
    # Test AA_DICT mapping
    assert len(AA_DICT) == 20
    assert all(isinstance(aa, str) for aa in AA_DICT.keys())
    assert all(isinstance(idx, int) for idx in AA_DICT.values())
    assert all(len(aa) == 1 for aa in AA_DICT.keys())
    assert set(AA_DICT.keys()) == set(AMINO_ACIDS)
    assert set(AA_DICT.values()) == set(range(20)) 