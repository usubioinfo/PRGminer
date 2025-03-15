"""
Tests for PRGminer main module functionality
Author: Naveen Duhan (naveen.duhan@usu.edu)
"""

import pytest
import os
import shutil
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from PRGminer.__main__ import (
    setup_logging,
    cleanup_output_directory,
    extract_sequences,
    create_summary_report,
    DNN,
    main,
    PRGminerError,
    validate_inputs
)

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path

@pytest.fixture
def sample_fasta(temp_dir):
    """Create a sample FASTA file for testing."""
    fasta_content = """>seq1
MAEGEQVQSGEDLGSPVAQVLQKAREQGAQAAVLVVPPGEEQVQSAEDLGSPVAQVLQKA
>seq2
MTKFTILLFFLSVALASNAQPGCNQSQTLSPNWQNVFGASAASSCP"""
    fasta_file = temp_dir / "test.fasta"
    fasta_file.write_text(fasta_content)
    return fasta_file

@pytest.fixture
def sample_results():
    """Create sample prediction results."""
    return pd.DataFrame({
        'SampleID': ['seq1', 'seq2'],
        'Prediction': ['Rgene', 'Non-Rgene'],
        'Rgene': [0.8, 0.2],
        'Non-Rgene': [0.2, 0.8]
    })

def test_validate_inputs(temp_dir, sample_fasta):
    """Test input validation."""
    # Test valid inputs
    output_dir = temp_dir / "output"
    validate_inputs(sample_fasta, output_dir, "Phase1")
    
    # Test non-existent input file
    with pytest.raises(PRGminerError):
        validate_inputs(temp_dir / "nonexistent.fasta", output_dir, "Phase1")
    
    # Test invalid prediction level
    with pytest.raises(PRGminerError):
        validate_inputs(sample_fasta, output_dir, "invalid")
    
    # Test output path exists as file
    output_file = temp_dir / "output"
    output_file.touch()
    with pytest.raises(PRGminerError):
        validate_inputs(sample_fasta, output_file, "Phase1")

def test_setup_logging(temp_dir):
    """Test logging setup functionality."""
    log_dir = temp_dir / "logs"
    logger = setup_logging(log_dir)
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'PRGminer'
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 2  # File and console handlers
    
    # Test log file creation
    log_file = log_dir / 'PRGminer.log'
    assert log_file.exists()
    
    # Test logging
    test_message = "Test log message"
    logger.info(test_message)
    log_content = log_file.read_text()
    assert test_message in log_content

def test_cleanup_output_directory(temp_dir):
    """Test output directory cleanup."""
    # Create test directory with some content
    output_dir = temp_dir / "test_output"
    output_dir.mkdir()
    (output_dir / "test_file.txt").write_text("test")
    
    logger = logging.getLogger('PRGminer')
    cleanup_output_directory(output_dir, logger)
    
    assert not output_dir.exists()
    
    # Test cleanup of non-existent directory
    with pytest.raises(PRGminerError):
        cleanup_output_directory(temp_dir / "nonexistent", logger)
    
    # Test cleanup of file instead of directory
    test_file = temp_dir / "test.txt"
    test_file.write_text("test")
    with pytest.raises(PRGminerError):
        cleanup_output_directory(test_file, logger)

def test_extract_sequences(sample_fasta):
    """Test sequence extraction from FASTA."""
    seq_ids = ['seq1']
    sequences = extract_sequences(sample_fasta, seq_ids)
    
    assert len(sequences) == 1
    assert 'seq1' in sequences
    assert sequences['seq1'] == "MAEGEQVQSGEDLGSPVAQVLQKAREQGAQAAVLVVPPGEEQVQSAEDLGSPVAQVLQKA"
    
    # Test with non-existent sequence IDs
    sequences = extract_sequences(sample_fasta, ['nonexistent'])
    assert len(sequences) == 0

def test_create_summary_report(temp_dir, sample_results):
    """Test summary report creation."""
    output_dir = temp_dir / "summary_test"
    output_dir.mkdir()
    
    # Test Phase1 summary
    create_summary_report(sample_results, output_dir, "Phase1", append=False)
    summary_file = output_dir / "prediction_summary.txt"
    assert summary_file.exists()
    content = summary_file.read_text()
    assert "Phase1" in content
    assert "Total sequences analyzed: 2" in content
    
    # Test Phase2 summary append
    phase2_results = pd.DataFrame({
        'SampleID': ['seq1'],
        'Prediction': ['CNL'],
        'CNL': [0.8],
        'TIR': [0.2]
    })
    create_summary_report(phase2_results, output_dir, "Phase2", append=True)
    content = summary_file.read_text()
    assert "Phase2" in content
    assert "CNL: 1 sequences" in content

@patch('PRGminer.__main__.PRGPredictor')
@patch('PRGminer.__main__.utils.preprocess')
def test_dnn_prediction(mock_preprocess, mock_predictor, temp_dir, sample_fasta):
    """Test DNN prediction function."""
    output_dir = temp_dir / "dnn_test"
    output_dir.mkdir()
    
    # Mock preprocessed data
    mock_preprocess.return_value = {
        'Samples': np.array([[1, 2, 3], [4, 5, 6]]),
        'SeqID': ['seq1', 'seq2']
    }
    
    # Mock predictor behavior
    mock_instance = Mock()
    mock_predictor.return_value = mock_instance
    mock_instance.model.predict.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
    mock_instance.threshold = 0.5
    
    # Test Phase1 prediction
    results, message = DNN(sample_fasta, output_dir, "Phase1")
    assert isinstance(results, pd.DataFrame)
    assert message == ""
    assert 'Prediction' in results.columns
    assert len(results) == 2
    
    # Test error handling
    mock_instance.model.predict.side_effect = Exception("Test error")
    with pytest.raises(PRGminerError):
        DNN(sample_fasta, output_dir, "Phase1")

@patch('PRGminer.utils.argument_parser')
@patch('PRGminer.__main__.DNN')
def test_main_workflow(mock_dnn, mock_parser, temp_dir, sample_fasta):
    """Test main workflow."""
    # Setup mock arguments
    mock_args = Mock()
    mock_args.fasta_file = str(sample_fasta)
    mock_args.output_dir = str(temp_dir / "main_test")
    mock_args.output_file = "results.tsv"
    mock_args.level = "Phase1"
    mock_parser.return_value.parse_args.return_value = mock_args
    
    # Mock DNN results
    mock_dnn.return_value = (pd.DataFrame({
        'SampleID': ['seq1'],
        'Prediction': ['Rgene'],
        'Rgene': [0.8],
        'Non-Rgene': [0.2]
    }), "")
    
    # Test successful execution
    assert main() == 0
    
    # Verify output directory and files
    output_dir = Path(mock_args.output_dir)
    assert output_dir.exists()
    assert (output_dir / mock_args.output_file).exists()
    assert (output_dir / "PRGminer.log").exists()
    
    # Test error handling
    mock_dnn.side_effect = PRGminerError("Test error")
    assert main() == 1

@patch('PRGminer.utils.argument_parser')
def test_main_error_handling(mock_parser):
    # Test with invalid arguments
    mock_parser.return_value = None
    mock_parser.error.side_effect = SystemExit(2)
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2

    # Test with non-existent input file
    mock_args = MagicMock()
    mock_args.fasta_file = 'nonexistent.fasta'
    mock_args.output_dir = 'output'
    mock_args.output_file = 'results.txt'
    mock_args.level = 'Phase1'
    mock_parser.return_value = mock_args
    
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

    # Test with invalid output directory (file instead of directory)
    mock_args.fasta_file = 'tests/data/test.fasta'
    mock_args.output_dir = 'tests/data/test.fasta'  # Using a file as output dir
    
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

def test_main_phase2_workflow(temp_dir, sample_fasta):
    """Test main workflow with Phase2 prediction."""
    # Setup test environment
    output_dir = temp_dir / "phase2_test"
    output_file = "PRGminer_results.txt"
    
    # Mock command line arguments
    args = [
        'prgminer',
        '-i', str(sample_fasta),
        '-od', str(output_dir),
        '-o', output_file,
        '-l', 'Phase2'
    ]
    
    with patch('sys.argv', args), \
         patch('PRGminer.__main__.DNN') as mock_dnn:
        
        # Mock Phase1 results (with R-genes)
        phase1_results = pd.DataFrame({
            'SampleID': ['seq1', 'seq2'],
            'Prediction': ['Rgene', 'Rgene'],
            'Rgene': [0.8, 0.7],
            'Non-Rgene': [0.2, 0.3]
        })
        
        # Mock Phase2 results
        phase2_results = pd.DataFrame({
            'SampleID': ['seq1', 'seq2'],
            'Prediction': ['CNL', 'TIR'],
            'CNL': [0.8, 0.2],
            'TIR': [0.2, 0.8]
        })
        
        # Configure mock to return different results for Phase1 and Phase2
        mock_dnn.side_effect = [
            (phase1_results, ""),  # Phase1 result
            (phase2_results, "")   # Phase2 result
        ]
        
        # Run main function
        assert main() == 0
        
        # Verify intermediate files
        intermediate_dir = output_dir / "intermediate_files"
        assert (intermediate_dir / "Phase1_predictions.tsv").exists()
        assert (intermediate_dir / "phase2_input.fasta").exists()
        
        # Verify final output
        assert (output_dir / output_file).exists()
        assert (output_dir / "prediction_summary.txt").exists()

def test_main_phase2_no_rgenes(temp_dir, sample_fasta):
    """Test main workflow for Phase2 when no R-genes are identified."""
    output_dir = temp_dir / "output"
    output_file = "PRGminer_results.txt"

    # Mock DNN prediction to return no R-genes
    with patch('PRGminer.__main__.DNN') as mock_dnn:
        mock_dnn.return_value = (pd.DataFrame({
            'SampleID': ['seq1', 'seq2'],
            'Prediction': ['Non-Rgene', 'Non-Rgene'],
            'Rgene': [0.1, 0.2],
            'Non-Rgene': [0.9, 0.8]
        }), "")

        # Run main with Phase2
        with patch('sys.argv', [
            'prgminer',
            '-i', str(sample_fasta),
            '-od', str(output_dir),
            '-o', output_file,
            '-l', 'Phase2'
        ]):
            main()

        # Verify output
        assert output_dir.exists()
        assert (output_dir / output_file).exists()
        summary_content = (output_dir / "prediction_summary.txt").read_text()
        # Check that all sequences are Non-Rgene
        assert "Non-Rgene: 2 sequences" in summary_content
        # Check that Phase2 was skipped
        assert "No R-genes identified in Phase1, skipping Phase2" in summary_content 