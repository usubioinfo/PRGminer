"""
Tests for model prediction functionality
Author: Naveen Duhan (naveen.duhan@usu.edu)
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from PRGminer.model import PRGPredictor, PredictionError, validate_input_data, load_model_safely, pred_Phase1, pred_Phase2
from unittest.mock import Mock, patch
import os

@pytest.fixture
def sample_sequence():
    return {
        'seq1': 'MAEGEQVQSGEDLGSPVAQVLQKAREQGAQAAVLVVPPGEEQVQSAEDLGSPVAQVLQKA',
        'seq2': 'MTKFTILLFFLSVALASNAQPGCNQSQTLSPNWQNVFGASAASSCP'
    }

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Samples': [np.array([1, 2, 3]), np.array([4, 5, 6])],
        'Labels': ['A', 'B'],
        'SeqID': ['seq1', 'seq2']
    })

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path

@pytest.fixture
def invalid_dataframe():
    """Create invalid DataFrame for testing."""
    return pd.DataFrame({
        'Invalid': [1, 2, 3]
    })

@pytest.fixture
def malformed_dataframe():
    """Create DataFrame with missing values."""
    return pd.DataFrame({
        'Samples': [np.array([1, 2, 3]), None],
        'Labels': ['A', 'B'],
        'SeqID': ['seq1', 'seq2']
    })

@pytest.fixture
def sample_input_data():
    """Create sample input data for prediction functions."""
    return pd.DataFrame({
        'Samples': [np.array([0.1] * 400, dtype=np.float32), np.array([0.2] * 400, dtype=np.float32)],
        'Labels': [0, 0],
        'SeqID': ['seq1', 'seq2']
    })

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock = Mock()
    mock.predict.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
    return mock

def test_model_initialization():
    """Test model initialization."""
    predictor = PRGPredictor(testing=True)
    assert predictor.prediction_level == 'Phase1'
    assert predictor.threshold == 0.5
    assert predictor.max_sequence_length == 4000

    # Test invalid prediction level
    with pytest.raises(ValueError):
        PRGPredictor(prediction_level='InvalidPhase', testing=True)

def test_model_loading():
    """Test model loading functionality."""
    predictor = PRGPredictor(testing=True)
    assert predictor.model is not None
    
    # Test mock model behavior for Phase1
    phase1_predictor = PRGPredictor(prediction_level='Phase1', testing=True)
    mock_input = np.array([[1, 2, 3]])
    result = phase1_predictor.model.predict(mock_input)
    assert result.shape == (1, 2)
    
    # Test mock model behavior for Phase2
    phase2_predictor = PRGPredictor(prediction_level='Phase2', testing=True)
    result = phase2_predictor.model.predict(mock_input)
    assert result.shape == (1, 8)

def test_prediction(sample_sequence):
    """Test basic prediction functionality."""
    predictor = PRGPredictor(testing=True)
    results = predictor.predict(sample_sequence)
    assert isinstance(results, dict)
    assert len(results) == len(sample_sequence)
    
    # Test result structure
    for seq_id, result in results.items():
        assert 'predicted_class' in result
        assert 'probabilities' in result
        assert isinstance(result['probabilities'], dict)

def test_prediction_thresholds(sample_sequence):
    """Test prediction threshold behavior."""
    predictor = PRGPredictor(testing=True)
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        predictor.threshold = threshold
        results = predictor.predict(sample_sequence)
        assert isinstance(results, dict)
        assert len(results) == len(sample_sequence)

def test_phase1_prediction(sample_sequence):
    """Test Phase1 prediction specifics."""
    predictor = PRGPredictor(prediction_level='Phase1', testing=True)
    results = predictor.predict(sample_sequence)
    
    # Verify Phase1 specific output
    for seq_id, result in results.items():
        assert result['predicted_class'] in ['Rgene', 'Non-Rgene']
        assert len(result['probabilities']) == 2
        assert 'Rgene' in result['probabilities']
        assert 'Non-Rgene' in result['probabilities']

def test_phase2_prediction(sample_sequence):
    """Test Phase2 prediction specifics."""
    predictor = PRGPredictor(prediction_level='Phase2', testing=True)
    results = predictor.predict(sample_sequence)
    
    # Verify Phase2 specific output
    expected_classes = ['CNL', 'KIN', 'LYK', 'LECRK', 'RLK', 'RLP', 'TIR', 'TNL']
    for seq_id, result in results.items():
        assert result['predicted_class'] in expected_classes
        assert len(result['probabilities']) == 8
        for class_name in expected_classes:
            assert class_name in result['probabilities']

def test_model_versioning():
    """Test model versioning functionality."""
    predictor = PRGPredictor(testing=True)
    assert isinstance(predictor.version, str)
    assert predictor.version == "1.0.0"

def test_error_handling():
    """Test error handling in prediction pipeline."""
    predictor = PRGPredictor(testing=True)
    
    # Test invalid sequence type
    with pytest.raises(TypeError):
        predictor.predict(["invalid type"])
    
    # Test empty sequence dictionary
    with pytest.raises(ValueError):
        predictor.predict({})
    
    # Test sequence too long
    long_sequence = {"test_seq": "A" * 5000}
    with pytest.raises(ValueError):
        predictor.predict(long_sequence)

def test_validate_input_data(sample_dataframe):
    """Test input data validation."""
    # Test valid input
    validate_input_data(sample_dataframe)
    
    # Test invalid DataFrame
    with pytest.raises(ValueError):
        validate_input_data("not a dataframe")
    
    # Test missing columns
    invalid_df = pd.DataFrame({'Samples': [1, 2]})
    with pytest.raises(ValueError):
        validate_input_data(invalid_df)
    
    # Test empty DataFrame
    empty_df = pd.DataFrame(columns=['Samples', 'Labels', 'SeqID'])
    with pytest.raises(ValueError):
        validate_input_data(empty_df)

def test_validate_input_data_invalid_type():
    """Test validate_input_data with invalid input types."""
    # Test with non-DataFrame input
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        validate_input_data("not a dataframe")
    
    # Test with list input
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        validate_input_data([1, 2, 3])

def test_validate_input_data_missing_columns(invalid_dataframe):
    """Test validate_input_data with missing columns."""
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input_data(invalid_dataframe)

def test_validate_input_data_empty():
    """Test validate_input_data with empty DataFrame."""
    empty_df = pd.DataFrame(columns=['Samples', 'Labels', 'SeqID'])
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        validate_input_data(empty_df)

def test_validate_input_data_missing_values(malformed_dataframe):
    """Test validate_input_data with missing values."""
    with pytest.raises(ValueError, match="Sample data contains missing values"):
        validate_input_data(malformed_dataframe)

def test_load_model_safely():
    """Test safe model loading."""
    # Test non-existent model
    with pytest.raises(PredictionError):
        load_model_safely("nonexistent_model.h5")
    
    # Test invalid model path
    with pytest.raises(PredictionError):
        load_model_safely("")

def test_load_model_safely_nonexistent():
    """Test load_model_safely with non-existent model file."""
    with pytest.raises(PredictionError, match="Model file not found"):
        load_model_safely("nonexistent_model.h5")

def test_load_model_safely_invalid_path():
    """Test load_model_safely with invalid model path."""
    with pytest.raises(PredictionError):
        load_model_safely("")

def test_load_model_safely_corrupted(tmp_path):
    """Test load_model_safely with corrupted model file."""
    model_path = tmp_path / "corrupted_model.h5"
    model_path.write_text("corrupted data")
    
    with pytest.raises(PredictionError):
        load_model_safely(model_path)

def test_predictor_invalid_sequence_type():
    """Test PRGPredictor with invalid sequence type."""
    predictor = PRGPredictor(testing=True)
    with pytest.raises(TypeError, match="Input must be a sequence dictionary or string"):
        predictor.predict([1, 2, 3])

def test_predictor_empty_sequence():
    """Test PRGPredictor with empty sequence dictionary."""
    predictor = PRGPredictor(testing=True)
    with pytest.raises(ValueError, match="Input sequence dictionary is empty"):
        predictor.predict({})

def test_predictor_sequence_too_long():
    """Test PRGPredictor with sequence exceeding max length."""
    predictor = PRGPredictor(testing=True)
    long_sequence = "A" * 5000
    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        predictor.predict({"test": long_sequence})

def test_predictor_phase2_classification():
    """Test PRGPredictor Phase2 classification."""
    predictor = PRGPredictor(prediction_level='Phase2', testing=True)
    sequences = {
        'seq1': 'MAEGEQVQSGEDLGSPVAQVLQKAREQGAQAAVLVVPPGEEQVQSAEDLGSPVAQVLQKA',
        'seq2': 'MTKFTILLFFLSVALASNAQPGCNQSQTLSPNWQNVFGASAASSCP'
    }
    results = predictor.predict(sequences)
    
    # Verify Phase2 specific output
    expected_classes = ['CNL', 'KIN', 'LYK', 'LECRK', 'RLK', 'RLP', 'TIR', 'TNL']
    for seq_id, result in results.items():
        assert result['predicted_class'] in expected_classes
        assert len(result['probabilities']) == 8
        assert all(cls in result['probabilities'] for cls in expected_classes)
        assert abs(sum(result['probabilities'].values()) - 1.0) < 1e-6

def test_predictor_single_sequence():
    """Test PRGPredictor with single sequence input."""
    predictor = PRGPredictor(testing=True)
    sequence = "MAEGEQVQSGEDLGSPVAQVLQKAREQGAQAAVLVVPPGEEQVQSAEDLGSPVAQVLQKA"
    result = predictor.predict({"test": sequence})
    
    assert isinstance(result, dict)
    assert "test" in result
    assert "predicted_class" in result["test"]
    assert "probabilities" in result["test"]
    assert len(result["test"]["probabilities"]) == 2

def test_predictor_threshold_behavior():
    """Test PRGPredictor threshold behavior."""
    predictor = PRGPredictor(testing=True)
    sequence = "MAEGEQVQSGEDLGSPVAQVLQKAREQGAQAAVLVVPPGEEQVQSAEDLGSPVAQVLQKA"
    
    # Test with different thresholds
    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        predictor.threshold = threshold
        result = predictor.predict({"test": sequence})
        prob = result["test"]["probabilities"]["Rgene"]
        predicted_class = result["test"]["predicted_class"]
        assert (predicted_class == "Rgene") == (prob >= threshold)

@pytest.mark.integration
def test_prediction_workflow(sample_sequence):
    """Integration test for full prediction workflow."""
    # Test Phase1 workflow
    phase1_predictor = PRGPredictor(prediction_level='Phase1', testing=True)
    phase1_results = phase1_predictor.predict(sample_sequence)
    
    # Test Phase2 workflow
    phase2_predictor = PRGPredictor(prediction_level='Phase2', testing=True)
    phase2_results = phase2_predictor.predict(sample_sequence)
    
    # Verify workflow connection
    rgene_sequences = {
        seq_id: seq for seq_id, seq in sample_sequence.items()
        if phase1_results[seq_id]['predicted_class'] == 'Rgene'
    }
    if rgene_sequences:
        phase2_results_filtered = phase2_predictor.predict(rgene_sequences)
        assert all(result['predicted_class'] in ['CNL', 'KIN', 'LYK', 'LECRK', 'RLK', 'RLP', 'TIR', 'TNL'] 
                  for result in phase2_results_filtered.values())

def test_pred_Phase1(tmp_path, sample_input_data, mock_model):
    """Test Phase1 prediction functionality."""
    outfile = tmp_path / "phase1_results.tsv"
    
    with patch('PRGminer.model.load_model_safely', return_value=mock_model):
        # Test successful prediction
        rgene_ids, results = pred_Phase1(sample_input_data, outfile)
        
        assert isinstance(rgene_ids, list)
        assert isinstance(results, pd.DataFrame)
        assert len(rgene_ids) == 1  # One R-gene prediction
        assert len(results) == 2  # Two total predictions
        assert all(col in results.columns for col in ['SampleID', 'Prediction', 'Rgene', 'Non-Rgene'])
        assert outfile.exists()
        
        # Test prediction failure
        mock_model.predict.side_effect = Exception("Test error")
        with pytest.raises(PredictionError, match="Model prediction failed"):
            pred_Phase1(sample_input_data, outfile)

def test_pred_Phase1_invalid_input(tmp_path):
    """Test Phase1 prediction with invalid input."""
    outfile = tmp_path / "phase1_results.tsv"
    
    # Test with invalid DataFrame
    invalid_df = pd.DataFrame({'Invalid': [1, 2]})
    with pytest.raises(ValueError):
        pred_Phase1(invalid_df, outfile)
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame(columns=['Samples', 'Labels', 'SeqID'])
    with pytest.raises(ValueError):
        pred_Phase1(empty_df, outfile)

def test_pred_Phase2(tmp_path, sample_input_data, mock_model):
    """Test Phase2 prediction functionality."""
    outfile = tmp_path / "phase2_results.tsv"
    
    # Mock model for Phase2 (8 classes)
    mock_model.predict.return_value = np.array([
        [0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05],
        [0.1, 0.5, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
    ])
    
    with patch('PRGminer.model.load_model_safely', return_value=mock_model):
        # Test successful prediction
        results = pred_Phase2(sample_input_data, outfile)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2  # Two predictions
        assert all(col in results.columns for col in ['SampleID', 'Prediction', 'CNL', 'KIN', 'LYK', 'LECRK', 'RLK', 'RLP', 'TIR', 'TNL'])
        assert outfile.exists()
        
        # Test prediction failure
        mock_model.predict.side_effect = Exception("Test error")
        with pytest.raises(PredictionError, match="Model prediction failed"):
            pred_Phase2(sample_input_data, outfile)

def test_pred_Phase2_invalid_input(tmp_path):
    """Test Phase2 prediction with invalid input."""
    outfile = tmp_path / "phase2_results.tsv"
    
    # Test with invalid DataFrame
    invalid_df = pd.DataFrame({'Invalid': [1, 2]})
    with pytest.raises(ValueError):
        pred_Phase2(invalid_df, outfile)
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame(columns=['Samples', 'Labels', 'SeqID'])
    with pytest.raises(ValueError):
        pred_Phase2(empty_df, outfile)

def test_pred_Phase2_io_error(tmp_path, sample_input_data, mock_model):
    """Test Phase2 prediction with IO errors."""
    # Test with invalid output path
    with patch('PRGminer.model.load_model_safely', return_value=mock_model):
        with pytest.raises(IOError):
            pred_Phase2(sample_input_data, "/invalid/path/results.tsv") 