"""
Neural Network Prediction Module for PRGminer

This module provides functionality for predicting and classifying resistance genes (R-genes) 
using pre-trained neural network models. It consists of two main phases:
1. Phase 1: Binary classification of sequences as R-genes or non-R-genes
2. Phase 2: Multi-class classification of R-genes into specific categories

The module uses Keras models and handles data preprocessing, prediction, and result formatting.

Dependencies:
    - keras
    - numpy
    - pandas
    - pkg_resources
"""

from keras.models import load_model, Sequential
import numpy as np
import pandas as pd
import os
import logging
pd.options.mode.chained_assignment = None
import pkg_resources
from PRGminer import utils
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from .sequence import SequenceProcessor
from unittest.mock import Mock

# Model paths
PHASE1_MODEL_PATH = pkg_resources.resource_filename('PRGminer', 'models/prgminer_phase1.h5')
PHASE2_MODEL_PATH = pkg_resources.resource_filename('PRGminer', 'models/prgminer_phase2.h5')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionError(Exception):
    """Custom exception for prediction-related errors."""
    pass

def validate_input_data(df, required_columns=['Samples', 'Labels', 'SeqID']):
    """
    Validates input DataFrame for required columns and data integrity.

    Args:
        df (pandas.DataFrame): Input DataFrame to validate
        required_columns (list): List of required column names

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    if df['Samples'].isna().any():
        raise ValueError("Sample data contains missing values")

def load_model_safely(model_path):
    """
    Safely loads a Keras model with error handling.

    Args:
        model_path (str): Path to the model file

    Returns:
        keras.Model: Loaded model

    Raises:
        PredictionError: If model loading fails
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return load_model(model_path, compile=False)
    except Exception as e:
        raise PredictionError(f"Failed to load model from {model_path}: {str(e)}")

def pred_Phase1(df, outfile):
    """
    Performs Phase 1 prediction to identify R-genes.

    Args:
        df (pandas.DataFrame): Input DataFrame containing:
            - Samples: Sequence data in the appropriate format
            - Labels: Corresponding labels
            - SeqID: Sequence identifiers
        outfile (str): Path to save the prediction results

    Returns:
        tuple: (list of R-gene IDs, DataFrame with prediction results)

    Raises:
        ValueError: If input validation fails
        PredictionError: If prediction process fails
        IOError: If file operations fail
    """
    try:
        # Validate input data
        validate_input_data(df)
        logger.info("Starting Phase 1 prediction")

        # Extract and reshape data
        testData = np.stack(df['Samples'].values)
        SampleID = df['SeqID']
        try:
            Samples = testData.reshape(testData.shape[0], 1, testData.shape[1], 1)
        except Exception as e:
            raise ValueError(f"Failed to reshape input data: {str(e)}")

        # Load model and make predictions
        try:
            model = load_model_safely(PHASE1_MODEL_PATH)
            predictions = model.predict(Samples)
            # Model outputs are already probabilities, convert to percentages
            predictions_pct = predictions * 100
        except Exception as e:
            raise PredictionError(f"Model prediction failed: {str(e)}")

        # Process predictions
        try:
            # Format probabilities as strings with 4 decimal places
            rgene_probs = ['{:.4f}'.format(x) for x in predictions_pct[:, 0]]
            non_rgene_probs = ['{:.4f}'.format(x) for x in predictions_pct[:, 1]]
            
            results = pd.DataFrame({
                'SampleID': SampleID,
                'Prediction': ['Rgene' if float(p)/100 >= 0.5 else 'Non-Rgene' for p in rgene_probs],
                'Rgene': non_rgene_probs,
                'Non-Rgene': rgene_probs
            })
            
            # Save with tab separator
            results.to_csv(outfile, sep='\t', index=False)
            
            rgene_ids = results[results['Prediction'] == 'Rgene']['SampleID'].tolist()
            return rgene_ids, results
        except Exception as e:
            raise IOError(f"Failed to save results: {str(e)}")
    except Exception as e:
        logger.error(f"Phase 1 prediction failed: {str(e)}")
        raise

def pred_Phase2(df, outfile):
    """
    Performs Phase 2 prediction to classify R-genes into specific categories.

    Args:
        df (pandas.DataFrame): Input DataFrame containing:
            - Samples: Sequence data in the appropriate format
            - Labels: Corresponding labels
            - SeqID: Sequence identifiers
        outfile (str): Path to save the prediction results

    Returns:
        pandas.DataFrame: Complete prediction results with probabilities for each class

    Raises:
        ValueError: If input validation fails
        PredictionError: If prediction process fails
        IOError: If file operations fail
    """
    try:
        # Validate input data
        validate_input_data(df)
        logger.info("Starting Phase 2 prediction")

        # Extract and reshape data
        testData = np.stack(df['Samples'].values)
        SampleID = df['SeqID']
        try:
            Samples = testData.reshape(testData.shape[0], 1, testData.shape[1], 1)
        except Exception as e:
            raise ValueError(f"Failed to reshape input data: {str(e)}")

        # Load model and make predictions
        try:
            model = load_model_safely(PHASE2_MODEL_PATH)
            predictions = model.predict(Samples)
        except Exception as e:
            raise PredictionError(f"Model prediction failed: {str(e)}")

        # Process predictions
        try:
            class_names = ['CNL', 'KIN', 'LYK', 'LECRK', 'RLK', 'RLP', 'TIR', 'TNL']
            
            # Create initial DataFrame with SampleID and Prediction
            data = {
                'SampleID': SampleID,
                'Prediction': [class_names[i] for i in np.argmax(predictions, axis=1)]
            }
            
            # Add probabilities for each class (convert to 0-100 scale and round to 4 decimals)
            predictions_pct = predictions * 100
            for i, name in enumerate(class_names):
                data[name] = [f"{x:.4f}" for x in predictions_pct[:, i]]
            
            results = pd.DataFrame(data)
            
            # Save with tab separator
            results.to_csv(outfile, sep='\t', index=False)
            
            return results
        except Exception as e:
            raise IOError(f"Failed to save results: {str(e)}")
    except Exception as e:
        logger.error(f"Phase 2 prediction failed: {str(e)}")
        raise

class PRGPredictor:
    """Main class for R-gene prediction."""
    
    def __init__(self, prediction_level: str = "Phase1", testing: bool = False):
        """Initialize PRG predictor with specified prediction level."""
        if prediction_level not in ['Phase1', 'Phase2']:
            raise ValueError("Invalid prediction level. Must be 'Phase1' or 'Phase2'")
        self.prediction_level = prediction_level
        self.threshold = 0.5  # Default threshold for raw probabilities
        self.max_sequence_length = 4000  # Maximum allowed sequence length
        self.testing = testing
        self.version = "1.0.0"
        self.processor = SequenceProcessor()
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the model based on the prediction level."""
        if self.testing:
            # Create a mock model for testing
            mock_model = Mock()
            if self.prediction_level == 'Phase1':
                mock_model.predict.return_value = np.array([[0.2, 0.8]])  # Shape (1,2) as probabilities
            else:
                mock_model.predict.return_value = np.array([[0.7, 0.1, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025]])  # Shape (1,8) as probabilities
            return mock_model
            
        # Load actual model
        if self.prediction_level == 'Phase1':
            model_path = PHASE1_MODEL_PATH
        elif self.prediction_level == 'Phase2':
            model_path = PHASE2_MODEL_PATH
        else:
            raise ValueError("Invalid prediction level")
        
        return load_model_safely(model_path)
    
    def predict(self, sequences):
        """Predict R-gene classes for input sequences."""
        # Validate input type
        if not isinstance(sequences, (dict, str)):
            raise TypeError("Input must be a sequence dictionary or string")
        
        # Validate input is not empty
        if isinstance(sequences, dict) and not sequences:
            raise ValueError("Input sequence dictionary is empty")
            
        # Validate sequence length
        if isinstance(sequences, str):
            if len(sequences) > self.max_sequence_length:
                raise ValueError(f"Sequence length {len(sequences)} exceeds maximum allowed length of {self.max_sequence_length}")
        else:
            for seq_id, seq in sequences.items():
                if len(seq) > self.max_sequence_length:
                    raise ValueError(f"Sequence {seq_id} length {len(seq)} exceeds maximum allowed length of {self.max_sequence_length}")
        
        results = {}
        for seq_id, sequence in sequences.items():
            # Preprocess sequence
            processed = self.processor.preprocess(sequence)
            
            # Make prediction
            pred = self.model.predict(processed, verbose=0)
            
            if self.prediction_level == 'Phase1':
                # Binary classification
                class_names = ['Rgene', 'Non-Rgene']
                # Model outputs are already probabilities
                non_rgene_prob = float(pred[0][0])
                rgene_prob = float(pred[0][1])
                probs = [(100 * non_rgene_prob).round(4), (100 * rgene_prob).round(4)]
                predicted_class = class_names[1] if rgene_prob >= self.threshold else class_names[0]
            else:
                # Multi-class classification
                class_names = ['CNL', 'KIN', 'LYK', 'LECRK', 'RLK', 'RLP', 'TIR', 'TNL']
                # For Phase2, apply softmax
                pred = tf.nn.softmax(pred, axis=1).numpy()
                probs = (pred[0] * 100).round(4).tolist()  # Convert to percentages and round
                predicted_class = class_names[np.argmax(pred[0])]

            results[seq_id] = {
                'predicted_class': predicted_class,
                'probabilities': dict(zip(class_names, probs))
            }
        
        return results