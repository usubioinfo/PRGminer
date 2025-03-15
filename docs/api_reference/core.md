# Core API Reference

This document describes the core functionality of PRGminer.

## Main Classes

### PRGPredictor

The main class for making predictions on protein sequences.

```python
class PRGPredictor:
    def __init__(self, model_path=None, prediction_level='Phase2'):
        """
        Initialize the PRG predictor.
        
        Args:
            model_path (str): Path to the model files
            prediction_level (str): 'Phase1' or 'Phase2'
        """
        pass

    def predict(self, sequences):
        """
        Make predictions on protein sequences.
        
        Args:
            sequences (list): List of protein sequences
            
        Returns:
            dict: Prediction results with probabilities
        """
        pass
```

### SequenceProcessor

Handles sequence preprocessing and validation.

```python
class SequenceProcessor:
    def validate_sequence(sequence):
        """
        Validate protein sequence format.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            bool: True if valid, False otherwise
        """
        pass

    def encode_sequence(sequence):
        """
        Encode protein sequence for model input.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            numpy.array: Encoded sequence
        """
        pass
```

## Functions

### Data Loading

```python
def load_fasta(file_path):
    """
    Load sequences from FASTA file.
    
    Args:
        file_path (str): Path to FASTA file
        
    Returns:
        dict: Dictionary of sequence IDs and sequences
    """
    pass

def save_results(results, output_file):
    """
    Save prediction results to file.
    
    Args:
        results (dict): Prediction results
        output_file (str): Output file path
    """
    pass
```

## Constants

```python
# Supported prediction levels
PREDICTION_LEVELS = ['Phase1', 'Phase2']

# R-gene categories
RGENE_CATEGORIES = [
    'CNL', 'KIN', 'LYK', 'LECRK',
    'RLK', 'RLP', 'TIR', 'TNL'
]

# Model parameters
MAX_SEQUENCE_LENGTH = 2000
EMBEDDING_DIM = 128
```

## Error Classes

```python
class SequenceValidationError(Exception):
    """Raised when sequence validation fails"""
    pass

class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass
```

For detailed usage examples, please refer to the [Examples](../examples/basic_usage.md) section. 