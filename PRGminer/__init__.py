"""
PRGminer: Deep Neural Network-Based Plant Resistance Gene Prediction
Author: Naveen Duhan (naveen.duhan@usu.edu)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

A comprehensive bioinformatics package for predicting and classifying plant resistance genes
using state-of-the-art deep learning models. This package provides tools for sequence analysis,
feature extraction, and multi-class prediction of R-genes.

Key Features:
    - Two-phase prediction pipeline
    - Advanced deep learning models
    - Comprehensive sequence analysis
    - Efficient data preprocessing
    - Detailed output reporting

Modules:
    - model: Neural network prediction models and functions
        - PRGPredictor: Main predictor class
    - sequence: Sequence processing and analysis
        - SequenceProcessor: Processor for sequence analysis
    - utils: Utility functions and data processing
        - load_config: Load configuration from file
        - save_predictions: Save predictions to file
        - validate_input_file: Validate input file
        - setup_logging: Set up logging
        - get_version: Get package version

Usage Example:
    >>> from PRGminer.model import PRGPredictor
    >>> from PRGminer.sequence import SequenceProcessor
    >>> from PRGminer.utils import preprocess
    >>> 
    >>> # Preprocess sequences
    >>> data = preprocess('input.fasta', 'DPC', [0, 0], 400)
    >>> 
    >>> # Predict using PRGPredictor
    >>> predictor = PRGPredictor()
    >>> rgenes, results = predictor.predict(data)

Dependencies:
    Required:
        - TensorFlow >= 2.0
        - Keras
        - NumPy
        - Pandas
        - Biopython
        - Scikit-learn
    
    Optional:
        - CUDA (for GPU acceleration)

Author: Naveen Duhan
Lab: KAABiL (Kaundal Artificial Intelligence & Advanced Bioinformatics Lab)
Version: 0.1.0
License: GPL-3.0
"""

__version__ = "0.1.0"
__author__ = "Naveen Duhan"
__email__ = "naveen.duhan@usu.edu"
__license__ = "GPL-3.0"
__citation__ = '''
@article{PRGminer2023,
    author = {Duhan, Naveen and Kaundal, Rakesh},
    title = {PRGminer: Deep learning-based prediction and classification of plant resistance genes},
    journal = {Bioinformatics},
    year = {2023}
}
'''

from .model import PRGPredictor
from .sequence import SequenceProcessor
from .utils import (
    load_config,
    save_predictions,
    validate_input_file,
    setup_logging,
    get_version
)

# Define public API
__all__ = [
    'PRGPredictor',
    'SequenceProcessor',
    'load_config',
    'save_predictions',
    'validate_input_file',
    'setup_logging',
    'get_version',
]

