#!/usr/bin/python

"""
PRGminer: Plant Resistance Gene Prediction Tool

This is the main entry point for PRGminer, a deep learning-based tool for predicting 
and classifying plant resistance genes (R-genes). The tool operates in two phases:

Phase 1: Binary classification of sequences as R-genes or non-R-genes
Phase 2: Detailed classification of identified R-genes into specific categories

Author: Naveen Duhan
Lab: KAABiL (Kaundal Artificial Intelligence & Advanced Bioinformatics Lab)
Version: 0.1

Usage:
    python -m PRGminer [options] -f <fasta_file> -o <output_dir> -l <level>

Dependencies:
    - pandas
    - tensorflow
    - keras
    - numpy
"""

# Standard library imports
import os
import sys
import time
import logging
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict, Union

# Force CPU-only mode and suppress TensorFlow warnings - must be before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent CUDA memory errors
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda/'  # Suppress XLA warnings
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'  # Disable XLA devices
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'] = 'false'  # Disable eager execution warnings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Force PCI bus ID order
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Force synchronous CUDA execution
os.environ['TF_USE_CUDNN'] = '0'  # Disable cuDNN
os.environ['TF_DISABLE_CUDNN_RNN'] = '1'  # Disable cuDNN RNN ops
os.environ['TF_DISABLE_CUDNN_TENSOR_OP_MATH'] = '1'  # Disable cuDNN tensor ops

# Suppress future warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*experimental feature.*')
warnings.filterwarnings('ignore', message='.*cuda.*')
warnings.filterwarnings('ignore', message='.*gpu.*')

# Configure logging before imports
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').addHandler(logging.NullHandler())
logging.getLogger('tensorboard').setLevel(logging.ERROR)
logging.getLogger('tensorflow_hub').setLevel(logging.ERROR)
logging.getLogger('h5py').setLevel(logging.ERROR)
logging.getLogger('numexpr').setLevel(logging.ERROR)

# Third-party imports
import pandas as pd
import tensorflow as tf

# Configure TensorFlow to use CPU only - must be done before any TF operations
tf.config.set_visible_devices([], 'GPU')  # Hide all GPUs
tf.config.threading.set_inter_op_parallelism_threads(1)  # Limit to single thread
tf.config.threading.set_intra_op_parallelism_threads(1)  # Limit to single thread

# Additional TensorFlow warning suppression
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
tf.get_logger().addHandler(logging.NullHandler())

# Configure Numpy
import numpy as np
np.seterr(all='ignore')  # Suppress numpy warnings

# Disable TensorFlow debugging and performance logs
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Local imports
from PRGminer import (
    model,
    utils,
    sequence,
    __version__
)
from PRGminer.model import PRGPredictor
from PRGminer.sequence import SequenceProcessor

# Additional TensorFlow CPU configuration
tf.keras.backend.set_floatx('float32')  # Use float32 for better CPU performance
tf.config.optimizer.set_jit(False)  # Disable XLA JIT compilation
tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})  # Disable meta optimizer

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging configuration"""
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path in the main output directory
    log_file = output_dir / 'PRGminer.log'
    
    # Configure logging format
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    
    # Get logger and set propagate to False to prevent duplicate logs
    logger = logging.getLogger('PRGminer')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class PRGminerError(Exception):
    """Custom exception for PRGminer-related errors."""
    pass

def validate_inputs(fasta_file: Union[str, Path], output_dir: Union[str, Path], level: str) -> None:
    """
    Validates input parameters for the PRGminer tool.

    Args:
        fasta_file: Path to input FASTA file
        output_dir: Path to output directory
        level: Prediction level (Phase1 or Phase2)

    Raises:
        PRGminerError: If validation fails
    """
    fasta_file = Path(fasta_file)
    output_dir = Path(output_dir)
    
    # Validate FASTA file
    if not fasta_file.exists():
        raise PRGminerError(f"Input FASTA file not found: {fasta_file}")
    
    if not fasta_file.is_file():
        raise PRGminerError(f"Input FASTA path is not a file: {fasta_file}")
    
    # Validate prediction level
    valid_levels = ["Phase1", "Phase2"]
    if level not in valid_levels:
        raise PRGminerError(f"Invalid prediction level. Must be one of: {', '.join(valid_levels)}")
    
    # Validate output directory path
    if output_dir.exists() and not output_dir.is_dir():
        raise PRGminerError(f"Output path exists but is not a directory: {output_dir}")

def DNN(fasta_file: Union[str, Path], output_dir: Union[str, Path], level: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Performs deep neural network prediction for R-gene classification.

    Args:
        fasta_file: Path to the input FASTA file
        output_dir: Directory to save output files
        level: Prediction level (Phase1 or Phase2)

    Returns:
        Tuple containing:
            - Prediction results DataFrame or None if error
            - Error message if any

    Raises:
        PRGminerError: If prediction process fails
    """
    logger = logging.getLogger('PRGminer')
    
    try:
        # Convert paths to Path objects
        fasta_file = Path(fasta_file)
        output_dir = Path(output_dir)
        
        # Validate inputs
        validate_inputs(fasta_file, output_dir, level)
        
        # Log prediction start
        logger.info(f"Starting {level} prediction")
        
        # Initialize components with CPU configuration
        with tf.device('/CPU:0'):
            predictor = PRGPredictor(prediction_level=level)
            logger.info(f"Initialized {level} predictor")
            
            # Extract DPC features from sequences
            logger.info("Extracting DPC features from sequences")
            preprocessed_data = utils.preprocess(
                file=str(fasta_file),
                feat='DPC',
                class_label=[0],  # Dummy label since we're only predicting
                size=400  # Required feature size for the model
            )
            
            if not isinstance(preprocessed_data, dict) or 'Samples' not in preprocessed_data:
                raise PRGminerError("Failed to preprocess sequences")
                
            # Reshape features to match model input shape
            samples = preprocessed_data['Samples']
            seq_ids = preprocessed_data['SeqID']
            logger.info(f"Successfully preprocessed {len(seq_ids)} sequences")
            
            try:
                # Reshape to (batch_size, 1, 400, 1) as expected by the model
                reshaped_samples = samples.reshape(samples.shape[0], 1, samples.shape[1], 1)
                logger.info(f"Reshaped features to {reshaped_samples.shape}")
            except Exception as e:
                raise PRGminerError(f"Failed to reshape features: {str(e)}")
            
            # Make predictions
            logger.info("Running model predictions")
            try:
                predictions = predictor.model.predict(reshaped_samples, verbose=0)
                logger.info("Model predictions completed successfully")
            except Exception as e:
                raise PRGminerError(f"Model prediction failed: {str(e)}")
        
        # Process predictions based on level
        if level == 'Phase1':
            class_names = ['Rgene', 'Non-Rgene']
            results = []
            for seq_id, prob in zip(seq_ids, predictions):
                # Get Rgene probability
                rgene_prob = float(prob[0])  # Second output is Rgene probability
                non_rgene_prob = float(prob[1])  # First output is Non-Rgene probability
                
                # Convert to percentages and round
                rgene_pct = round(rgene_prob * 100, 4)
                non_rgene_pct = round(non_rgene_prob * 100, 4)
                
                results.append({
                    'SampleID': seq_id,
                    'Prediction': class_names[0] if rgene_prob >= predictor.threshold else class_names[1],
                    'Non-Rgene': f"{non_rgene_pct:.4f}",
                    'Rgene': f"{rgene_pct:.4f}"
                })
            logger.info(f"Processed Phase1 predictions with threshold {predictor.threshold}")
        else:  # Phase2
            class_names = ['CNL', 'KIN', 'LYK', 'LECRK', 'RLK', 'RLP', 'TIR', 'TNL']
            results = []
            for seq_id, probs in zip(seq_ids, predictions):
                # Convert probabilities to percentages and round
                probs_pct = [round(float(p) * 100, 4) for p in probs]
                result = {
                    'SampleID': seq_id,
                    'Prediction': class_names[np.argmax(probs)]
                }
                # Add formatted probabilities for each class
                for name, prob in zip(class_names, probs_pct):
                    result[name] = f"{prob:.4f}"
                results.append(result)
            logger.info("Processed Phase2 predictions with class probabilities")
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Save detailed results in intermediate directory
        intermediate_dir = output_dir / 'intermediate_files'
        intermediate_dir.mkdir(exist_ok=True)
        
        output_file = intermediate_dir / f"{level}_predictions.tsv"
        df_results.to_csv(output_file, sep='\t', index=False)
        logger.info(f"Detailed results saved to {output_file}")
        
        return df_results, ""

    except Exception as e:
        logger.error(f"DNN prediction failed: {str(e)}")
        raise PRGminerError(f"Prediction failed: {str(e)}")

def create_summary_report(results: pd.DataFrame, output_dir: Path, level: str, append: bool = False) -> None:
    """
    Creates a summary report showing the count of sequences in each predicted class.

    Args:
        results: DataFrame containing prediction results
        output_dir: Directory to save the summary
        level: Prediction level (Phase1 or Phase2)
        append: Whether to append to existing summary file
    """
    logger = logging.getLogger('PRGminer')
    
    try:
        # Get class counts
        class_counts = results['Prediction'].value_counts().sort_index()
        
        # Create summary text
        summary_lines = []
        if not append:
            summary_lines.extend([
                "PRGminer Prediction Summary",
                "========================\n"
            ])
        
        summary_lines.extend([
            f"Prediction Level: {level}",
            f"Total sequences analyzed: {len(results)}",
            "\nClass Distribution:",
            "-------------------"
        ])
        
        # Add counts for each class
        for class_name, count in class_counts.items():
            percentage = (count / len(results)) * 100
            summary_lines.append(f"{class_name}: {count} sequences ({percentage:.2f}%)")
            
        # Add warning if no R-genes found in Phase1
        if level == "Phase1" and "Rgene" not in class_counts:
            summary_lines.append("\nNo R-genes identified in Phase1, skipping Phase2")

        summary_lines.append("\n" + "="*50 + "\n")  # Add separator between phases
        
        # Save summary
        summary_file = output_dir / 'prediction_summary.txt'
        mode = 'a' if append else 'w'
        with open(summary_file, mode) as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Summary report {'updated' if append else 'saved'} to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to create summary report: {str(e)}")

def extract_sequences(fasta_file: Path, seq_ids: list) -> Dict[str, str]:
    """
    Extract sequences from a FASTA file for given sequence IDs.
    
    Args:
        fasta_file: Path to the FASTA file
        seq_ids: List of sequence IDs to extract
        
    Returns:
        Dictionary mapping sequence IDs to their sequences
    """
    from Bio import SeqIO
    sequences = {}
    for record in SeqIO.parse(str(fasta_file), "fasta"):
        if record.id in seq_ids:
            sequences[record.id] = str(record.seq)
    return sequences

def cleanup_output_directory(output_dir: Path, logger: logging.Logger) -> None:
    """
    Check for and cleanup previous output directory.
    
    Args:
        output_dir: Path to the output directory
        logger: Logger instance for logging messages
        
    Raises:
        PRGminerError: If cleanup fails or path is invalid
    """
    if not output_dir.exists():
        raise PRGminerError(f"Output directory does not exist: {output_dir}")
        
    if not output_dir.is_dir():
        raise PRGminerError(f"Output path exists but is not a directory: {output_dir}")
        
    try:
        shutil.rmtree(output_dir)
    except Exception as e:
        raise PRGminerError(f"Failed to remove previous output directory: {str(e)}")

def main() -> int:
    """
    Main entry point for the PRGminer tool.
    """
    try:
        # Parse command line arguments first
        try:
            parser = utils.argument_parser(version=__version__)
            options = parser.parse_args()
        except Exception as e:
            sys.exit(2)  # Exit with code 2 for argument parsing errors
        
        # Extract and validate parameters
        fasta_file = Path(options.fasta_file)
        output_dir = Path(options.output_dir)
        output = options.output_file
        level = options.level

        # Validate inputs before proceeding
        try:
            validate_inputs(fasta_file, output_dir, level)
        except PRGminerError as e:
            logger = logging.getLogger('PRGminer')
            logger.error(str(e))
            sys.exit(1)  # Exit with code 1 for validation errors

        # Get the logger and clear any existing handlers
        logger = logging.getLogger('PRGminer')
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        
        # Setup initial console-only logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        logger.info("Starting PRGminer v0.1.0")
        
        # Check and cleanup output directory if it exists
        if output_dir.exists() and output_dir.is_dir():
            logger.warning(f"Previous output directory '{output_dir}' exists and will be removed")
            cleanup_output_directory(output_dir, logger)
            logger.info(f"Successfully removed previous output directory: {output_dir}")
        
        # Create output directory and add file handler
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / 'PRGminer.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        # Log initial information
        logger.info("PRGminer started")
        logger.info(f"Version: {__version__}")
        logger.info(f"Input file: {fasta_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Prediction level: {level}")

        start = time.time()

        # Create intermediate directory
        try:
            intermediate_dir = output_dir / 'intermediate_files'
            intermediate_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PRGminerError(f"Failed to create output directory: {str(e)}")

        # Always run Phase1 first to identify R-genes
        logger.info("Starting Phase1 prediction to identify R-genes")
        phase1_results, message = DNN(fasta_file, output_dir, "Phase1")
        
        # Save Phase1 results and create summary
        phase1_file = intermediate_dir / "Phase1_predictions.tsv"
        phase1_results.to_csv(phase1_file, sep="\t", index=False)
        logger.info(f"Phase1 results saved to: {phase1_file}")
        create_summary_report(phase1_results, output_dir, "Phase1", append=False)
        
        # Filter R-genes for Phase2 if requested
        if level == "Phase2":
            # Get R-gene predictions
            r_genes = phase1_results[phase1_results['Prediction'] == 'Rgene']
            if len(r_genes) == 0:
                logger.warning("No R-genes identified in Phase1, skipping Phase2")
                results = phase1_results
            else:
                logger.info(f"Identified {len(r_genes)} R-genes for Phase2 analysis")
                
                # Extract R-gene sequences
                r_gene_ids = r_genes['SampleID'].tolist()
                r_gene_sequences = extract_sequences(fasta_file, r_gene_ids)
                
                # Create FASTA file for Phase2
                phase2_fasta = intermediate_dir / "phase2_input.fasta"
                with open(phase2_fasta, 'w') as f:
                    for seq_id, sequence in r_gene_sequences.items():
                        f.write(f">{seq_id}\n{sequence}\n")
                logger.info(f"Created Phase2 input FASTA with {len(r_gene_sequences)} sequences")
                
                # Run Phase2 prediction on R-genes only
                results, message = DNN(phase2_fasta, output_dir, level)
                # Create Phase2 summary
                create_summary_report(results, output_dir, "Phase2", append=True)
        else:
            results = phase1_results
        
        # Save final results in main output directory
        output_file = output_dir / output
        try:
            if not message:
                results.to_csv(output_file, sep="\t", index=False)
                logger.info(f"Final results saved to: {output_file}")
            else:
                logger.warning(f"Prediction completed with message: {message}")
                (intermediate_dir / 'error_message.txt').write_text(message)
                logger.info(f"Message written to: {intermediate_dir / 'error_message.txt'}")
        except Exception as e:
            raise PRGminerError(f"Failed to save results: {str(e)}")

        end = time.time()
        logger.info(f"Total execution time: {end - start:.2f} seconds")
        logger.info("PRGminer completed successfully")
        return 0

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"PRGminer execution failed: {str(e)}")
        else:
            print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())


