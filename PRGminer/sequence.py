"""
Sequence processing module for PRGminer
Author: Naveen Duhan (naveen.duhan@usu.edu)
"""

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from typing import Dict, List, Union, Optional

class SequenceProcessor:
    """Class for processing protein sequences for R-gene prediction."""
    
    def __init__(self, max_length: int = 2000):
        """Initialize sequence processor.
        
        Args:
            max_length (int): Maximum sequence length for padding/truncation
        """
        self.max_length = max_length
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.aa_to_int = {aa: i for i, aa in enumerate(self.amino_acids)}
    
    def validate_sequence(self, sequence: str) -> bool:
        """Validate protein sequence.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            bool: True if sequence is valid
            
        Raises:
            ValueError: If sequence is invalid
        """
        if not sequence:
            raise ValueError("Empty sequence")
            
        valid_chars = set(self.amino_acids)
        sequence_chars = set(sequence.upper())
        invalid_chars = sequence_chars - valid_chars
        
        if invalid_chars:
            raise ValueError(f"Invalid characters in sequence: {invalid_chars}")
            
        return True
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode protein sequence as numeric array.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            np.ndarray: Encoded sequence
        """
        # Convert to uppercase and pad/truncate
        sequence = sequence.upper()[:self.max_length]
        sequence = sequence.ljust(self.max_length, 'X')
        
        # One-hot encode
        encoding = np.zeros((1, self.max_length, len(self.amino_acids)), dtype='float32')
        for i, aa in enumerate(sequence):
            if aa in self.aa_to_int:
                encoding[0, i, self.aa_to_int[aa]] = 1.0
                
        return encoding
    
    def load_fasta(self, fasta_path: str) -> Dict[str, str]:
        """Load sequences from FASTA file.
        
        Args:
            fasta_path (str): Path to FASTA file
            
        Returns:
            Dict[str, str]: Dictionary of sequence IDs and sequences
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        try:
            sequences = {}
            for record in SeqIO.parse(fasta_path, "fasta"):
                sequences[record.id] = str(record.seq)
                
            if not sequences:
                raise ValueError(f"No valid sequences found in {fasta_path}")
                
            return sequences
            
        except (FileNotFoundError, ValueError) as e:
            raise
        except Exception as e:
            raise ValueError(f"Error reading FASTA file: {e}")
    
    def preprocess(self, sequence: str) -> np.ndarray:
        """Preprocess a single sequence.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            np.ndarray: Preprocessed sequence
        """
        self.validate_sequence(sequence)
        return self.encode_sequence(sequence)
    
    def process_batch(self, sequences: Dict[str, str]) -> List[np.ndarray]:
        """Process multiple sequences.
        
        Args:
            sequences (Dict[str, str]): Dictionary of sequence IDs and sequences
            
        Returns:
            List[np.ndarray]: List of processed sequences
        """
        if not sequences:
            raise ValueError("Empty sequence batch")
            
        processed = []
        for seq_id, sequence in sequences.items():
            processed.append(self.preprocess(sequence))
            
        return processed
    
    def format_output(self, predictions: Dict) -> Dict:
        """Format model predictions.
        
        Args:
            predictions (Dict): Raw prediction output
            
        Returns:
            Dict: Formatted predictions
            
        Raises:
            ValueError: If predictions are invalid
        """
        if not predictions:
            raise ValueError("Empty predictions")
            
        formatted = {}
        for seq_id, pred in predictions.items():
            if not isinstance(pred, dict) or 'class' not in pred or 'probabilities' not in pred:
                raise ValueError(f"Invalid prediction format for sequence {seq_id}")
            formatted[seq_id] = {
                'class': pred['class'],
                'probabilities': pred['probabilities']
            }
            
        return formatted 