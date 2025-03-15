# Basic Usage Examples

This guide provides practical examples of using PRGminer for R-gene prediction and classification.

## 1. Simple Prediction

### Command-line Usage

```bash
# Predict R-genes in a FASTA file
PRGminer -i proteins.fasta -od results -l Phase2
```

### Python API Usage

```python
from PRGminer import PRGPredictor

# Initialize predictor
predictor = PRGPredictor(prediction_level='Phase2')

# Load sequences
sequences = {
    'protein1': 'MAEGEQVQSGEDLGSPVAQVLQKAREQGAQAAVLVVPPG',
    'protein2': 'MTKFTILLFFLSVALASNAQPGCNQSQTLSPNWQNVFGAS'
}

# Make predictions
results = predictor.predict(sequences)

# Print results
for seq_id, predictions in results.items():
    print(f"Sequence: {seq_id}")
    print(f"Predicted class: {predictions['class']}")
    print(f"Probabilities: {predictions['probabilities']}")
```

## 2. Batch Processing

For large datasets, you can process sequences in batches:

```python
from PRGminer import PRGPredictor, load_fasta
import pandas as pd

# Load sequences from FASTA file
sequences = load_fasta('large_dataset.fasta')

# Initialize predictor
predictor = PRGPredictor(prediction_level='Phase2')

# Process in batches of 100
batch_size = 100
all_results = []

for i in range(0, len(sequences), batch_size):
    batch = dict(list(sequences.items())[i:i+batch_size])
    results = predictor.predict(batch)
    all_results.extend(results)

# Save results to CSV
df = pd.DataFrame(all_results)
df.to_csv('predictions.csv', index=False)
```

## 3. Custom Output Formatting

Example of customizing the output format:

```python
from PRGminer import PRGPredictor
import json

def custom_output_formatter(predictions):
    formatted = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0'
        },
        'predictions': predictions
    }
    return formatted

# Make predictions
predictor = PRGPredictor()
results = predictor.predict(sequences)

# Format and save results
formatted_results = custom_output_formatter(results)
with open('custom_results.json', 'w') as f:
    json.dump(formatted_results, f, indent=2)
```

## 4. Error Handling

Example showing proper error handling:

```python
from PRGminer import PRGPredictor, SequenceValidationError

try:
    predictor = PRGPredictor()
    
    # Invalid sequence example
    invalid_sequences = {
        'protein1': 'INVALID123'  # Contains invalid characters
    }
    
    results = predictor.predict(invalid_sequences)

except SequenceValidationError as e:
    print(f"Sequence validation failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
```

## 5. Using Different Prediction Levels

Example showing both Phase 1 and Phase 2 predictions:

```python
from PRGminer import PRGPredictor

# Phase 1 prediction (R-gene vs non-R-gene)
phase1_predictor = PRGPredictor(prediction_level='Phase1')
phase1_results = phase1_predictor.predict(sequences)

# If sequence is predicted as R-gene, perform Phase 2 classification
phase2_predictor = PRGPredictor(prediction_level='Phase2')
for seq_id, result in phase1_results.items():
    if result['class'] == 'Rgene':
        phase2_result = phase2_predictor.predict({seq_id: sequences[seq_id]})
        print(f"Detailed classification for {seq_id}: {phase2_result}")
```

For more advanced examples, please refer to the [Advanced Usage Guide](advanced_usage.md). 