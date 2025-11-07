# Semantic Feature Alignment for Video Action Recognition

This module aligns visual features extracted from videos with semantic embeddings from language models using Temporal Convolutional Networks (TCN). The enhanced features can be used for downstream action segmentation tasks.

## Overview

The semantic conditioning approach:
1. Takes visual features from a video encoder (e.g., VideoMAE)
2. Processes them through a Temporal Convolutional Network (TCN)
3. Aligns them with semantic embeddings from text descriptions
4. Outputs enhanced features for action segmentation

## Key Features

✅ **Dynamic Batching** - Pads sequences to batch max length (not fixed), reducing memory usage  
✅ **Temporal Modeling** - TCN with dilated convolutions captures temporal context  
✅ **Semantic Alignment** - Aligns visual and semantic features in shared embedding space  
✅ **Flexible Loss Functions** - Supports adaptive, MSE, cosine, and smooth L1 losses  
✅ **Precomputed Embeddings** - Uses cached semantic embeddings for efficiency  
✅ **Comprehensive Evaluation** - Includes visualizations and metrics  

## Directory Structure

```
semantic_conditioning/
├── main.py              # Main training script
├── config.py            # Configuration file
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── checkpoints/         # Saved models (created during training)
└── enhanced_features/   # Enhanced visual features (created during training)
```

## Installation

### 1. Create a Conda Environment

```bash
conda create -n semantic python=3.9
conda activate semantic
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install numpy matplotlib seaborn scikit-learn
```

## Usage

### Quick Start

Simply run the main script with default configuration:

```bash
python main.py
```

### Custom Configuration

Edit `config.py` to customize:
- Data paths
- Model architecture
- Training hyperparameters
- Loss function type
- Semantic embedding model

Example modifications:

```python
# In config.py
TCN_HIDDEN_DIMS = [640, 512, 384]  # Larger model
BATCH_SIZE = 8  # Larger batch
LEARNING_RATE = 5e-4  # Different learning rate
LOSS_TYPE = 'smooth_l1'  # Different loss
```

### Training

The training process includes:
1. **Data Loading** - Loads visual features and frame annotations
2. **Model Initialization** - Creates TCN-based alignment model
3. **Training Loop** - Trains with early stopping and checkpointing
4. **Evaluation** - Computes alignment metrics
5. **Feature Saving** - Saves enhanced features for downstream tasks

Training output:
```
Epoch 1/150
Batch 0/100, Loss: 0.234567
...
Train Loss: 0.234567, Val Loss: 0.345678
Current LR: 3.00e-04
New best model saved with val_loss: 0.345678
```

### Output Files

After training, you'll have:

1. **Checkpoints** (`./checkpoints/`)
   - `best_model.pth` - Best model based on validation loss
   - `checkpoint_epoch_20.pth`, `checkpoint_epoch_40.pth`, etc.

2. **Enhanced Features** (`./enhanced_features/`)
   - `<video_id>.npy` - Enhanced features for each video (transposed format)

## Model Architecture

```
Visual Features [B, T, 768]
        ↓
Temporal Convolutional Network (TCN)
  - Layer 1: 768 → 512 (dilation=1)
  - Layer 2: 512 → 128 (dilation=2)
  - Layer 3: 128 → 64 (dilation=4)
        ↓
Visual Projector
  - Linear: 64 → 512
  - LayerNorm + ReLU + Dropout
  - Linear: 512 → 384
  - LayerNorm + Dropout
        ↓
Aligned Features [B, T, 384]
```

## Loss Functions

### Adaptive Loss (Default)
Combines cosine similarity and MSE:
```python
loss = α * (1 - cosine_sim) + (1 - α) * mse
```
- **α = 0.7**: Emphasizes directional alignment
- **(1 - α) = 0.3**: Maintains feature magnitude

### Other Options
- **MSE**: Mean squared error
- **Cosine**: Cosine similarity loss
- **Smooth L1**: Huber loss variant

## Evaluation Metrics

The evaluator computes:
- **MSE Loss** - Mean squared error between aligned and target features
- **Mean Cosine Similarity** - Average cosine similarity
- **Median/Std Cosine Similarity** - Distribution statistics
- **Per-Action-Type Performance** - Separate metrics for:
  - Regular actions
  - Null/transition states
  - Wrong actions

Example output:
```
Alignment Quality Results:
  mse_loss: 0.123456
  mean_cosine_similarity: 0.876543
  median_cosine_similarity: 0.890123
  actions_mean_similarity: 0.890234
  null_mean_similarity: 0.856789
  wrong_mean_similarity: 0.834567
```

## Data Format

### Input Requirements

1. **Visual Features** (`.npy` files)
   - Shape: `(feature_dim, seq_len)` - will be transposed to `(seq_len, feature_dim)`
   - Typical dimension: 768 (ViT-base backbone)

2. **Frame Annotations** (`.txt` files)
   - One action label per line
   - Example:
     ```
     action_1
     action_1
     action_2
     null
     action_3
     ```

3. **Action Mapping** (`havid_description.txt`)
   - Format: `<label> "<description>"`
   - Example:
     ```
     action_1 "pick up the cup"
     action_2 "pour water"
     null "no action or transition"
     ```

4. **Split Files** (e.g., `train.split1.bundle`)
   - One video ID per line
   - Example:
     ```
     video_001
     video_002
     video_003
     ```

5. **Semantic Embeddings** (`.pt` file, optional but recommended)
   - Precomputed embeddings for all action labels
   - Format: `{'embeddings': {label: tensor, ...}}`

### Output Format

Enhanced features are saved as `.npy` files:
- Shape: `(feature_dim, seq_len)` - transposed for downstream compatibility
- Typical dimension: 384 (MiniLM semantic dim)

## Advanced Usage

### Using Different Semantic Models

To use a different semantic model (e.g., MPNet):

1. Update `config.py`:
```python
SEMANTIC_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
SEMANTIC_DIM = 768  # MPNet dimension
SEMANTIC_EMBEDDINGS_PATH = None  # Or path to precomputed embeddings
```

2. If using precomputed embeddings, generate them first:
```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = 'sentence-transformers/all-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Generate embeddings for your action descriptions
# ... (similar to the semantic embedding generation script)
```

### Training on Different Datasets

To adapt for a different dataset:

1. Update paths in `config.py`
2. Ensure your action mapping file follows the format
3. Verify visual feature dimensions match `VISUAL_DIM`

### Resuming Training

To resume from a checkpoint:

```python
# In main.py, after model initialization:
checkpoint = torch.load('./checkpoints/checkpoint_epoch_40.pth')
model.load_state_dict(checkpoint['model_state_dict'])
trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## Integration with Downstream Tasks

The enhanced features can be directly used with action segmentation models:

```python
# In your action segmentation script
enhanced_features = np.load('enhanced_features/video_001.npy')
# Shape: (384, seq_len) - ready for segmentation model input
```

## Contact

For questions or issues, please open an issue in the repository.

