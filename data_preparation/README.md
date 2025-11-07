# Data Preparation Tools

Utilities for preparing datasets for dual-hand action segmentation.

## Overview

This folder contains scripts for preprocessing datasets, extracting features, and generating semantic embeddings for action labels.

## Dataset Preparation Guide

This section explains how to download and prepare action segmentation datasets from their original sources.

### Download datasets

We benchmark on three datasets:

**HA-VID (a human assembly video dataset)** a human assembly video dataset with dual-hand action annotations. Download HA-ViD from the official website: [HAVID Dataset](https://iai-hrc.github.io/ha-vid).

**ATTACH** annotated two-handed assembly actions for human action understanding. Download ATTACH from the official website: [ATTACH Dataset](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/attach-dataset).

**Breakfast** a single-stream action segmentation dataset for cooking activities. Download Breakfast dataset from: [ms-tcn](https://github.com/yabufarha/ms-tcn).

### Directory Structure

After downloading, organize your data as follows:

```
data/dataset/
├── videos                  # Original video files (.mp4)
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── features/                    # Original I3D features (.npy)
│   ├── feature1.mp4
│   ├── feature2.mp4
│   └── ...
├── groundTruth/               # Frame-wise annotations (.txt)
│   ├── video1.txt
│   ├── video2.txt
│   └── ...
├── mapping.txt                # Label to ID mapping
│   # Format: "0 label1" or "label1 0"
├── splits/                    # Train/test splits
│   ├── train.split1.bundle
│   ├── test.split1.bundle
│   └── ...
└── havid_description.txt      # Action descriptions for semantic embeddings (stored in folder ./action_descriptions)
```

### Next Steps

After preparing your dataset:
1. **Generate action recognition dataset** (see [Script 1](#1-generate-action-recognition-dataset))
2. **Precompute semantic embeddings** (see [Script 2](#2-precompute-semantic-embeddings)) 

---

## Scripts

### 1. Generate Action Recognition Dataset

Extract video clips from action segmentation datasets and prepare them for training ADH-ViT.

**Script**: `prepare_dataset_for_ADH-ViT.py`

#### Features

- ✅ **Two Extraction Methods**:
  - **Segment-based**: Extracts entire action segments (consecutive frames with same label)
  - **Clip-based**: Samples random fixed-length clips with class balancing
- ✅ **Flexible Configuration**: Supports various video formats and annotation styles
- ✅ **Organized Output**: Clips organized by label folders with index files

#### Usage

**Segment-based extraction** (extract entire action segments):

```bash
python prepare_dataset_for_ADH-ViT.py \
    --video_dir /path/to/videos \
    --annotation_dir /path/to/annotations \
    --output_dir /path/to/output \
    --mapping_file /path/to/mapping.txt \
    --extraction_method segment \
    --split_list /path/to/train.split1.bundle
```

**Clip-based extraction** (sample fixed-length clips):

```bash
python prepare_dataset_for_ADH-ViT.py \
    --video_dir /path/to/videos \
    --annotation_dir /path/to/annotations \
    --output_dir /path/to/output \
    --mapping_file /path/to/mapping.txt \
    --extraction_method random \
    --clips_per_video 10 \
    --clip_length 16 \
    --prefer_non_null \
    --split_list /path/to/train.split1.bundle
```

#### Extraction Methods

**1. Segment-Based (`--extraction_method segment`)**

- Extracts entire action segments (consecutive frames with same label)
- Each segment becomes a separate video clip
- Preserves complete action boundaries
- Best for: Datasets with well-defined action segments

**Parameters:**
- `--min_segment_length`: Minimum segment length to extract (default: 1 frame)

**2. Random Clip-Based (`--extraction_method random`)**

- Samples random fixed-length clips from videos
- Balances class distribution automatically
- Prefers non-null labels (optional)
- Best for: Creating balanced training sets

**Parameters:**
- `--clips_per_video`: Number of clips per video (default: 10)
- `--clip_length`: Length of each clip in frames (default: 16)
- `--prefer_non_null`: Prefer clips with non-null labels (default: True)

#### Input Format

**Video Directory**: Contains `.mp4` video files

**Annotation Directory**: Contains `.txt` annotation files (one per video)

**Annotation Format** (frame-wise labels):
```
label1
label1
label2
label2
label2
...
```

Or space-separated:
```
label1 label1 label2 label2 label2 ...
```

**Mapping File** (`mapping.txt`):
```
0 label1
1 label2
2 label3
```

Or:
```
label1 0
label2 1
label3 2
```

**Split List** (bundle file):
```
video1.txt
video2.txt
video3.txt
```

#### Output Format

**Directory Structure:**
```
output_dir/
├── label1/
│   ├── video1_label1_0.mp4
│   ├── video1_label1_1.mp4
│   └── ...
├── label2/
│   ├── video2_label2_0.mp4
│   └── ...
└── ...
```

**Index File** (`train_list_video.txt`):
```
label1/video1_label1_0.mp4 0
label1/video1_label1_1.mp4 0
label2/video2_label2_0.mp4 1
...
```

Format: `{relative_path} {numeric_label_id}`

#### Examples

**Example 1: Assembly101 Dataset (Segment-based)**

```bash
python prepare_dataset_for_ADH-ViT.py \
    --video_dir /data/Assembly101/videos \
    --annotation_dir /data/Assembly101/groundTruth \
    --output_dir /data/Assembly101/action_recognition/videos_train \
    --mapping_file /data/Assembly101/mapping.txt \
    --extraction_method segment \
    --min_segment_length 5 \
    --split_list /data/Assembly101/splits/train.split1.bundle
```

**Example 2: HAVID Dataset (Clip-based)**

```bash
python prepare_dataset_for_ADH-ViT.py \
    --video_dir /data/havid/videos \
    --annotation_dir /data/havid/groundTruth \
    --output_dir /data/havid/action_recognition/videos_train \
    --mapping_file /data/havid/mapping.txt \
    --extraction_method random \
    --clips_per_video 15 \
    --clip_length 32 \
    --prefer_non_null \
    --split_list /data/havid/splits/train.split1.bundle
```

#### Statistics Output

After processing, the script prints:
- Total number of clips extracted
- Number of action classes
- Clips per class distribution
- Output directory and index file paths

#### Tips

1. **Segment-based** is better for:
   - Preserving complete action boundaries
   - Datasets with well-defined segments
   - When you need full action context

2. **Random clip-based** is better for:
   - Creating balanced training sets
   - Data augmentation
   - When you need fixed-length clips

3. **Class Balancing**: Random extraction automatically balances classes by favoring underrepresented labels

4. **Null Labels**: Use `--prefer_non_null` to avoid extracting clips with null/background labels

---

### 2. Precompute Semantic Embeddings

Generate semantic embeddings for action labels using transformer models.

**Script**: `precompute_semantic_embeddings.py`

#### Features

- ✅ Parses natural language action descriptions
- ✅ Converts to structured format (verb, object, target, tool)
- ✅ Generates embeddings using HuggingFace transformers
- ✅ Supports multiple semantic models
- ✅ Handles special labels ('null', 'w')

#### Usage

**Basic usage:**

```bash
python precompute_semantic_embeddings.py \
    --data_root /path/to/your/dataset \
    --mapping_file action_mapping.txt
```

**With specific model:**

```bash
python precompute_semantic_embeddings.py \
    --data_root /path/to/your/dataset \
    --mapping_file action_mapping.txt \
    --semantic_model_name sentence-transformers/all-MiniLM-L6-v2
```

**Custom output path:**

```bash
python precompute_semantic_embeddings.py \
    --data_root /path/to/your/dataset \
    --mapping_file action_mapping.txt \
    --output_path /path/to/output/embeddings.pt
```

#### Supported Models

| Model | Dimension | Speed | Quality |
|-------|-----------|-------|---------|
| `sentence-transformers/all-mpnet-base-v2` (default) | 768 | Medium | High |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast | Good |
| `BAAI/bge-large-en-v1.5` | 1024 | Slow | Highest |

#### Input Format

**Mapping File** (`action_mapping.txt`):
```
label1 "natural language description 1"
label2 "natural language description 2"
...
```

**Example** (HAVID dataset):
```
pour_liquid_into_cup "Pour the liquid into cup"
cut_bread_using_knife "Cut bread using knife"
```

#### Output Format

**PyTorch .pt file** containing:
```python
{
    'embeddings': {
        'label1': torch.Tensor([...]),  # Shape: [embedding_dim]
        'label2': torch.Tensor([...]),
        ...
    },
    'meta': {
        'semantic_model_name': 'model_name',
        'num_labels': N,
        'embedding_dim': D,
        'data_root': '/path/to/data',
        'mapping_file': '/path/to/mapping.txt'
    }
}
```

#### Structured Description Format

The script converts natural language descriptions to structured format:

**Input**: `"Pour liquid into cup using spoon"`

**Output**: 
```
The action is to pour. 
The manipulated object is liquid. 
The target object is cup. 
The tool used is spoon.
```

This structured format improves semantic embedding quality by explicitly identifying action components.

## Requirements

Install dependencies:

```bash
pip install torch transformers
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Examples

### Example 1: HAVID Dataset

```bash
python precompute_semantic_embeddings.py \
    --data_root /data/havid \
    --mapping_file havid_description.txt \
    --semantic_model_name sentence-transformers/all-mpnet-base-v2
```

**Output**: `/data/havid/semantic_embeddings/sentence-transformers_all-mpnet-base-v2.pt`

### Example 2: Breakfast Dataset

```bash
python precompute_semantic_embeddings.py \
    --data_root /data/breakfast \
    --mapping_file mapping.txt \
    --semantic_model_name sentence-transformers/all-MiniLM-L6-v2
```

**Output**: `/data/breakfast/semantic_embeddings/sentence-transformers_all-MiniLM-L6-v2.pt`

### Example 3: Custom Output Location

```bash
python precompute_semantic_embeddings.py \
    --data_root /data/my_dataset \
    --mapping_file actions.txt \
    --output_path /custom/path/my_embeddings.pt
```

## Loading Embeddings

To use the precomputed embeddings in your code:

```python
import torch

# Load embeddings
embeddings_data = torch.load('path/to/embeddings.pt')

# Access embeddings
label_embeddings = embeddings_data['embeddings']
meta_info = embeddings_data['meta']

# Get embedding for a specific label
label = 'pour_liquid_into_cup'
embedding = label_embeddings[label]  # torch.Tensor of shape [D]

print(f"Embedding dimension: {meta_info['embedding_dim']}")
print(f"Number of labels: {meta_info['num_labels']}")
```

## Special Labels

The script automatically handles special labels:

- **`null`**: Transitional state with no active manipulation
- **`w`**: Wrong or incorrect action

These labels get predefined structured descriptions even if not in the mapping file.

## Troubleshooting

### Issue: Mapping file not found

**Solution**: Ensure the mapping file exists and the path is correct:
```bash
ls /path/to/your/dataset/action_mapping.txt
```

### Issue: Model download fails

**Solution**: Check internet connection or use a local model cache:
```bash
export TRANSFORMERS_CACHE=/path/to/cache
```

### Issue: Out of memory

**Solution**: Use a smaller model:
```bash
--semantic_model_name sentence-transformers/all-MiniLM-L6-v2
```

## Best Practices

1. **Model Selection**:
   - Use `all-mpnet-base-v2` for best quality (768-dim)
   - Use `all-MiniLM-L6-v2` for faster processing (384-dim)
   - Use `bge-large-en-v1.5` for highest quality (1024-dim, slower)

2. **Batch Processing**:
   - Process multiple datasets with a shell script:
   ```bash
   for dataset in havid breakfast 50salads; do
       python precompute_semantic_embeddings.py \
           --data_root /data/$dataset \
           --mapping_file mapping.txt
   done
   ```

3. **Version Control**:
   - Keep track of which model was used
   - Save embeddings with descriptive filenames
   - Document embedding dimensions in your config

## Citation

If you use these tools in your research, please cite:

```bibtex
@inproceedings{your_paper,
    title={Dual-Hand Action Segmentation with Semantic Conditioning},
    author={Your Name},
    booktitle={Conference},
    year={2024}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

