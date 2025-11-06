# VideoMAEv2 - Clean Alternating Dual-Hand Training

This is a **clean, minimal version** of the VideoMAEv2 codebase containing only the essential files needed for **alternating dual-hand video action recognition training** on the HA-ViD dataset.

## 📁 Directory Structure

```
VideoMAEv2_clean_alternating/
├── run_alternating_hand_finetuning.py    # Main training script
├── engine_for_alternating_finetuning.py  # Training engine
├── engine_for_finetuning.py              # Helper functions (merge, etc.)
├── utils.py                               # Utility functions
├── optim_factory.py                       # Optimizer factory
├── models/
│   ├── __init__.py
│   ├── modeling_finetune_alternating.py  # Alternating dual-head ViT model
│   └── vit_b_k710_dl_from_giant.pth     # Pretrained weights
├── dataset/
│   ├── __init__.py
│   ├── build.py                          # Dataset builder
│   ├── datasets.py                       # Dataset classes
│   ├── functional.py                     # Data functions
│   ├── loader.py                         # Data loader
│   ├── masking_generator.py             # Masking utilities
│   ├── rand_augment.py                   # Random augmentation
│   ├── random_erasing.py                 # Random erasing
│   ├── transforms.py                     # Transform utilities
│   ├── video_transforms.py               # Video-specific transforms
│   └── volume_transforms.py              # Volume transforms
├── scripts/
│   └── finetune/
│       └── train_havid_alternating.sh   # Training shell script
├── requirements.txt                      # Python dependencies
├── LICENSE                               # License file
└── README.md                            # Original README

```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd VideoMAEv2_clean_alternating
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure your HA-ViD dataset is organized as:
```
data/havid_mmaction/
├── lh_v0/
│   ├── train_list_video.txt
│   ├── val_list_video.txt
│   └── videos/
└── rh_v0/
    ├── train_list_video.txt
    ├── val_list_video.txt
    └── videos/
```

### 3. Run Training

```bash
bash scripts/finetune/train_havid_alternating.sh
```

Or run directly:

```bash
python run_alternating_hand_finetuning.py \
    --model vit_base_patch16_224_alternating \
    --lh_data_path /path/to/havid_mmaction/lh_v0 \
    --lh_data_root /path/to/havid_mmaction/lh_v0 \
    --rh_data_path /path/to/havid_mmaction/rh_v0 \
    --rh_data_root /path/to/havid_mmaction/rh_v0 \
    --lh_num_classes 75 \
    --rh_num_classes 75 \
    --data_set HAVID \
    --finetune models/vit_b_k710_dl_from_giant.pth \
    --output_dir output/havid_alternating_hands \
    --batch_size 4 \
    --epochs 50 \
    --alternation_steps 50
```

## 🎯 Key Parameters

- `--alternation_steps`: Number of training steps before switching between left/right hand (default: 50)
- `--lh_num_classes` / `--rh_num_classes`: Number of action classes for each hand
- `--batch_size`: Batch size per GPU
- `--epochs`: Total training epochs
- `--lr`: Learning rate (default: 1e-3)
- `--drop_path`: DropPath rate (default: 0.3)

## 📊 Model Architecture

The model uses a **Vision Transformer (ViT) with dual classification heads**:
- **Shared backbone**: Extracts visual features from video frames
- **Left-hand head**: Classifies left-hand actions
- **Right-hand head**: Classifies right-hand actions
- **Alternating training**: Switches between hands every N steps

## 🔧 What Was Removed

This clean version removes:
- ❌ Semantic feature alignment scripts (TCN-based)
- ❌ Dual-hand semantic integration
- ❌ Language conditioning modules
- ❌ Feature extraction scripts
- ❌ Evaluation and visualization scripts
- ❌ Multiple training strategy variants (v2, v3, v4)
- ❌ One-stream training variants
- ❌ Pretraining scripts
- ❌ Assembly101 and Breakfast dataset scripts
- ❌ Documentation files (except this README)
- ❌ Log files and checkpoints
- ❌ Experimental and debug scripts

**Total: ~70+ files removed, keeping only 23 essential files**

## 📝 Files Breakdown

### Core Training (3 files)
- `run_alternating_hand_finetuning.py` - Main entry point
- `engine_for_alternating_finetuning.py` - Training/validation loops
- `engine_for_finetuning.py` - Helper functions

### Model (2 files)
- `models/__init__.py` - Model registry
- `models/modeling_finetune_alternating.py` - Alternating dual-head ViT

### Dataset (11 files)
- All files in `dataset/` directory - Data loading and augmentation

### Utils (2 files)
- `utils.py` - General utilities (distributed training, logging, etc.)
- `optim_factory.py` - Optimizer creation and layer-wise learning rate decay

### Config (3 files)
- `README.md` - Original project README
- `requirements.txt` - Python dependencies
- `LICENSE` - License information

### Scripts (1 file)
- `scripts/finetune/train_havid_alternating.sh` - Example training script

### Pretrained (1 file)
- `models/vit_b_k710_dl_from_giant.pth` - Pretrained ViT weights

## 🔗 Dependencies

Key dependencies (see `requirements.txt` for full list):
- PyTorch >= 1.8.0
- torchvision
- timm
- decord (for video loading)
- einops

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

This is a cleaned version of the VideoMAEv2 project, focusing only on alternating dual-hand training functionality for the HA-ViD dataset.

