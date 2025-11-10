# Unified Environment Setup Guide

This guide explains how to set up a unified conda environment that combines dependencies from both `videomaev2` and `polyphony` environments.

## Overview

The unified environment includes:
- **PyTorch 2.0+** (from polyphony, newer and more compatible)
- **Python 3.10** (compatible with both old and new packages)
- All dependencies from both environments
- Resolved version conflicts

## Quick Start

### Option 1: Using Conda Environment (Recommended)

```bash
# Create the unified environment
conda env create -f environment.yml

# Activate the environment
conda activate polyphony-unified

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Option 2: Using Requirements.txt

```bash
# Create a new conda environment
conda create -n polyphony-unified python=3.10
conda activate polyphony-unified

# Install PyTorch (choose based on your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision torchaudio

# Install remaining dependencies
pip install -r requirements.txt
```

## CUDA Setup

### Check Your CUDA Version

```bash
nvidia-smi
```

### Install PyTorch with CUDA Support

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

## Key Dependencies

### Core Framework
- **PyTorch 2.0+**: Deep learning framework
- **torchvision**: Computer vision utilities
- **torchaudio**: Audio processing

### Semantic Models
- **transformers**: HuggingFace transformers library
- **sentence-transformers**: Semantic embedding models
- **tokenizers**: Fast tokenization

### Video Processing
- **opencv-python**: Computer vision and video processing
- **decord**: Efficient video decoding
- **av**: Audio/video processing

### Computer Vision
- **timm**: PyTorch image models

### Data Processing
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities
- **h5py**: HDF5 file support

### Visualization
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization

### Training Tools
- **tensorboard**: Training visualization
- **tensorboardX**: Extended TensorBoard support

## Verification

After installation, verify key packages:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
python -c "import decord; print('decord: OK')"
```

## Migration from Old Environments

### From videomaev2

The unified environment uses:
- **PyTorch 2.0+** instead of 1.12.1 (backward compatible for most use cases)
- **Python 3.10** instead of 3.8 (better compatibility)

Most code should work without changes. If you encounter issues:

1. **PyTorch API changes**: Check [PyTorch migration guide](https://pytorch.org/docs/stable/migration.html)
2. **Python 3.10**: Most Python 3.8 code works in 3.10

### From polyphony

The unified environment maintains:
- **PyTorch 2.0+** (same as polyphony)
- **Python 3.10** (compatible with 3.12 code)
- All existing packages

## Troubleshooting

### Issue: CUDA not available

**Solution**: Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Package version conflicts

**Solution**: The environment.yml uses compatible versions. If conflicts occur:
```bash
conda env update -f environment.yml --prune
```

### Issue: Missing video codecs

**Solution**: Install ffmpeg via conda:
```bash
conda install -c conda-forge ffmpeg
```

### Issue: decord installation fails

**Solution**: Install from conda-forge:
```bash
conda install -c conda-forge decord
```

## Environment Comparison

| Package | videomaev2 | polyphony | Unified |
|---------|------------|-----------|---------|
| Python | 3.8.20 | 3.12.9 | 3.10.14 |
| PyTorch | 1.12.1 | 2.2.0 | 2.0+ |
| CUDA | 11.3 | 12.1 | 11.8/12.1 |
| transformers | 4.46.3 | - | 4.30+ |
| timm | 0.4.12 | - | 0.6+ |

## Updating the Environment

To update packages:

```bash
conda activate polyphony-unified
conda env update -f environment.yml --prune
```

Or using pip:

```bash
pip install --upgrade -r requirements.txt
```

## Removing the Environment

If you need to remove the unified environment:

```bash
conda deactivate
conda env remove -n polyphony-unified
```

## Notes

- The unified environment prioritizes newer versions for better compatibility
- PyTorch 2.0+ is backward compatible with most PyTorch 1.12 code
- Python 3.10 provides good compatibility with both old and new packages
- CUDA version should match your system's CUDA installation

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify CUDA compatibility with your GPU
3. Ensure all system dependencies are installed

