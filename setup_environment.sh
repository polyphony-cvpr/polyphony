#!/bin/bash
# Setup script for unified Polyphony environment
# This script creates a conda environment combining videomaev2 and polyphony dependencies

set -e  # Exit on error

echo "=========================================="
echo "Polyphony Unified Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check CUDA version
echo "Checking CUDA version..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "No NVIDIA GPU detected, will install CPU-only PyTorch"
    CUDA_VERSION="cpu"
fi

# Environment name
ENV_NAME="polyphony-unified"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Warning: Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Exiting. Use 'conda activate ${ENV_NAME}' to use existing environment."
        exit 0
    fi
fi

# Step 1: Create environment from environment.yml
echo ""
echo "Step 1: Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Step 2: Activate environment
echo ""
echo "Step 2: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Step 3: Install PyTorch with appropriate CUDA support
echo ""
echo "Step 3: Installing PyTorch with CUDA support..."

if [ "$CUDA_VERSION" = "cpu" ]; then
    echo "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
elif [ "$CUDA_VERSION" = "11.8" ] || [ "$CUDA_VERSION" = "11" ]; then
    echo "Installing PyTorch with CUDA 11.8 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [ "$CUDA_VERSION" = "12.1" ] || [ "$CUDA_VERSION" = "12" ]; then
    echo "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Unknown CUDA version. Installing PyTorch with CUDA 11.8 (default)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Step 4: Install remaining pip packages
echo ""
echo "Step 4: Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 5: Verify installation
echo ""
echo "Step 5: Verifying installation..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "  (CUDA check skipped)"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}')"
python -c "import timm; print(f'✓ timm: {timm.__version__}')" 2>/dev/null || echo "  (timm check skipped)"
python -c "import decord; print('✓ decord: OK')" 2>/dev/null || echo "  (decord check skipped)"

echo ""
echo "=========================================="
echo "✓ Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "For more information, see ENVIRONMENT_SETUP.md"

