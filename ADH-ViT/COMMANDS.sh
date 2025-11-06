#!/bin/bash
# ADH-ViT Command Reference
# All commands assume you're starting from /home/hao/Polyphony

# ============================================================================
# BASIC TRAINING
# ============================================================================

# 1. Navigate to ADH-ViT
cd /home/hao/Polyphony/ADH-ViT

# 2. Activate environment (replace with your env name)
conda activate your_pytorch_env
# OR: source /path/to/venv/bin/activate

# 3. Run training with default settings (DUAL-HAND)
bash scripts/finetune/train_havid_alternating.sh

# 4. Run training on SINGLE-STREAM data (one hand only)
python main.py \
    --one_stream \
    --lh_data_path /home/hao/Polyphony/data/single_stream/data \
    --lh_num_classes 75 \
    --rh_num_classes 75 \
    --model vit_base_patch16_224_alternating \
    --finetune models/vit_b_k710_dl_from_giant.pth \
    --output_dir output/single_stream \
    --batch_size 4 \
    --epochs 50 \
    --alternation_steps 50

# ============================================================================
# CUSTOM TRAINING
# ============================================================================

# Train with custom parameters
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --model vit_base_patch16_224_alternating \
    --lh_data_path /home/hao/Polyphony/data/havid_mmaction/lh_v0 \
    --lh_data_root /home/hao/Polyphony/data/havid_mmaction/lh_v0 \
    --rh_data_path /home/hao/Polyphony/data/havid_mmaction/rh_v0 \
    --rh_data_root /home/hao/Polyphony/data/havid_mmaction/rh_v0 \
    --lh_num_classes 75 \
    --rh_num_classes 75 \
    --data_set HAVID \
    --finetune models/vit_b_k710_dl_from_giant.pth \
    --output_dir output/my_experiment \
    --log_dir output/my_experiment \
    --batch_size 4 \
    --epochs 50 \
    --alternation_steps 50 \
    --lr 1e-3 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_workers 8

# ============================================================================
# MULTI-GPU TRAINING
# ============================================================================

# Train on 2 GPUs
cd /home/hao/Polyphony/ADH-ViT
torchrun --nproc_per_node=2 \
    run_alternating_hand_finetuning.py \
    [... same args as above ...]

# Train on 4 GPUs
cd /home/hao/Polyphony/ADH-ViT
torchrun --nproc_per_node=4 \
    run_alternating_hand_finetuning.py \
    [... same args as above ...]

# ============================================================================
# MONITORING
# ============================================================================

# Monitor GPU usage
watch -n 1 nvidia-smi

# View TensorBoard logs
cd /home/hao/Polyphony/ADH-ViT
tensorboard --logdir output/havid_alternating_hands --port 6006

# Check training log
cd /home/hao/Polyphony/ADH-ViT
tail -f output/havid_alternating_hands/log.txt

# ============================================================================
# RESUME TRAINING
# ============================================================================

# Resume from last checkpoint
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --resume output/havid_alternating_hands/checkpoint-latest.pth \
    [... other args ...]

# Resume from specific epoch
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --resume output/havid_alternating_hands/checkpoint-30.pth \
    --start_epoch 30 \
    [... other args ...]

# ============================================================================
# EVALUATION
# ============================================================================

# Evaluate on validation set
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --eval \
    --resume output/havid_alternating_hands/checkpoint-best.pth \
    --lh_data_path /home/hao/Polyphony/data/havid_mmaction/lh_v0 \
    --rh_data_path /home/hao/Polyphony/data/havid_mmaction/rh_v0 \
    [... other args ...]

# Evaluate on test set
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --eval \
    --resume output/havid_alternating_hands/checkpoint-best.pth \
    --lh_data_path /path/to/test/lh_data \
    --rh_data_path /path/to/test/rh_data \
    [... other args ...]

# ============================================================================
# EXPERIMENTS WITH DIFFERENT ALTERNATION STRATEGIES
# ============================================================================

# Very frequent switching (every 10 steps)
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --alternation_steps 10 \
    --output_dir output/alternation_10 \
    [... other args ...]

# Moderate switching (every 50 steps) - DEFAULT
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --alternation_steps 50 \
    --output_dir output/alternation_50 \
    [... other args ...]

# Infrequent switching (every 200 steps)
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --alternation_steps 200 \
    --output_dir output/alternation_200 \
    [... other args ...]

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

# Experiment 1: Higher learning rate
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --lr 2e-3 \
    --output_dir output/lr_2e3 \
    [... other args ...]

# Experiment 2: Lower learning rate
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --lr 5e-4 \
    --output_dir output/lr_5e4 \
    [... other args ...]

# Experiment 3: Larger batch size (if GPU memory allows)
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --batch_size 8 \
    --output_dir output/bs_8 \
    [... other args ...]

# Experiment 4: More frames per clip
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --num_frames 32 \
    --output_dir output/frames_32 \
    [... other args ...]

# Experiment 5: Different drop path rate
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py \
    --drop_path 0.5 \
    --output_dir output/droppath_0.5 \
    [... other args ...]

# ============================================================================
# TMUX/SCREEN (For Long Training Sessions)
# ============================================================================

# Start training in tmux
tmux new -s adhvit_training
cd /home/hao/Polyphony/ADH-ViT
conda activate your_env
bash scripts/finetune/train_havid_alternating.sh
# Press Ctrl+B, then D to detach

# Reattach to tmux session
tmux attach -t adhvit_training

# List tmux sessions
tmux ls

# Kill tmux session
tmux kill-session -t adhvit_training

# Start training in screen
screen -S adhvit_training
cd /home/hao/Polyphony/ADH-ViT
conda activate your_env
bash scripts/finetune/train_havid_alternating.sh
# Press Ctrl+A, then D to detach

# Reattach to screen session
screen -r adhvit_training

# ============================================================================
# USEFUL CHECKS
# ============================================================================

# Check if pretrained model exists
ls -lh /home/hao/Polyphony/ADH-ViT/models/vit_b_k710_dl_from_giant.pth

# Check dataset structure
ls /home/hao/Polyphony/data/havid_mmaction/lh_v0/
ls /home/hao/Polyphony/data/havid_mmaction/rh_v0/

# Count training videos
wc -l /home/hao/Polyphony/data/havid_mmaction/lh_v0/train_list_video.txt
wc -l /home/hao/Polyphony/data/havid_mmaction/rh_v0/train_list_video.txt

# Count validation videos
wc -l /home/hao/Polyphony/data/havid_mmaction/lh_v0/val_list_video.txt
wc -l /home/hao/Polyphony/data/havid_mmaction/rh_v0/val_list_video.txt

# Check GPU availability
nvidia-smi

# Check CUDA version
nvcc --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Check disk space
df -h /home/hao/Polyphony/ADH-ViT

# ============================================================================
# CLEANUP
# ============================================================================

# Remove old checkpoints (keep every 20 epochs)
cd /home/hao/Polyphony/ADH-ViT/output/havid_alternating_hands
rm checkpoint-{1..9}.pth checkpoint-1{1..9}.pth checkpoint-2{1..9}.pth checkpoint-3{1..9}.pth

# Remove TensorBoard logs
cd /home/hao/Polyphony/ADH-ViT/output/havid_alternating_hands
rm -rf events.out.tfevents.*

# ============================================================================
# INSTALLATION (First Time Only)
# ============================================================================

# Install dependencies
cd /home/hao/Polyphony/ADH-ViT
pip install -r requirements.txt

# Install specific versions (if needed)
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install timm==0.6.12
pip install decord
pip install einops
pip install tensorboard

# Download pretrained model (if missing)
cd /home/hao/Polyphony/ADH-ViT/models
wget https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/videomaev2/vit_b_k710_dl_from_giant.pth

# ============================================================================
# HELP
# ============================================================================

# Show all available arguments
cd /home/hao/Polyphony/ADH-ViT
python run_alternating_hand_finetuning.py --help

# ============================================================================
# Quick Copy-Paste Commands
# ============================================================================

# BASIC TRAINING:
cd /home/hao/Polyphony/ADH-ViT && conda activate your_env && bash scripts/finetune/train_havid_alternating.sh

# WITH TMUX:
tmux new -s train && cd /home/hao/Polyphony/ADH-ViT && conda activate your_env && bash scripts/finetune/train_havid_alternating.sh

# MULTI-GPU (2 GPUs):
cd /home/hao/Polyphony/ADH-ViT && conda activate your_env && sed -i 's/nproc_per_node=1/nproc_per_node=2/g' scripts/finetune/train_havid_alternating.sh && bash scripts/finetune/train_havid_alternating.sh

# ============================================================================

