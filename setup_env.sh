#!/bin/bash

# Nerfstudio Environment
echo "Creating Environment 1: nerfstudio..."
conda create --name nerfstudio -y python=3.10
source activate nerfstudio
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install nerfstudio

# Brain Environment (PyTorch 2.4 / CUDA 12.1)
echo "Creating Environment 2: brain..."
conda create --name brain -y python=3.10
source activate brain
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/transformers
pip install accelerate qwen-vl-utils flash-attn --no-build-isolation

echo "Setup Complete!"
