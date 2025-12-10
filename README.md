# The Interactive Visual Digital Twin
**Course:** COMS W 4995 - Deep Learning for Computer Vision (Columbia University)
**Author:** Aman Upganlawar (au2327)

## Project Overview
This project implements an autonomous, vision-based agent capable of inspecting a high-fidelity 3D digital twin of the Columbia University "Alma Mater" statue. The system integrates **3D Gaussian Splatting (3DGS)** for explicit scene reconstruction with a local **Vision-Language Model (Qwen2-VL)** for semantic reasoning.

The agent can "physically" move a virtual camera to specific coordinates to inspect objects (e.g., looking at the back of the statue) and answer questions about them in real-time.

## Architecture: The Microservice Solution
Due to dependency conflicts between `nerfstudio` (requires PyTorch 2.1) and modern VLMs (require PyTorch 2.4+), this project uses a decoupled microservice architecture:

1.  **The Environment (Controller):** Runs in `nerfstudio` env. Handles 3D rendering and path planning.
2.  **The Brain (Service):** Runs in `brain` env. Hosts the Qwen2-VL-7B model.
3.  **Communication:** The controller spawns the brain as a subprocess, passing rendered frames via the file system.

## Hardware Requirements
* **GPU:** NVIDIA L4 (24GB VRAM) or A100.
* **Disk:** 150GB (Dataset + Model Weights).
* **OS:** Linux (Debian 11 / Ubuntu 20.04).

## Installation

### 1. Environment Setup
Run the included script to create the two necessary Conda environments:
```bash
bash setup_envs.sh

2. Download Model Weights
Activate the brain environment and download the VLM weights (requires internet):

Bash

conda activate brain
python download_model.py
Reproduction Steps
Step 1: Data Processing
Place your video file in data/video.mp4.

Bash

conda activate nerfstudio
# Process video into keyframes and run COLMAP SfM
ns-process-data video --data data/video.mp4 --output-dir data/alma_mater
Step 2: Training the Digital Twin
We use splatfacto. Crucial: We force 1080p resolution (scale-factor 1.0) to ensure text legibility.

Bash

ns-train splatfacto --data data/alma_mater \
    --pipeline.datamanager.camera-res-scale-factor 1.0 \
    --max-num-iterations 30000
Step 3: Running the Agent
Configure: Update MODEL_CONFIG in camera_control.py with the path to your trained config.yml.

Launch:

Bash

conda activate nerfstudio
python camera_control.py
Usage
Once the agent is running, you can interact with it via the terminal:

Plaintext

USER QUERY: What does the design on the back of the statue signify?
AGENT STATUS: Keyword detected. Navigating to Camera Index 102.
AGENT ACTION: Rendering high-fidelity view...
AGENT RESPONSE: The design features two children holding a lamp, symbolizing the light of knowledge...
Acknowledgements
Nerfstudio: For the 3DGS implementation.

Qwen-VL: For the vision-language model.

Google Cloud: For compute infrastructure.