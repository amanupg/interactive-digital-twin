import os
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
from nerfstudio.utils.eval_utils import eval_setup

# --- CONFIGURATION ---
# UPDATE THIS PATH to point to your trained config.yml
MODEL_CONFIG = "/home/au2327/outputs/alma_mater/splatfacto/2025-12-01_233012/config.yml"

# PATHS to the Brain Environment
BRAIN_PYTHON = "/home/au2327/miniconda3/envs/brain/bin/python"
BRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "brain_service.py")

# Disable torch.compile (Dynamo) for L4 GPU stability
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

class SceneRenderer:
    def __init__(self, config_path):
        print(f"Loading 3DGS pipeline from: {config_path}")
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            print(f"Error: Configuration file not found at {self.config_path}")
            sys.exit(1)

        self.config, self.pipeline, self.checkpoint_path, _ = eval_setup(
            self.config_path,
            test_mode="inference",
        )
        print("Pipeline initialization complete.")

    def render_view(self, camera_index=0, save_path="view.jpg"):
        """Renders a novel view from the trained splatfacto model."""
        try:
            # Retrieve camera intrinsics/extrinsics from dataset
            camera = self.pipeline.datamanager.train_dataset.cameras[camera_index : camera_index + 1]
        except IndexError:
            print(f"Error: Camera index {camera_index} is out of valid range.")
            return None

        # Inference: Generate RGB output
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera(camera)
        
        # Handle model output variance
        if 'rgb' not in outputs:
            if 'rgb_fine' in outputs: outputs['rgb'] = outputs['rgb_fine']
            else: return None

        # Save rasterized output to disk
        image_np = outputs["rgb"].cpu().numpy()
        plt.imsave(save_path, image_np)
        return save_path

class RemoteBrain:
    def analyze(self, image_path, question):
        """Invokes the VLM microservice via subprocess call."""
        print(f"Sending visual data to VLM service (Qwen2-VL-7B)...")
        
        result = subprocess.run(
            [BRAIN_PYTHON, BRAIN_SCRIPT, image_path, question],
            capture_output=True, 
            text=True
        )
        
        stdout = result.stdout
        if "BRAIN_OUTPUT_START" in stdout:
            # Parse delimiter-separated response
            answer = stdout.split("BRAIN_OUTPUT_START|")[1].split("|BRAIN_OUTPUT_END")[0]
            return answer
        else:
            return f"Error: VLM service failed. Log: {stdout}"

if __name__ == "__main__":
    renderer = SceneRenderer(MODEL_CONFIG)
    brain = RemoteBrain()
    
    # Semantic mapping of locations (Simulating agent knowledge graph)
    # You must update these indices based on your own training data
    LOCATIONS = {
        "book": 45,       
        "back": 102,      
        "base": 15,       
        "default": 0
    }

    print("\n" + "="*40)
    print("   INTERACTIVE DIGITAL TWIN AGENT")
    print("="*40 + "\n")

    while True:
        try:
            user_query = input("USER QUERY: ")
        except KeyboardInterrupt: break
        if user_query.lower() in ["exit", "quit"]: break

        # Simulated Planning Logic
        cam_idx = LOCATIONS["default"]
        if "book" in user_query.lower(): cam_idx = LOCATIONS["book"]
        elif "text" in user_query.lower() or "base" in user_query.lower(): cam_idx = LOCATIONS["base"]
        elif "back" in user_query.lower() or "design" in user_query.lower(): cam_idx = LOCATIONS["back"]

        print(f"AGENT STATUS: Keyword detected. Navigating to Camera Index {cam_idx}.")
        
        # Execute Visual Inspection
        print(f"AGENT ACTION: Rendering high-fidelity view...")
        abs_img_path = str(Path("agent_view.jpg").resolve())
        renderer.render_view(camera_index=cam_idx, save_path=abs_img_path)
        
        # Perform Semantic Analysis
        answer = brain.analyze(abs_img_path, user_query)
        print(f"\nAGENT RESPONSE: {answer}\n")
        print("-" * 40)