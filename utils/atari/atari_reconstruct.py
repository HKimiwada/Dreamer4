import torch
import numpy as np
from PIL import Image
from pathlib import Path

def reconstruct_raw_frame(frame_path, output_name="utils/atari/reconstructed_frame.png"):
    # 1. Load the saved .pt tensor
    frame_tensor = torch.load(frame_path)
    
    # 2. Permute back to (H, W, C) for visualization
    # Original save was .permute(2, 0, 1), so we reverse it
    frame_np = frame_tensor.permute(1, 2, 0).numpy()
    
    # 3. Save as a standard image
    img = Image.fromarray(frame_np)
    img.save(output_name)
    print(f"Frame saved to {output_name}")

# Example usage
reconstruct_raw_frame("data/atari/raw/frame_000100.pt")