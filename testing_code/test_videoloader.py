# python testing_code/test_videoloader.py
from pathlib import Path
from tokenizer.dataset import VideoLoader
import numpy as np
import imageio.v2 as imageio

video_dir = Path("data")  # Replace with your video directory path
loader = VideoLoader(video_dir=video_dir, target_fps=20.0, resize=(384, 640), max_frames=64)
loader_v2 = VideoLoader(video_dir=video_dir, target_fps=20.0, resize=(384, 640), max_frames=None)    

for idx in range(3):
    frames, metadata = loader[idx]
    print(f"Testing loader with max_frames=64 for video index {idx}")
    print(f"Video {idx}:")
    print(f"  Frames shape: {frames.shape}")
    print(f"  Metadata: {metadata}")

for idx in range(3):
    frames, metadata = loader[idx]
    print(f"Testing loader with max_frames=None for video index {idx}")
    print(f"Video {idx}:")
    print(f"  Frames shape: {frames.shape}")
    print(f"  Metadata: {metadata}")

"""
print("Testing reconstruction to ensure that tensors represent the same video frames after processing.")
frames, metadata = loader[10]
frames_np = frames.permute(0,2,3,1).cpu().numpy()  # Convert to numpy array with shape (T, H, W, C)
frames_np = np.clip(frames_np * 255.0, 0, 255).astype(np.uint8)

out_path = "reconstruction.mp4"
imageio.mimsave(out_path, frames_np, fps=20)
print(f"Saved reconstructed clip → {out_path}")
"""

