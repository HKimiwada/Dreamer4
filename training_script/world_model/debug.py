# python training_script/world_model/debug.py
import torch
from pathlib import Path

# Load the first latent file
latent_path = Path("data/latent_sequences/video_00000.pt")
if not latent_path.exists():
    print("❌ Latent file not found.")
else:
    data = torch.load(latent_path)
    z = data["z"] # Shape: (T, N, D)
    
    print(f"Latent shape: {z.shape}")
    
    # Calculate difference between Frame 0 and Frame 1
    diff = (z[1] - z[0]).abs().mean()
    
    print(f"Diff between Frame 0 and Frame 1: {diff.item():.6f}")
    
    if diff.item() < 1e-5:
        print("❌ CRITICAL: Latents are STATIC. The Tokenizer is outputting the same code for every frame.")
        print("   Solution: Retrain Tokenizer or fix 'latent_tokenizer.py'.")
    else:
        print("✅ Latents have motion.")