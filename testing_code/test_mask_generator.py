# python testing_code/test_mask_generator.py
import torch
from tokenizer.patchify_mask import MaskGenerator

# Initialize with default settings
masker = MaskGenerator(mask_prob_range=(0.0, 0.9), per_frame_sampling=True, seed=42)

# Simulate a video clip: T = 64 frames, N = 960 patches per frame
T, N = 64, 960

mask = masker(T=T, N=N)

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)
print("Masked fraction (mean):", mask.float().mean().item())

# Check per-frame variation (since per_frame_sampling=True)
per_frame_fraction = mask.float().mean(dim=1)
print("First 5 frame masked ratios:", per_frame_fraction[:5])

# Verify that values are boolean 0/1
unique_vals = mask.unique(sorted=True)
print("Unique values in mask:", unique_vals.tolist())

# Example of applying the mask to dummy tokens
patches = torch.rand(T, N, 768)
masked_patches = torch.where(mask.unsqueeze(-1), torch.zeros_like(patches), patches)
print("Masked patches shape:", masked_patches.shape)
