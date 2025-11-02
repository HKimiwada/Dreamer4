# python testing_code/test_patchify.py      
from tokenizer.patchify_mask import Patchifier
import torch
import einops

frames = torch.rand(64, 3, 384, 640)  # Stage B output
patchify = Patchifier(patch_size=16)
tokens = patchify(frames)
print("Patch tokens:", tokens.shape)

# sanity check: reverse reconstruction
recon = patchify.unpatchify(tokens, (384, 640))
print("Reconstructed:", recon.shape)
print("MSE difference:", (frames - recon).pow(2).mean().item())
