import torch
import numpy as np
import cv2
from pathlib import Path
from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier

@torch.no_grad()
def test_latent_file_quality():
    # 1. Setup paths
    latent_path = Path("data/atari/latent_sequences/video_full_frames.pt")
    tokenizer_ckpt = Path("checkpoints/atari/tokenizer_v3/best_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not latent_path.exists():
        print(f"Error: {latent_path} not found.")
        return

    # 2. Load Tokenizer (v3)
    # Architecture must match your training config
    tokenizer = CausalTokenizer(
        input_dim=192, embed_dim=256, num_heads=8, num_layers=8, latent_dim=256
    )
    ckpt = torch.load(tokenizer_ckpt, map_location="cpu")
    tokenizer.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["model_state"].items()})
    tokenizer.to(device).eval()

    # 3. Load Latent Data
    print(f"Loading latents from {latent_path}...")
    data = torch.load(latent_path, map_location="cpu")
    all_z = data["z"] # (Total_Frames, N, D)
    
    # Grab a slice of 64 frames (matching your training clip_length)
    z_slice = all_z[100:164].unsqueeze(0).to(device) # (B=1, T=64, N=64, D=256)

    # 4. Decode
    print("Decoding latent slice...")
    # Project from bottleneck
    x = tokenizer.from_latent(z_slice) 
    B, T, N, E = x.shape
    x = x.view(B, T * N, E)
    
    # Run through decoder stack with full T=64 context
    x = tokenizer._run_stack(x, tokenizer.decoder, T=T, N=N)
    
    # Project back to pixels
    x = x.view(B, T, N, E)
    patches = tokenizer.output_proj(x)
    
    # Unpatchify the last frame of the slice
    patchifier = Patchifier(patch_size=8)
    frames = patchifier.unpatchify(patches.squeeze(0), (64, 64), 8)
    target_frame = frames[-1].clamp(0, 1) # Look at the 64th frame for best context

    # 5. Save and Inspect
    img_np = (target_frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite("test_latent_quality.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    print("✓ Result saved to 'test_latent_quality.png'.")
    print("If this image is pixelated/grainy, your 'video_full_frames.pt' is low quality.")

if __name__ == "__main__":
    test_latent_file_quality()