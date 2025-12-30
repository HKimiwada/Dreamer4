import torch
import numpy as np
import cv2
from pathlib import Path
from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier
from training_script.world_model.atari.train_world_model_atari import AtariWMConfig

@torch.no_grad()
def compare_tokenizers():
    cfg = AtariWMConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Paths
    v1_path = Path("checkpoints/atari/tokenizer_v1/best_model.pt")
    v2_path = Path("checkpoints/atari/tokenizer_v2/best_model.pt")
    raw_frame_path = sorted(list(Path("data/atari/raw").glob("frame_*.pt")))[0]
    raw_frame = torch.load(raw_frame_path, map_location="cpu").float() / 255.0

    def run_reconstruction(ckpt_path, patch_size):
        if not ckpt_path.exists():
            print(f"Skipping {ckpt_path} (Not found)")
            return None
            
        # A. Initialize architecture specific to this checkpoint
        input_dim = 3 * patch_size * patch_size
        model = CausalTokenizer(
            input_dim=input_dim, 
            embed_dim=256, 
            num_heads=8, 
            num_layers=8, 
            latent_dim=256
        )
        
        # B. Load Weights
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model_state"].items()}
        model.load_state_dict(state_dict)
        model.to(device).eval()
        
        # C. Patchify (Returns (T=1, N, D))
        p = Patchifier(patch_size=patch_size)
        patches = p(raw_frame.unsqueeze(0)).to(device)
        
        # D. Reconstruct
        # mask must be (B, T, N). Current shape is (T, N).
        mask = torch.zeros(1, patches.shape[1], dtype=torch.bool, device=device)
        
        # --- FIX: Add Batch dimension (unsqueeze(0)) ---
        # Resulting shapes: patches=(1, 1, N, D), mask=(1, 1, N)
        recon = model(patches.unsqueeze(0), mask.unsqueeze(0))
        
        # E. Unpatchify
        # unpatchify expects (T, N, D). Remove batch dimension (squeeze(0)).
        img = p.unpatchify(recon.squeeze(0), cfg.resize, patch_size)
        return img[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    print("--- Diagnostic Comparison ---")
    # V1 used patch_size=16 (input_dim 768)
    img_v1 = run_reconstruction(v1_path, patch_size=16)
    
    # V2 uses patch_size=8 (input_dim 192)
    img_v2 = run_reconstruction(v2_path, patch_size=8)

    # 3. Save side-by-side [Original | V1 (16x16) | V2 (8x8)]
    orig_np = raw_frame.permute(1, 2, 0).cpu().numpy()
    cols = [orig_np]
    if img_v1 is not None: cols.append(img_v1)
    if img_v2 is not None: cols.append(img_v2)
    
    final = (np.concatenate(cols, axis=1) * 255).astype(np.uint8)
    cv2.imwrite("tokenizer_comparison.png", cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"✓ Comparison saved to 'tokenizer_comparison.png'")

if __name__ == "__main__":
    compare_tokenizers()