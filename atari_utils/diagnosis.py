import torch
import numpy as np
import cv2
from pathlib import Path

# Import components from your project
from tokenizer.model.encoder_decoder import CausalTokenizer
from world_model.wm.dynamics_model_atari import WorldModel
from tokenizer.patchify_mask import Patchifier
from training_script.world_model.atari.train_world_model_atari import (
    AtariWMConfig, 
    AtariWorldModelDataset, 
    AtariDataBuilder
)

@torch.no_grad()
def run_visualization_test():
    # 1. Setup Config and Device
    cfg = AtariWMConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths to your trained V3 weights
    wm_ckpt_path = Path("checkpoints/world_model/atari_v3/best_wm.pt")
    tokenizer_ckpt = Path("checkpoints/atari/tokenizer_v3/best_model.pt")
    
    print(f"--- World Model Visualization Diagnostic (V3) ---")
    
    # 2. Initialize Models (V3 Architectures)
    tokenizer = CausalTokenizer(
        input_dim=cfg.input_dim, 
        embed_dim=256, 
        num_heads=8, 
        num_layers=8, 
        latent_dim=256
    )
    if tokenizer_ckpt.exists():
        tk_data = torch.load(tokenizer_ckpt, map_location="cpu")
        tokenizer.load_state_dict({k.replace("module.", ""): v for k, v in tk_data["model_state"].items()})
        print("✓ Loaded Tokenizer V3 weights.")
    tokenizer.to(device).eval()

    wm = WorldModel(
        d_model=cfg.embed_dim, 
        d_latent=cfg.latent_dim, 
        num_layers=cfg.num_layers, 
        num_heads=cfg.num_heads, 
        n_latents=cfg.n_latents, 
        Sr=cfg.Sr, 
        use_checkpoint=False
    )
    if wm_ckpt_path.exists():
        wm_state = torch.load(wm_ckpt_path, map_location="cpu")
        wm.load_state_dict({k.replace("module.", ""): v for k, v in wm_state.items()})
        print("✓ Loaded World Model V3 weights.")
    else:
        print("! Warning: WM checkpoint not found. Results will be random.")
    wm.to(device).eval()

    builder = AtariDataBuilder(cfg).to(device).eval()

    # 3. Load Data Sample (Full Sequence)
    dataset = AtariWorldModelDataset(cfg)
    sample = dataset[100] 
    
    # Maintain full clip_length (64) context
    latents = sample["latents"].unsqueeze(0).to(device) # (1, T=64, N=64, D=256)
    actions = sample["actions"].unsqueeze(0).to(device) # (1, T=64)
    start_idx = torch.tensor([sample["start_idx"]], device=device) # (1,)
    
    B, T, N, D = latents.shape

    # 4. Scenario: Denoising Reconstruction (tau=0.5)
    tau = torch.full((B, T), 0.5, device=device)
    d = torch.full((B,), 0.25, device=device)
    noise = torch.randn_like(latents)
    z_corr = (1.0 - 0.5) * noise + 0.5 * latents
    
    tokens = builder(z_corr, actions, tau, d)
    wm_input = {
        "wm_input_tokens": tokens, 
        "tau": tau, 
        "d": d, 
        "z_clean": latents, 
        "z_corrupted": z_corr
    }
    
    pred_z = wm(wm_input, time_offsets=start_idx)

    # 5. FIXED DECODING LOGIC: Manual Positional Embedding Injection
    def decode_full_then_slice(z_seq):
        """
        FIXED: Explicitly injects local positional embeddings (0-63).
        This removes corruption by ensuring the Tokenizer decoder uses 
        the exact coordinate system it learned during training.
        """
        B_val, T_val, N_val, D_val = z_seq.shape # (1, 64, 64, 256)
        
        # A. Project from latent bottleneck to embedding space
        x = tokenizer.from_latent(z_seq) # (B, T, N, E)
        x = x.view(B_val, T_val * N_val, tokenizer.embed_dim)
        
        # B. CRITICAL FIX: Manually add Local Positional Embeddings
        # The Tokenizer only knows indices 0 to (64*64)-1.
        # Global offsets (like 50,000) will break the reconstruction.
        x = x + tokenizer.pos_embed[:, :T_val * N_val, :]
        
        # C. Run through the Decoder Transformer stack
        x = tokenizer._run_stack(x, tokenizer.decoder, T=T_val, N=N_val)
        
        # D. Project back to pixel space
        x = x.view(B_val, T_val, N_val, tokenizer.embed_dim)
        patches = tokenizer.output_proj(x)
        
        # E. Unpatchify (T, C, H, W)
        full_frames = Patchifier(cfg.patch_size).unpatchify(
            patches.squeeze(0), cfg.resize, cfg.patch_size
        )
        
        # Return the LAST 4 frames (maximum temporal context)
        return full_frames[-4:]

    print("Decoding frames with Local Positional Context...")
    # Pass the full 64-frame sequences
    gt_frames = decode_full_then_slice(latents)
    pred_frames = decode_full_then_slice(pred_z)

    # 6. Build Diagnostic Grid
    rows = []
    for i in range(4):
        # Clamp to [0, 1] to prevent weird color artifacts
        gt_img = gt_frames[i].clamp(0, 1)
        pr_img = pred_frames[i].clamp(0, 1)
        
        combined = torch.cat([gt_img, pr_img], dim=-1)
        img_np = (combined.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rows.append(img_np)
    
    final_diag = np.concatenate(rows, axis=0)
    output_path = "final_wm_reconstruction_perfect.png"
    cv2.imwrite(output_path, cv2.cvtColor(final_diag, cv2.COLOR_RGB2BGR))
    
    print(f"✓ Diagnostic saved to '{output_path}'.")

if __name__ == "__main__":
    run_visualization_test()