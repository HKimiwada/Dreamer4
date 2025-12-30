import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Import components from your project
from tokenizer.model.encoder_decoder import CausalTokenizer
from world_model.wm.dynamics_model_atari import WorldModel
from tokenizer.patchify_mask import Patchifier
from training_script.world_model.atari.train_world_model_atari import AtariWMConfig, AtariWorldModelDataset, AtariDataBuilder

@torch.no_grad()
def run_diagnostic():
    # 1. Setup Config and Device
    cfg = AtariWMConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Path to your latest V2 model
    wm_ckpt_path = Path("checkpoints/world_model/atari_v2/best_wm.pt")
    
    print(f"Running diagnostic using WM: {wm_ckpt_path}")
    
    # 2. Initialize Models (Matching your specific Atari architecture)
    # Tokenizer is 256-dim
    tokenizer = CausalTokenizer(
        input_dim=cfg.input_dim, 
        embed_dim=256, 
        num_heads=8, 
        num_layers=8, 
        latent_dim=256
    )
    tk_ckpt = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
    tokenizer.load_state_dict({k.replace("module.", ""): v for k, v in tk_ckpt["model_state"].items()})
    tokenizer.to(device).eval()

    builder = AtariDataBuilder(cfg).to(device).eval()

    # World Model is 512-dim
    wm = WorldModel(
        d_model=cfg.embed_dim, 
        d_latent=cfg.latent_dim, 
        num_layers=cfg.num_layers, 
        num_heads=cfg.num_heads, 
        n_latents=cfg.n_latents, 
        Sr=cfg.Sr, 
        use_checkpoint=False
    ).to(device)
    
    if wm_ckpt_path.exists():
        wm_state = torch.load(wm_ckpt_path, map_location="cpu")
        wm.load_state_dict(wm_state)
        print("✓ Loaded World Model weights.")
    else:
        print("! Warning: WM checkpoint not found. Testing with random weights.")
    wm.eval()

    # 3. Load Data
    dataset = AtariWorldModelDataset(cfg)
    sample = dataset[0]
    latents = sample["latents"].unsqueeze(0).to(device) # (1, T, N, D)
    actions = sample["actions"].unsqueeze(0).to(device)
    start_idx = sample["start_idx"]
    B, T, N, D = latents.shape

    # 4. Scenario A: Clean Reconstruction (tau=1.0, d=0)
    # This tests if the WM can handle perfect data.
    tau_clean = torch.ones((B, T), device=device)
    d_clean = torch.zeros(B, device=device)
    tokens_clean = builder(latents, actions, tau_clean, d_clean)
    pred_clean = wm({"wm_input_tokens": tokens_clean, "tau": tau_clean, "d": d_clean}, 
                     time_offset=start_idx)

    # 5. Scenario B: Denoising Reconstruction (tau=0.5, d=0.25)
    # This tests the "Shortcut Forcing" capability.
    tau_noise = torch.full((B, T), 0.5, device=device)
    d_noise = torch.full((B,), 0.25, device=device)
    noise = torch.randn_like(latents)
    z_corr = (1.0 - 0.5) * noise + 0.5 * latents
    
    tokens_noise = builder(z_corr, actions, tau_noise, d_noise)
    pred_noise = wm({"wm_input_tokens": tokens_noise, "tau": tau_noise, "d": d_noise}, 
                     time_offset=start_idx)

    # 6. Decoding Logic (Corrected: No manual pos_embed)
    @torch.no_grad()
    def decode_frame_strip(z_seq):
        """
        Decodes (T, N, D) latents to (T, H, W, 3) in training-sized chunks.
        """
        T, N, D = z_seq.shape
        chunk_size = 4  # MUST match tokenizer training clip_length
        all_frames = []
        patchifier = Patchifier(cfg.patch_size)

        # Process in chunks of 4 to match positional embedding training
        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)
            T_curr = t_end - t_start
            
            # 1. Slice and project
            z_chunk = z_seq[t_start:t_end].unsqueeze(0) # (1, T_curr, N, D)
            x = tokenizer.from_latent(z_chunk) 
            x = x.view(1, T_curr * N, tokenizer.embed_dim)
            
            # 2. Run Decoder Stack
            # Passing T_curr ensures the attention mask matches training
            x = tokenizer._run_stack(x, tokenizer.decoder, T=T_curr, N=N)
            
            # 3. Project to pixels
            x = x.view(1, T_curr, N, tokenizer.embed_dim)
            patches = tokenizer.output_proj(x)
            
            # 4. Unpatchify
            chunk_frames = patchifier.unpatchify(patches.squeeze(0), cfg.resize, cfg.patch_size)
            all_frames.append(chunk_frames.clamp(0, 1))

        return torch.cat(all_frames, dim=0)

    print("Decoding diagnostic columns...")
    gt_frames = decode_frame_strip(latents[0, :4])
    clean_frames = decode_frame_strip(pred_clean[0, :4])
    noise_frames = decode_frame_strip(pred_noise[0, :4])

    # 7. Stack into Diagnostic Grid [GT | Clean Pred | Denoised Pred]
    rows = []
    for i in range(4):
        # Concatenate horizontally
        combined = torch.cat([gt_frames[i], clean_frames[i], noise_frames[i]], dim=-1)
        # Scale to uint8 HWC
        img_np = (combined.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rows.append(img_np)
    
    final_diag = np.concatenate(rows, axis=0)
    cv2.imwrite("diagnostic_reconstruction.png", cv2.cvtColor(final_diag, cv2.COLOR_RGB2BGR))
    print(f"✓ Diagnostic saved to 'diagnostic_reconstruction.png'.")
    print("Column 1: Ground Truth | Column 2: Clean Pred (1.0) | Column 3: Denoised Pred (0.5)")

if __name__ == "__main__":
    run_diagnostic()