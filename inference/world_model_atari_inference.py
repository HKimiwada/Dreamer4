import torch
import torch.nn as nn
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm

from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier
from world_model.wm.dynamics_model_atari import WorldModel
# Note: Use your Atari builder logic from the training script
from training_script.world_model.atari.train_world_model_atari import AtariDataBuilder

# --- Configuration (MUST match your training config) ---
class AtariInferenceConfig:
    # Paths
    tokenizer_ckpt = Path("checkpoints/atari/tokenizer_v2/best_model.pt")
    wm_ckpt_path = Path("checkpoints/world_model/atari_v1/best_wm.pt")
    latent_path = Path("data/atari/latent_sequences/video_full_frames.pt")
    actions_jsonl = Path("data/atari/raw/actions.jsonl")
    output_dir = Path("inference/results/world_model/atari")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Architecture
    resize = (64, 64)
    patch_size = 8
    n_latents = 64         # 8x8 patches
    latent_dim = 256
    embed_dim = 256
    action_dim = 4
    Sr = 8                 # Register tokens
    Sa = 1                 # Action tokens
    
    # Inference Params
    num_ode_steps = 10     # Steps for Euler solver (Flow Matching)
    gen_length = 64        # Number of frames to "dream"
    fps = 10

# --- Helper Functions ---

def load_atari_models(cfg):
    print(f"Loading Atari models on {cfg.device}...")
    
    # 1. Tokenizer
    tokenizer = CausalTokenizer(
        input_dim=3 * cfg.patch_size * cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=8,
        num_layers=8,
        latent_dim=cfg.latent_dim,
        use_checkpoint=False,
    )
    tok_state = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in tok_state["model_state"].items()}
    tokenizer.load_state_dict(state_dict)
    tokenizer.to(cfg.device).eval()
    
    # 2. World Model & Builder
    # Initialize with training config params
    world_model = WorldModel(
        d_model=cfg.embed_dim,
        d_latent=cfg.latent_dim,
        num_layers=8,
        num_heads=8,
        n_latents=cfg.n_latents,
        Sr=cfg.Sr,
        use_checkpoint=False 
    )
    # Mock cfg for builder
    builder = AtariDataBuilder(cfg)
    
    wm_state = torch.load(cfg.wm_ckpt_path, map_location="cpu")
    # If the checkpoint was saved from DDP, strip 'module.'
    wm_state = {k.replace("module.", ""): v for k, v in wm_state.items()}
    world_model.load_state_dict(wm_state)
    
    world_model.to(cfg.device).eval()
    builder.to(cfg.device).eval()
    
    return tokenizer, world_model, builder

@torch.no_grad()
def decode_latents_to_video(tokenizer, latents, cfg):
    """
    Decodes (T, N, D) latents to (T, H, W, 3) images in chunks.
    Matches the training clip_length (4) to avoid pos_embed noise.
    """
    T, N, D = latents.shape
    device = latents.device
    chunk_size = 4 # MUST match tokenizer training clip_length
    
    all_frames = []
    patchifier = Patchifier(cfg.patch_size)
    
    # Process in chunks of 4 to match training temporal window
    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        T_curr = t_end - t_start
        
        # 1. Prepare Chunk (1, T_curr, N, D)
        z_chunk = latents[t_start:t_end].unsqueeze(0)
        
        # 2. Latent -> Embeddings
        x = tokenizer.from_latent(z_chunk) # (1, T_curr, N, E)
        x = x.view(1, T_curr * N, tokenizer.embed_dim)
        
        # 3. Add Positional Embeddings (Same indices 0..255 used in training)
        seq_len = T_curr * N
        x = x + tokenizer.pos_embed[:, :seq_len, :]
        
        # 4. Decoder Stack
        x = tokenizer._run_stack(x, tokenizer.decoder, T=T_curr, N=N)
        
        # 5. Project to Pixels
        x = x.view(1, T_curr, N, tokenizer.embed_dim)
        patches = tokenizer.output_proj(x)
        
        # 6. Unpatchify frames in this chunk
        for t in range(T_curr):
            frame = patchifier.unpatchify(patches[0, t:t+1], cfg.resize, cfg.patch_size)[0]
            # Convert to [0, 255] uint8 HWC
            frame_np = (frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            all_frames.append(frame_np)
            
    return np.array(all_frames)

@torch.no_grad()
def dream_sequence(cfg, wm, builder, tokenizer):
    device = cfg.device
    T = cfg.gen_length
    
    # 1. Load Ground Truth for the SEED frame
    print("Loading seed frame and actions...")
    data = torch.load(cfg.latent_path, map_location="cpu")
    all_latents = data["z"] # [100000, 64, 256]
    
    # Start the dream with frame 0 from your dataset
    seed_z = all_latents[0:1].to(device) # (1, 64, 256)
    
    actions = []
    with open(cfg.actions_jsonl, "r") as f:
        for i, line in enumerate(f):
            if i >= T: break
            actions.append(json.loads(line)["action"])
    actions_tensor = torch.tensor(actions, device=device).unsqueeze(0)

    # 2. Initialize sequence: Seed frame + Noise for the rest
    # z: (1, T, 64, 256)
    z = torch.randn(1, T, cfg.n_latents, cfg.latent_dim, device=device)
    z[:, 0] = seed_z # Set the first frame to ground truth

    # 3. ODE Solver
    num_steps = 50 # Increased for better quality
    dt = 1.0 / num_steps
    
    for i in tqdm(range(num_steps)):
        t_val = i / num_steps
        tau = torch.full((1, T), t_val, device=device)
        # IMPORTANT: d=0 indicates we are looking for the 'cleanest' prediction
        d = torch.zeros(1, device=device) 
        
        # Builder creates tokens including actions, registers, and shortcut(tau, d)
        tokens = builder(z, actions_tensor, tau, d)
        
        # Dynamics model expects the full dict including 'd'
        input_dict = {
            "wm_input_tokens": tokens, 
            "tau": tau,
            "d": d 
        }
        
        pred_z_clean = wm(input_dict, time_offset=0)
        
        # Euler update (Don't update the seed frame)
        v_pred = pred_z_clean - z
        z[:, 1:] = z[:, 1:] + (v_pred[:, 1:] * dt)
        
    print("Decoding dream...")
    return decode_latents_to_video(tokenizer, z.squeeze(0), cfg)

def main():
    cfg = AtariInferenceConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer, wm, builder = load_atari_models(cfg)
    
    video_frames = dream_sequence(cfg, wm, builder, tokenizer)
    
    save_path = cfg.output_dir / "dream_test.mp4"
    H, W, _ = video_frames[0].shape
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), cfg.fps, (W, H))
    
    for frame in video_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"✓ Video saved to {save_path}")

if __name__ == "__main__":
    main()