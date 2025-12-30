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
# Using the builder logic from your Atari training script
from training_script.world_model.atari.train_world_model_atari import AtariDataBuilder

# --- Configuration (Must match your Atari training config) ---
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
    num_ode_steps = 50     # Flow Matching Euler steps
    gen_length = 64        # Sequence length
    chunk_size = 16        # Small chunks for memory efficiency
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
    world_model = WorldModel(
        d_model=cfg.embed_dim,
        d_latent=cfg.latent_dim,
        num_layers=8,
        num_heads=8,
        n_latents=cfg.n_latents,
        Sr=cfg.Sr,
        use_checkpoint=False 
    )
    builder = AtariDataBuilder(cfg)
    
    wm_state = torch.load(cfg.wm_ckpt_path, map_location="cpu")
    # Strip DDP 'module.' prefix if needed
    wm_state = {k.replace("module.", ""): v for k, v in wm_state.items()}
    world_model.load_state_dict(wm_state)
    
    world_model.to(cfg.device).eval()
    builder.to(cfg.device).eval()
    
    return tokenizer, world_model, builder

@torch.no_grad()
def decode_latents_chunked(tokenizer, latents, cfg):
    """
    Decodes (T, N, D) latents to (T, H, W, 3) images.
    Uses chunking to match tokenizer training temporal window (4 frames).
    """
    T, N, D = latents.shape
    device = latents.device
    chunk_size = 4 # Match tokenizer training clip_length
    
    all_frames = []
    patchifier = Patchifier(cfg.patch_size)
    
    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        T_curr = t_end - t_start
        
        z_chunk = latents[t_start:t_end].unsqueeze(0)
        
        x = tokenizer.from_latent(z_chunk) 
        x = x.view(1, T_curr * N, tokenizer.embed_dim)
        
        # Add Positional Embeddings for current context
        seq_len = T_curr * N
        x = x + tokenizer.pos_embed[:, :seq_len, :]
        
        x = tokenizer._run_stack(x, tokenizer.decoder, T=T_curr, N=N)
        
        x = x.view(1, T_curr, N, tokenizer.embed_dim)
        patches = tokenizer.output_proj(x)
        
        for t in range(T_curr):
            frame = patchifier.unpatchify(patches[0, t:t+1], cfg.resize, cfg.patch_size)[0]
            frame_np = (frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            all_frames.append(frame_np)
            
    return np.array(all_frames)

@torch.no_grad()
def generate_atari_dream(cfg, wm, builder, tokenizer):
    device = cfg.device
    T = cfg.gen_length
    
    # 1. Load Seed and Actions
    print("Loading seed and actions...")
    data = torch.load(cfg.latent_path, map_location="cpu")
    seed_z = data["z"][0:1].to(device) # Use first real frame as seed
    
    actions = []
    with open(cfg.actions_jsonl, "r") as f:
        for i, line in enumerate(f):
            if i >= T: break
            actions.append(json.loads(line)["action"])
    actions_tensor = torch.tensor(actions, device=device).unsqueeze(0) # (1, T)

    # 2. ODE Initialization
    z = torch.randn(1, T, cfg.n_latents, cfg.latent_dim, device=device)
    z[:, 0] = seed_z 
    
    timesteps = torch.linspace(0, 1, cfg.num_ode_steps + 1, device=device)
    dt = 1.0 / cfg.num_ode_steps
    
    # 3. ODE Solver Loop
    for i in tqdm(range(cfg.num_ode_steps)):
        t_curr = timesteps[i]
        tau_map = torch.full((1, T), t_curr.item(), device=device)
        d_map = torch.zeros(1, device=device)
        
        pred_chunks = []
        for t_start in range(0, T, cfg.chunk_size):
            t_end = min(t_start + cfg.chunk_size, T)
            curr_len = t_end - t_start
            
            z_chunk = z[:, t_start:t_end]
            tau_chunk = tau_map[:, t_start:t_end]
            act_chunk = actions_tensor[:, t_start:t_end]
            
            # Use Atari Builder logic
            tokens = builder(z_chunk, act_chunk, tau_chunk, d_map)
            
            wm_input = {
                "wm_input_tokens": tokens,
                "tau": tau_chunk,
                "d": d_map
            }
            
            pred_chunk = wm(wm_input, time_offset=t_start)
            pred_chunks.append(pred_chunk)
            
        pred_z_clean = torch.cat(pred_chunks, dim=1)
        
        # Euler update (Skipping the seed frame update)
        v_pred = pred_z_clean - z
        z[:, 1:] = z[:, 1:] + (v_pred[:, 1:] * dt)
        
    print("Decoding latents...")
    return decode_latents_chunked(tokenizer, z.squeeze(0), cfg)

def main():
    cfg = AtariInferenceConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer, wm, builder = load_atari_models(cfg)
    video_frames = generate_atari_dream(cfg, wm, builder, tokenizer)
    
    save_path = cfg.output_dir / "atari_dream_latest.mp4"
    H, W, _ = video_frames[0].shape
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), cfg.fps, (W, H))
    for frame in video_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"✓ Video saved to {save_path}")

if __name__ == "__main__":
    main()