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
from training_script.world_model.atari.train_world_model_atari import AtariDataBuilder

# --- Configuration (Aligned with Atari v4 Training) ---
class AtariInferenceConfig:
    tokenizer_ckpt = Path("checkpoints/atari/tokenizer_v3/best_model.pt")
    wm_ckpt_path = Path("checkpoints/world_model/atari_v4/best_wm.pt")
    latent_path = Path("data/atari/latent_sequences/video_full_frames.pt")
    actions_jsonl = Path("data/atari/raw/actions.jsonl")
    output_dir = Path("inference/results/world_model/atari")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    resize = (64, 64)
    patch_size = 8
    n_latents = 64         
    latent_dim = 256
    
    embed_dim = 512        # World Model (v4)
    tokenizer_dim = 256    # Tokenizer (v3)
    num_layers = 12        # World Model (v4)
    
    action_dim = 4
    Sr = 8                 
    Sa = 1                 
    
    num_ode_steps = 50     
    gen_length = 64        
    fps = 10

# --- Helper Functions ---

def load_atari_models(cfg):
    print(f"Loading Atari models on {cfg.device}...")
    
    # 1. Tokenizer (256-dim as per training v3)
    tokenizer = CausalTokenizer(
        input_dim=3 * cfg.patch_size * cfg.patch_size,
        embed_dim=cfg.tokenizer_dim, 
        num_heads=8,
        num_layers=8,
        latent_dim=cfg.latent_dim,
        use_checkpoint=False,
    )
    tok_state = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in tok_state["model_state"].items()}
    tokenizer.load_state_dict(state_dict)
    tokenizer.to(cfg.device).eval()
    
    # 2. World Model & Builder (512-dim, 12 layers as per training v4)
    world_model = WorldModel(
        d_model=cfg.embed_dim,
        d_latent=cfg.latent_dim,
        num_layers=cfg.num_layers,
        num_heads=8,
        n_latents=cfg.n_latents,
        Sr=cfg.Sr,
        use_checkpoint=False 
    )
    builder = AtariDataBuilder(cfg)
    
    wm_state = torch.load(cfg.wm_ckpt_path, map_location="cpu")
    wm_state = {k.replace("module.", ""): v for k, v in wm_state.items()}
    world_model.load_state_dict(wm_state)
    
    world_model.to(cfg.device).eval()
    builder.to(cfg.device).eval()
    
    return tokenizer, world_model, builder

@torch.no_grad()
def decode_latents_full_context(tokenizer, latents, cfg):
    """Refined decoding with latent range protection."""
    T, N, D = latents.shape
    device = latents.device
    patchifier = Patchifier(cfg.patch_size)
    
    # 1. LATENT CLAMPING: Prevents graininess caused by out-of-bounds WM predictions
    # This keeps the WM output in the same 'neighborhood' as the Tokenizer Encoder.
    latents = torch.clamp(latents, -5.0, 5.0) 
    
    z = latents.unsqueeze(0)
    x = tokenizer.from_latent(z) 
    x = x.view(1, T * N, tokenizer.embed_dim)
    
    # Matches WANDB local context fix
    x = x + tokenizer.pos_embed[:, :T * N, :]
    
    x = tokenizer._run_stack(x, tokenizer.decoder, T=T, N=N)
    x = x.view(1, T, N, tokenizer.embed_dim)
    patches = tokenizer.output_proj(x)
    
    frames = patchifier.unpatchify(patches[0], cfg.resize, cfg.patch_size)
    
    all_frames = []
    for t in range(T):
        # Apply slight clipping here to ensure valid RGB range
        img_np = (frames[t].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        all_frames.append(img_np)
            
    return np.array(all_frames)

@torch.no_grad()
def generate_atari_dream(cfg, wm, builder, tokenizer):
    device = cfg.device
    T = cfg.gen_length
    
    # 1. Load Seed and Actions
    data = torch.load(cfg.latent_path, map_location="cpu")
    seed_z = data["z"][0:1].to(device) 
    
    actions = []
    with open(cfg.actions_jsonl, "r") as f:
        for i, line in enumerate(f):
            if i >= T: break
            actions.append(json.loads(line)["action"])
    actions_tensor = torch.tensor(actions, device=device).unsqueeze(0) 

    # 2. Initialization
    z = torch.randn(1, T, cfg.n_latents, cfg.latent_dim, device=device)
    z[:, 0] = seed_z 
    
    # 3. Solver parameters
    timesteps = torch.linspace(0, 1, cfg.num_ode_steps + 1, device=device)
    dt = 1.0 / cfg.num_ode_steps
    
    print(f"Dreaming {T} frames...")
    pred_z_clean = None
    
    for i in tqdm(range(cfg.num_ode_steps)):
        t_curr = timesteps[i]
        tau = torch.full((1, T), t_curr.item(), device=device)
        
        # FIX 1: Set 'd' to 0.5 (Neutral signal) instead of 0.0
        # This often matches the training distribution mean better.
        d_map = torch.full((1,), 0.5, device=device) 
        
        tokens = builder(z, actions_tensor, tau, d_map)
        wm_input = {"wm_input_tokens": tokens, "tau": tau, "d": d_map}
        
        offsets = torch.zeros(1, dtype=torch.long, device=device)
        pred_z_clean = wm(wm_input, time_offsets=offsets)
        
        # FIX 2: Clamp the clean prediction at every step to keep trajectory stable
        pred_z_clean = torch.clamp(pred_z_clean, -5.0, 5.0)
        
        # ODE Step: v = (z_clean - z) / (1 - t)
        eps = 1e-5
        v_pred = (pred_z_clean - z) / (1.0 - t_curr + eps)
        
        # Euler step for frames 1:T
        z[:, 1:] = z[:, 1:] + (v_pred[:, 1:] * dt)
        
    print("Decoding final clean prediction...")
    # Decoding pred_z_clean (the clean manifold) instead of z (the noisy path)
    # is the secret to WANDB-level quality.
    return decode_latents_full_context(tokenizer, pred_z_clean.squeeze(0), cfg)

def main():
    cfg = AtariInferenceConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer, wm, builder = load_atari_models(cfg)
    video_frames = generate_atari_dream(cfg, wm, builder, tokenizer)
    
    save_path = cfg.output_dir / "final_latest_atari_dream_latest.mp4"
    H, W, _ = video_frames[0].shape
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), cfg.fps, (W, H))
    for frame in video_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"✓ Video saved to {save_path}")

if __name__ == "__main__":
    main()