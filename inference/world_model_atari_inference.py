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
    """Converts (T, N, D) latents to (T, H, W, 3) images."""
    T, N, D = latents.shape
    z_in = latents.unsqueeze(0) # (1, T, N, D)
    
    x = tokenizer.from_latent(z_in)
    x = x.view(1, T * N, tokenizer.embed_dim)
    x = tokenizer._run_stack(x, tokenizer.decoder, T=T, N=N)
    x = x.view(1, T, N, tokenizer.embed_dim)
    patches = tokenizer.output_proj(x)
    
    patchifier = Patchifier(cfg.patch_size)
    frames = patchifier.unpatchify(patches.squeeze(0), cfg.resize, cfg.patch_size)
    
    # Convert to numpy uint8
    frames_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    return frames_np

@torch.no_grad()
def dream_sequence(cfg, wm, builder, tokenizer):
    """Generates a sequence using the ODE solver."""
    device = cfg.device
    T = cfg.gen_length
    
    # 1. Load Actions for conditioning
    # We take a slice of actions from your real data to see if the physics match
    print("Loading actions from ground truth...")
    actions = []
    with open(cfg.actions_jsonl, "r") as f:
        for i, line in enumerate(f):
            if i >= T: break
            actions.append(json.loads(line)["action"])
    actions_tensor = torch.tensor(actions, device=device).unsqueeze(0) # (1, T)

    # 2. ODE Solver (Euler Method)
    # Start with pure noise
    z = torch.randn(1, T, cfg.n_latents, cfg.latent_dim, device=device)
    dt = 1.0 / cfg.num_ode_steps
    
    print(f"Dreaming {T} frames via {cfg.num_ode_steps} ODE steps...")
    for i in tqdm(range(cfg.num_ode_steps)):
        tau = torch.full((1, T), i / cfg.num_ode_steps, device=device)
        d = torch.full((1,), 0.0, device=device) # d=0 for inference
        
        # Build tokens
        tokens = builder(z, actions_tensor, tau, d)
        
        # Predict clean latents
        # We assume inference starts at offset 0
        pred_z_clean = wm({"wm_input_tokens": tokens, "tau": tau}, time_offset=0)
        
        # Euler update: z_{t+dt} = z_t + (z_clean - z_t) * dt
        v_pred = pred_z_clean - z
        z = z + v_pred * dt
        
    print("Decoding dream to pixels...")
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