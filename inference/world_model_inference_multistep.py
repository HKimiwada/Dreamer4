# Inference script for World Model (for WM trained via train_world_model_v4.py)
# CUDA_VISIBLE_DEVICES=3 python inference/world_model_inference_multistep.py
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm

from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier
from world_model.wm_preprocessing.wm_databuilder import DataBuilderWM
from world_model.wm.dynamics_model import WorldModel
from world_model.wm_preprocessing.wm_dataset import WorldModelDataset, encode_action_components

# --- Configuration ---
class InferenceConfig:
    # Paths
    tokenizer_ckpt = Path("checkpoints/tokenizer/complete_overfit_mse/v1_weights.pt")
    wm_ckpt_path = Path("checkpoints/world_model/train_world_model_v4/best_model.pt")
    output_dir = Path("inference/results/world_model")
    
    # Data Params
    action_path = "data/actions.jsonl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Architecture (Must match training)
    resize = (256, 448)
    patch_size = 16
    input_dim = 3 * 16 * 16
    embed_dim = 512   # Tokenizer dim
    latent_dim = 256  # World Model latent dim
    wm_layers = 12
    wm_heads = 8
    
    # Inference Params
    num_inference_steps = 30  
    clip_length = 600         # Total frames to generate
    chunk_size = 40           # Process frames in small chunks to avoid OOM
    guidance_scale = 1.0 

# --- Helper Functions ---

def load_models(cfg):
    print(f"Loading models on {cfg.device}...")
    
    # 1. Tokenizer
    tokenizer = CausalTokenizer(
        input_dim=cfg.input_dim,
        embed_dim=cfg.embed_dim,
        num_heads=16,
        num_layers=18,
        latent_dim=cfg.latent_dim,
        use_checkpoint=False,
    )
    tok_state = torch.load(cfg.tokenizer_ckpt, map_location="cpu", weights_only=False)
    if "model_state" in tok_state: tok_state = tok_state["model_state"]
    tok_state = {k.replace("module.", ""): v for k, v in tok_state.items()}
    tokenizer.load_state_dict(tok_state)
    tokenizer.to(cfg.device).eval()
    
    # 2. Data Builder & World Model
    data_builder = DataBuilderWM(d_model=512, d_latent=cfg.latent_dim)
    world_model = WorldModel(
        d_model=512,
        d_latent=cfg.latent_dim,
        num_layers=cfg.wm_layers,
        num_heads=cfg.wm_heads,
        n_latents=448,
        clip_length=600, # Max capacity
        use_checkpoint=False 
    )
    
    # Load WM Checkpoint
    if not cfg.wm_ckpt_path.exists():
        raise FileNotFoundError(f"World Model weights not found at {cfg.wm_ckpt_path}")
        
    wm_ckpt = torch.load(cfg.wm_ckpt_path, map_location="cpu", weights_only=False)
    world_model.load_state_dict(wm_ckpt['world_model'])
    data_builder.load_state_dict(wm_ckpt['data_builder'])
    
    world_model.to(cfg.device).eval()
    data_builder.to(cfg.device).eval()
    
    print("✓ All models loaded.")
    return tokenizer, world_model, data_builder

@torch.no_grad()
def decode_latents(tokenizer, latents, size=(256, 448)):
    """Decodes (1, T, N, D) latents into images (T, H, W, 3) numpy array"""
    B, T, N, D = latents.shape
    
    # Chunk decoding to save memory
    chunk_size = 10 
    images_np = []
    patchifier = Patchifier(16)
    
    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        
        # 1. Slice latents
        lat_chunk = latents[:, t_start:t_end]
        
        # 2. Decode chunk
        x = tokenizer.from_latent(lat_chunk)
        x = x.view(B, (t_end - t_start) * N, tokenizer.embed_dim)
        x = tokenizer._run_stack(x, tokenizer.decoder, T=(t_end - t_start), N=N)
        x = x.view(B, (t_end - t_start), N, tokenizer.embed_dim)
        patches = tokenizer.output_proj(x)
        
        # 3. Unpatchify
        for t in range(t_end - t_start):
            frame = patchifier.unpatchify(patches[0, t:t+1], size, 16)[0]
            frame = frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            images_np.append((frame * 255).astype(np.uint8))
            
    return np.array(images_np)

@torch.no_grad()
def generate_video_flow_matching(cfg, world_model, data_builder, tokenizer, actions):
    """
    Generates video using Euler ODE Solver with Chunked Forward Pass to prevent OOM.
    """
    B = 1 
    T = cfg.clip_length
    device = cfg.device
    
    # 1. Initialize random noise (Gaussian)
    N = 448
    z = torch.randn(B, T, N, cfg.latent_dim, device=device)
    
    # 2. Setup Time Grid (0 to 1)
    timesteps = torch.linspace(0, 1, cfg.num_inference_steps + 1, device=device)
    dt = 1.0 / cfg.num_inference_steps
    
    print(f"Generating {T} frames over {cfg.num_inference_steps} steps (Chunk size: {cfg.chunk_size})...")
    
    # 3. ODE Solver Loop (Euler Method)
    for i in tqdm(range(cfg.num_inference_steps)):
        t_curr = timesteps[i]
        
        tau_map = torch.full((B, T), t_curr.item(), device=device)
        d_map   = torch.full((B,), 0.0, device=device) 
        
        # Chucked Forward Pass
        # Instead of passing all 600 frames at once, we pass small chunks
        pred_chunks = []
        
        for t_start in range(0, T, cfg.chunk_size):
            t_end = min(t_start + cfg.chunk_size, T)
            
            # Slice inputs for this chunk
            z_chunk = z[:, t_start:t_end]
            tau_chunk = tau_map[:, t_start:t_end]
            d_chunk = d_map # (B,) doesn't need slicing
            
            # Slice actions
            act_chunk = {k: v[:, t_start:t_end].to(device) for k, v in actions.items()}
            
            # Prepare Tokens for this chunk
            z_proj = data_builder.latent_project(z_chunk)
            a_tokens = data_builder.action_tokenizer(act_chunk)
            
            Sr = data_builder.register_tokens
            reg_base = data_builder.register_embed(torch.arange(Sr, device=device))
            # Expand registers to match chunk length
            current_chunk_len = t_end - t_start
            register_tokens = reg_base.view(1, 1, Sr, -1).expand(B, current_chunk_len, -1, -1)
            
            d_expanded = d_chunk.view(B, 1).expand(B, current_chunk_len)
            feat = torch.stack([tau_chunk, d_expanded], dim=-1)
            shortcut_vec = data_builder.shortcut_mlp(feat) + data_builder.shortcut_slot.view(1, 1, -1)
            shortcut_tokens = shortcut_vec.unsqueeze(2)
            
            wm_tokens = torch.cat([z_proj, a_tokens, register_tokens, shortcut_tokens], dim=2)
            
            L_total = wm_tokens.shape[2]
            wm_input_tokens = wm_tokens.view(B, current_chunk_len * L_total, -1)
            
            wm_input_dict = {
                "wm_input_tokens": wm_input_tokens,
                "tau": tau_chunk
            }
            
            # Run Model on Chunk
            # Pass time_offset=t_start so positional embeddings are correct!
            pred_chunk = world_model(wm_input_dict, time_offset=t_start)
            pred_chunks.append(pred_chunk)
            
        # Concatenate chunks back to full sequence
        pred_z_clean = torch.cat(pred_chunks, dim=1)
        
        # Euler Step
        v_pred = pred_z_clean - z
        z = z + v_pred * dt
        
    print("Decoding latents to pixels...")
    video_frames = decode_latents(tokenizer, z)
    
    return video_frames

def save_video(frames, path, fps=20): 
    H, W, C = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to {path}")

# --- Main ---
def main():
    cfg = InferenceConfig()
    cfg.output_dir.mkdir(exist_ok=True)
    
    # 1. Load Everything
    tokenizer, wm, db = load_models(cfg)
    
    # 2. Load Actions
    print("Loading actions...")
    temp_dataset = WorldModelDataset(
        latent_dir="data/latent_sequences_long", 
        action_jsonl=cfg.action_path,
        clip_length=cfg.clip_length,
        device="cpu"
    )
    
    raw_actions = {}
    
    # Parse actions correctly
    if isinstance(temp_dataset.actions, list):
        print(f"Parsing {len(temp_dataset.actions)} action frames (using first {cfg.clip_length})...")
        
        # Ensure we don't request more actions than available
        available_actions = len(temp_dataset.actions)
        if available_actions < cfg.clip_length:
            print(f"Warning: Requested {cfg.clip_length} frames but only {available_actions} actions found.")
            cfg.clip_length = available_actions
            
        clip_actions = temp_dataset.actions[:cfg.clip_length]
        
        accumulated = {
            "mouse_cat": [], "scroll": [], "yaw_pitch": [], "hotbar": [], 
            "gui": [], "buttons": [], "keys": []
        }
        
        for frame in clip_actions:
            enc = encode_action_components(frame)
            for k in accumulated:
                accumulated[k].append(enc[k])
        
        for k, v in accumulated.items():
            if len(v) > 0:
                raw_actions[k] = torch.stack(v).unsqueeze(0)
    
    if not raw_actions:
        print("Warning: Could not parse actions automatically. Using zero-actions.")
        T = cfg.clip_length
        raw_actions = {
            "mouse_cat": torch.zeros(1, T, dtype=torch.long),
            "scroll":    torch.zeros(1, T, dtype=torch.long),
            "yaw_pitch": torch.zeros(1, T, 2, dtype=torch.float32),
            "hotbar":    torch.zeros(1, T, dtype=torch.long),
            "gui":       torch.zeros(1, T, 2, dtype=torch.float32),
            "buttons":   torch.zeros(1, T, 3, dtype=torch.float32),
            "keys":      torch.zeros(1, T, 23, dtype=torch.float32),
        }

    # 3. Run Inference
    video_frames = generate_video_flow_matching(cfg, wm, db, tokenizer, raw_actions)
    
    # 4. Save
    save_path = cfg.output_dir / "inference_result_600_frames.mp4"
    save_video(video_frames, save_path)

if __name__ == "__main__":
    main()