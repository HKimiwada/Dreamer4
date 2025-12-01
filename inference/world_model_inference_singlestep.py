# CUDA_VISIBLE_DEVICES=3 python inference/world_model_inference_singlestep.py
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
    tokenizer_ckpt = Path("checkpoints/tokenizer/complete_overfit_mse/v1_weights.pt")
    wm_ckpt_path = Path("checkpoints/world_model/train_world_model_v4/best_model.pt")
    output_dir = Path("inference/results/world_model")
    action_path = "data/actions.jsonl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resize = (256, 448)
    patch_size = 16
    input_dim = 3 * 16 * 16
    embed_dim = 512 
    latent_dim = 256
    wm_layers = 12
    wm_heads = 8
    
    # 600 frames total
    clip_length = 600
    chunk_size = 40 

# --- Helper Functions ---
def load_models(cfg):
    print(f"Loading models on {cfg.device}...")
    tokenizer = CausalTokenizer(
        input_dim=cfg.input_dim, embed_dim=cfg.embed_dim, num_heads=16, num_layers=18, latent_dim=cfg.latent_dim, use_checkpoint=False
    )
    tok_state = torch.load(cfg.tokenizer_ckpt, map_location="cpu", weights_only=False)
    if "model_state" in tok_state: tok_state = tok_state["model_state"]
    tok_state = {k.replace("module.", ""): v for k, v in tok_state.items()}
    tokenizer.load_state_dict(tok_state)
    tokenizer.to(cfg.device).eval()
    
    data_builder = DataBuilderWM(d_model=512, d_latent=cfg.latent_dim)
    world_model = WorldModel(
        d_model=512, d_latent=cfg.latent_dim, num_layers=cfg.wm_layers, num_heads=cfg.wm_heads, n_latents=448, clip_length=600, use_checkpoint=False 
    )
    wm_ckpt = torch.load(cfg.wm_ckpt_path, map_location="cpu", weights_only=False)
    world_model.load_state_dict(wm_ckpt['world_model'])
    data_builder.load_state_dict(wm_ckpt['data_builder'])
    world_model.to(cfg.device).eval()
    data_builder.to(cfg.device).eval()
    return tokenizer, world_model, data_builder

@torch.no_grad()
def decode_latents(tokenizer, latents, size=(256, 448)):
    B, T, N, D = latents.shape
    chunk_size = 10 
    images_np = []
    patchifier = Patchifier(16)
    
    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        lat_chunk = latents[:, t_start:t_end]
        x = tokenizer.from_latent(lat_chunk)
        x = x.view(B, (t_end - t_start) * N, tokenizer.embed_dim)
        x = tokenizer._run_stack(x, tokenizer.decoder, T=(t_end - t_start), N=N)
        x = x.view(B, (t_end - t_start), N, tokenizer.embed_dim)
        patches = tokenizer.output_proj(x)
        for t in range(t_end - t_start):
            frame = patchifier.unpatchify(patches[0, t:t+1], size, 16)[0]
            frame = frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            images_np.append((frame * 255).astype(np.uint8))
    return np.array(images_np)

@torch.no_grad()
def generate_one_step_denoise(cfg, world_model, data_builder, tokenizer, actions):
    """
    ONE-STEP PREDICTION (Like WandB).
    Takes ground truth, adds 50% noise, and asks model to remove it in ONE step.
    Does NOT use ODE loop.
    """
    B = 1 
    T = cfg.clip_length
    device = cfg.device
    
    print(f"\n--- ONE-STEP DENOISING MODE (Like WandB) ---")
    
    # 1. Load Ground Truth
    print("Loading ground truth latents...")
    temp_dataset = WorldModelDataset(
        latent_dir="data/latent_sequences_long", 
        action_jsonl=cfg.action_path,
        clip_length=cfg.clip_length,
        device="cpu"
    )
    z_clean = temp_dataset[0]["latents"].unsqueeze(0).to(device)
    if z_clean.shape[1] > T: z_clean = z_clean[:, :T]
    
    # 2. Corrupt to tau=0.5
    tau_val = 0
    print(f"Corrupting inputs to tau={tau_val}...")
    noise = torch.randn_like(z_clean)
    z_corrupted = (1.0 - tau_val) * noise + tau_val * z_clean
    
    tau_map = torch.full((B, T), tau_val, device=device)
    d_map   = torch.full((B,), 0.0, device=device) 

    # 3. Predict Clean Latents (Chunked to save memory)
    pred_chunks = []
    
    print("Running one-step prediction in chunks...")
    for t_start in tqdm(range(0, T, cfg.chunk_size)):
        t_end = min(t_start + cfg.chunk_size, T)
        current_chunk_len = t_end - t_start
        
        # Slice inputs
        z_chunk = z_corrupted[:, t_start:t_end]
        tau_chunk = tau_map[:, t_start:t_end]
        act_chunk = {k: v[:, t_start:t_end].to(device) for k, v in actions.items()}
        
        # Build Tokens
        z_proj = data_builder.latent_project(z_chunk)
        a_tokens = data_builder.action_tokenizer(act_chunk)
        
        Sr = data_builder.register_tokens
        reg_base = data_builder.register_embed(torch.arange(Sr, device=device))
        register_tokens = reg_base.view(1, 1, Sr, -1).expand(B, current_chunk_len, -1, -1)
        
        d_chunk = d_map
        d_expanded = d_chunk.view(B, 1).expand(B, current_chunk_len)
        feat = torch.stack([tau_chunk, d_expanded], dim=-1)
        shortcut_vec = data_builder.shortcut_mlp(feat) + data_builder.shortcut_slot.view(1, 1, -1)
        shortcut_tokens = shortcut_vec.unsqueeze(2)
        
        wm_tokens = torch.cat([z_proj, a_tokens, register_tokens, shortcut_tokens], dim=2)
        L_total = wm_tokens.shape[2]
        wm_input_tokens = wm_tokens.view(B, current_chunk_len * L_total, -1)
        
        wm_input_dict = {"wm_input_tokens": wm_input_tokens, "tau": tau_chunk}
        
        # PREDICT (No loop!)
        # We ask the model: "What is z_clean?"
        pred_chunk = world_model(wm_input_dict, time_offset=t_start)
        pred_chunks.append(pred_chunk)
        
    # Reassemble
    z_final = torch.cat(pred_chunks, dim=1)
        
    print("Decoding result...")
    video_frames = decode_latents(tokenizer, z_final)
    
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

def main():
    cfg = InferenceConfig()
    cfg.output_dir.mkdir(exist_ok=True)
    tokenizer, wm, db = load_models(cfg)
    
    # Load Actions
    temp_dataset = WorldModelDataset(latent_dir="data/latent_sequences_long", action_jsonl=cfg.action_path, clip_length=cfg.clip_length, device="cpu")
    raw_actions = {}
    if isinstance(temp_dataset.actions, list):
        if len(temp_dataset.actions) < cfg.clip_length: cfg.clip_length = len(temp_dataset.actions)
        clip_actions = temp_dataset.actions[:cfg.clip_length]
        accumulated = {k: [] for k in ["mouse_cat", "scroll", "yaw_pitch", "hotbar", "gui", "buttons", "keys"]}
        for frame in clip_actions:
            enc = encode_action_components(frame)
            for k in accumulated: accumulated[k].append(enc[k])
        for k, v in accumulated.items():
            if len(v) > 0: raw_actions[k] = torch.stack(v).unsqueeze(0)

    # RUN ONE-STEP TEST
    video_frames = generate_one_step_denoise(cfg, wm, db, tokenizer, raw_actions)
    save_video(video_frames, cfg.output_dir / "tau_zero_result.mp4")

if __name__ == "__main__":
    main()