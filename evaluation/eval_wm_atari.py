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
from training_script.world_model.atari.latest_train_world_model_atari import AtariDataBuilder, AtariWorldModelDataset

# --- Configuration (Aligned with Atari v4 Training) ---
class EvalConfig:
    # Paths
    tokenizer_ckpt = Path("checkpoints/atari/tokenizer_v3/best_model.pt")
    wm_ckpt_path = Path("checkpoints/world_model/atari_v4/best_wm.pt")
    latent_path = Path("data/atari/latent_sequences/video_full_frames.pt")
    actions_jsonl = Path("data/atari/raw/actions.jsonl")
    output_dir = Path("evaluation/results/atari_v4")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Architecture
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
    
    # Eval Params
    num_ode_steps = 50     
    clip_length = 64       # Context window
    stride = 64
    batch_size = 1         

def load_models(cfg):
    print(f"[*] Loading models on {cfg.device}...")
    
    # 1. Tokenizer
    tokenizer = CausalTokenizer(
        input_dim=3 * cfg.patch_size * cfg.patch_size,
        embed_dim=cfg.tokenizer_dim, 
        num_heads=8, num_layers=8, latent_dim=cfg.latent_dim
    )
    tok_state = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
    tokenizer.load_state_dict({k.replace("module.", ""): v for k, v in tok_state["model_state"].items()})
    tokenizer.to(cfg.device).eval()
    
    # 2. World Model & Builder
    world_model = WorldModel(
        d_model=cfg.embed_dim, d_latent=cfg.latent_dim, num_layers=cfg.num_layers,
        num_heads=8, n_latents=cfg.n_latents, Sr=cfg.Sr
    )
    wm_state = torch.load(cfg.wm_ckpt_path, map_location="cpu")
    world_model.load_state_dict({k.replace("module.", ""): v for k, v in wm_state.items()})
    world_model.to(cfg.device).eval()
    
    builder = AtariDataBuilder(cfg).to(cfg.device).eval()
    
    return tokenizer, world_model, builder

@torch.no_grad()
def decode_sequence(tokenizer, latents, cfg):
    """Decodes latents with the 'Perfect' diagnostic logic (Local Context Fix)."""
    # Clamp to prevent graininess from out-of-distribution predictions
    latents = torch.clamp(latents, -5.0, 5.0) 
    B, T, N, D = latents.shape
    patchifier = Patchifier(cfg.patch_size)
    
    x = tokenizer.from_latent(latents) 
    x = x.view(B, T * N, tokenizer.embed_dim)
    # CRITICAL: Apply local positional context
    x = x + tokenizer.pos_embed[:, :T * N, :]
    x = tokenizer._run_stack(x, tokenizer.decoder, T=T, N=N)
    x = x.view(B, T, N, tokenizer.embed_dim)
    patches = tokenizer.output_proj(x)
    
    full_frames = patchifier.unpatchify(patches.squeeze(0), cfg.resize, cfg.patch_size)
    return (full_frames.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

@torch.enable_grad()
def identify_action_bottleneck(cfg, wm, builder, tokenizer, sample):
    """
    Diagnostic tool adjusted for the AtariWorldModel architecture.
    """
    device = cfg.device
    wm.train() 
    builder.train()

    # 1. Setup Data
    latents = sample["latents"].unsqueeze(0).to(device) # (1, T, N, D)
    actions = sample["actions"].unsqueeze(0).to(device) # (1, T)
    T = latents.shape[1]
    tau = torch.full((1, T), 0.5, device=device)
    d = torch.full((1,), 0.5, device=device)
    
    b_module = builder.module if hasattr(builder, "module") else builder
    
    # --- FIX: RETAIN GRAD ON NON-LEAF TENSOR ---
    action_embeddings = b_module.action_embed(actions) # (1, T, D_model)
    action_embeddings.retain_grad() # Force PyTorch to keep gradients for this tensor
    # -------------------------------------------

    # 2. Check Signal Magnitudes
    z_proj = b_module.latent_proj(latents) 
    z_norm = torch.norm(z_proj).item()
    a_norm = torch.norm(action_embeddings).item()
    
    print(f"\n--- Signal Magnitude Check ---")
    print(f"Latent Signal Norm: {z_norm:.4f}")
    print(f"Action Signal Norm: {a_norm:.4f}")
    print(f"Ratio (Z/A): {z_norm/a_norm:.2f}x")

    # 3. Manual Token Construction
    E = cfg.embed_dim
    Sa = 1 
    Sr = cfg.Sr
    
    a_tokens = action_embeddings.unsqueeze(2)
    reg_ids = torch.arange(Sr, device=device)
    reg_tokens = b_module.register_embed(reg_ids).view(1, 1, Sr, E).expand(1, T, -1, -1)
    
    feat = torch.stack([tau, d.view(1, 1).expand(1, T)], dim=-1)
    s_tokens = (b_module.shortcut_mlp(feat) + b_module.shortcut_slot).unsqueeze(2)
    
    tokens = torch.cat([z_proj, a_tokens, reg_tokens, s_tokens], dim=2) 
    tokens_flat = tokens.view(1, -1, E)
    
    # 4. Sensitivity Calculation
    wm_input = {"wm_input_tokens": tokens_flat, "tau": tau, "d": d}
    pred_z = wm(wm_input, time_offsets=torch.zeros(1, device=device, dtype=torch.long))
    
    # Summing output to compute scalar loss for backward pass
    loss = pred_z.sum()
    loss.backward()
    
    # Now action_embeddings.grad will be populated
    grad_norm = torch.norm(action_embeddings.grad).item()
    print(f"\n--- Gradient Sensitivity Check ---")
    print(f"Action Gradient Norm: {grad_norm:.8e}")
    
    if grad_norm < 1e-12:
        print("RESULT: ACTION DISCONNECT. The model output is mathematically independent of actions.")
    elif grad_norm < 1e-7:
        print("RESULT: WEAK SENSITIVITY. The model barely notices actions.")
    else:
        print("RESULT: SIGNAL PRESENT. The model sees actions; failure to react is likely Mode Collapse.")

@torch.no_grad()
def run_evaluation():
    cfg = EvalConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, wm, builder = load_models(cfg)
    
    # Load Real Data for Comparison
    dataset = AtariWorldModelDataset(cfg)
    sample = dataset[0]
    gt_latents = sample["latents"].unsqueeze(0).to(cfg.device)   # (1, T, N, D)
    actions = sample["actions"].unsqueeze(0).to(cfg.device)      # (1, T)
    T = gt_latents.shape[1]

    # --- TEST 1: ONE-STEP DENOISING (WandB Quality Check) ---
    print("[1/3] Testing One-Step Denoising (Tau=0.5)...")
    tau_fixed = torch.full((1, T), 0.5, device=cfg.device)
    d_fixed = torch.full((1,), 0.5, device=cfg.device)
    noise = torch.randn_like(gt_latents)
    z_corr = (0.5 * noise) + (0.5 * gt_latents)
    
    tokens = builder(z_corr, actions, tau_fixed, d_fixed)
    pred_denoise = wm({"wm_input_tokens": tokens, "tau": tau_fixed, "d": d_fixed}, 
                      time_offsets=torch.zeros(1, device=cfg.device, dtype=torch.long))
    
    # --- TEST 2: MULTI-STEP DREAMING (Imagination Check) ---
    print("[2/3] Testing 64-frame Imagination...")
    z_dream = torch.randn(1, T, cfg.n_latents, cfg.latent_dim, device=cfg.device)
    z_dream[:, 0] = gt_latents[:, 0] # Seed with first real frame
    
    timesteps = torch.linspace(0, 1, cfg.num_ode_steps + 1, device=cfg.device)
    dt = 1.0 / cfg.num_ode_steps
    
    for i in tqdm(range(cfg.num_ode_steps)):
        t_curr = timesteps[i]
        tau = torch.full((1, T), t_curr.item(), device=cfg.device)
        d = torch.full((1,), 0.5, device=cfg.device)
        
        tokens = builder(z_dream, actions, tau, d)
        pred_clean = wm({"wm_input_tokens": tokens, "tau": tau, "d": d}, 
                        time_offsets=torch.zeros(1, device=cfg.device, dtype=torch.long))
        
        # ODE Update: Scaling by (1-t) to ensure clean manifold convergence
        eps = 1e-5
        v_pred = (torch.clamp(pred_clean, -5, 5) - z_dream) / (1.0 - t_curr + eps)
        z_dream[:, 1:] = z_dream[:, 1:] + (v_pred[:, 1:] * dt)

    # --- TEST 3: ANALYSIS & VIDEO EXPORT ---
    print("[3/3] Exporting Comparison...")
    frames_gt = decode_sequence(tokenizer, gt_latents, cfg)
    frames_dn = decode_sequence(tokenizer, pred_denoise, cfg)
    frames_dr = decode_sequence(tokenizer, z_dream, cfg) # Using final path z for 'dream' feel
    
    save_path = cfg.output_dir / "eval_comparison.mp4"
    H, W = cfg.resize
    # Vertical stack: GT (top), Denoise (middle), Dream (bottom)
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (W, H * 3))
    
    mse_dn = torch.nn.functional.mse_loss(pred_denoise, gt_latents).item()
    mse_dr = torch.nn.functional.mse_loss(z_dream, gt_latents).item()
    
    for t in range(T):
        # Concatenate frames vertically
        combined = np.vstack([frames_gt[t], frames_dn[t], frames_dr[t]])
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    out.release()
    
    print(f"\n--- Evaluation Results ---")
    print(f"Denoising MSE: {mse_dn:.6f} (Lower is better)")
    print(f"Imagination MSE: {mse_dr:.6f} (Stability indicator)")
    print(f"Latent Max/Min: {z_dream.max().item():.2f} / {z_dream.min().item():.2f} (Should be ~ +/- 5)")
    print(f"✓ Results saved to {save_path}")

    print("\n" + "="*40)
    print("RUNNING ACTION SENSITIVITY DIAGNOSTIC")
    print("="*40)
    
    # We pass the first sample from the dataset to the debugger
    # identify_action_bottleneck is run with gradients enabled internally
    identify_action_bottleneck(cfg, wm, builder, tokenizer, sample)

if __name__ == "__main__":
    run_evaluation()
    