# Training script v1 (v0 + but w/batch decoding for visualization)
# python training_script/world_model/short_frames/train_world_model_v1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier
from world_model.wm_preprocessing.wm_dataset import WorldModelDataset
from world_model.wm_preprocessing.wm_databuilder import DataBuilderWM
from world_model.wm.dynamics_model import WorldModel
from world_model.wm.loss import flow_loss_v2

# Constants
PATCH_SIZE = 16
RESIZE = (256, 448)

class TokenizerConfig:
    # Path to your best tokenizer checkpoint
    ckpt_path = Path("checkpoints/tokenizer/complete_overfit_mse/v1_weights.pt")

    # Model / dataset params (must match tokenizer training)
    resize = RESIZE
    patch_size = PATCH_SIZE
    clip_length = 8
    input_dim = 3 * patch_size * patch_size
    embed_dim = 512
    latent_dim = 256
    num_heads = 16
    num_layers = 18
    visualize_interval = 20  # Frequency of visualization
    device = "cuda" if torch.cuda.is_available() else "cpu"

def load_tokenizer(cfg):
    print(f"[Load] Loading tokenizer checkpoint: {cfg.ckpt_path}")
    model = CausalTokenizer(
        input_dim=cfg.input_dim,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        latent_dim=cfg.latent_dim,
        use_checkpoint=False,
    )

    if not cfg.ckpt_path.exists():
        print(f"⚠️ Warning: Checkpoint {cfg.ckpt_path} not found. Visualization will be random.")
        model.to(cfg.device)
        return model

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    # Handle DDP state dict keys if present
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()
    
    # Freeze tokenizer completely
    for p in model.parameters():
        p.requires_grad = False
        
    print("✓ Tokenizer loaded and frozen")
    return model

@torch.no_grad()
def decode_latents_batch(tokenizer, latents):
    """
    Decode a full batch of latents to images, preserving temporal context.
    
    Args:
        latents: (B, T, N, D_latent)
    Returns:
        images: (B, T, 3, H, W) in range [0, 1]
    """
    B, T, N, D = latents.shape
    
    # 1. Project from latent space back to model embedding space
    x = tokenizer.from_latent(latents)  # (B, T, N, embed_dim)

    # 2. Flatten for transformer: (B, T*N, embed_dim)
    x = x.view(B, T * N, tokenizer.embed_dim)

    # 3. Run decoder with FULL T to preserve temporal attention
    #    The tokenizer was trained to look at past frames; decoding T=1 breaks this.
    x = tokenizer._run_stack(x, tokenizer.decoder, T=T, N=N)

    # 4. Project to patch pixels
    x = x.view(B, T, N, tokenizer.embed_dim)
    patches = tokenizer.output_proj(x)  # (B, T, N, patch_dim)

    # 5. Unpatchify batch
    patchifier = Patchifier(PATCH_SIZE) 
    
    # Reconstruct images for each item in batch
    images = []
    for b in range(B):
        batch_imgs = []
        for t in range(T):
            # Unpatchify single frame
            frame = patchifier.unpatchify(
                patches[b, t:t+1], (RESIZE[0], RESIZE[1]), PATCH_SIZE
            )[0]
            batch_imgs.append(frame.clamp(0, 1))
        images.append(torch.stack(batch_imgs))
        
    return torch.stack(images) # (B, T, 3, H, W)

@torch.no_grad()
def visualize_world_model(world_model, data_builder, sample, tokenizer, step, device, num_frames=8):
    """
    Visualize Ground Truth vs Corrupted Input vs World Model Prediction.
    Uses batch decoding to ensure GT looks correct.
    """
    world_model.eval()
    
    # Extract data
    latents = sample["latents"].to(device)  # (B, T, N, D)
    actions = {k: v.to(device) for k, v in sample["actions"].items()}
    
    B, T, N, D_latent = latents.shape

    # 1. Create Input at Fixed Noise Level (tau=0.5) for consistent eval
    tau_fixed = torch.full((B, T), 0.5, device=device)
    d_fixed = torch.full((B,), 0.25, device=device)
    
    noise = torch.randn_like(latents)
    tau_expanded = tau_fixed.unsqueeze(-1).unsqueeze(-1)
    z_corrupted = (1.0 - tau_expanded) * noise + tau_expanded * latents
    
    # 2. Build WM Tokens manually (to force fixed tau)
    z_corrupted_proj = data_builder.latent_project(z_corrupted)
    action_tokens = data_builder.action_tokenizer(actions)
    
    Sr = data_builder.register_tokens
    register_ids = torch.arange(Sr, device=device)
    reg_base = data_builder.register_embed(register_ids).view(1, 1, Sr, -1)
    register_tokens = reg_base.expand(B, T, Sr, -1)
    
    # Shortcut tokens
    d_expanded = d_fixed.view(B, 1).expand(B, T)
    feat = torch.stack([tau_fixed, d_expanded], dim=-1)
    shortcut_vec = data_builder.shortcut_mlp(feat) + data_builder.shortcut_slot.view(1, 1, -1)
    shortcut_tokens = shortcut_vec.unsqueeze(2)
    
    wm_tokens = torch.cat([z_corrupted_proj, action_tokens, register_tokens, shortcut_tokens], dim=2)
    B, T, L_total, Dm = wm_tokens.shape
    wm_input_tokens = wm_tokens.view(B, T * L_total, Dm)
    
    wm_input = {
        "wm_input_tokens": wm_input_tokens,
        "tau": tau_fixed,
        "d": d_fixed,
        "z_clean": latents,
        "z_corrupted": z_corrupted,
    }
    
    # 3. Predict Clean Latents
    pred_z = world_model(wm_input)
    
    # 4. DECODE ALL VIDEO SEQUENCES BATCHWISE
    #    This fixes the blurry GT issue by using T=8 context
    gt_video   = decode_latents_batch(tokenizer, latents)     # (B, T, 3, H, W)
    corr_video = decode_latents_batch(tokenizer, z_corrupted) # (B, T, 3, H, W)
    pred_video = decode_latents_batch(tokenizer, pred_z)      # (B, T, 3, H, W)
    
    # 5. Construct Visualization Grid
    # Visualize up to num_frames from the first sequence in batch
    frame_indices = np.linspace(0, T - 1, min(num_frames, T), dtype=int)
    
    rows = []
    for t in frame_indices:
        # Get frame t from batch 0
        gt_frame   = gt_video[0, t]
        corr_frame = corr_video[0, t]
        pred_frame = pred_video[0, t]
        
        # Convert to uint8 numpy
        gt_np   = (gt_frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        corr_np = (corr_frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pred_np = (pred_frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Concatenate: GT | Corrupted | Predicted
        row = np.concatenate([gt_np, corr_np, pred_np], axis=1)
        rows.append(row)
    
    final_img = np.concatenate(rows, axis=0)
    
    wandb.log({
        "reconstruction": wandb.Image(final_img, caption=f"Step {step} (τ=0.5)"),
        "step": step
    })
    
    world_model.train()

def train_overfit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Tokenizer (Frozen)
    cfg = TokenizerConfig()
    tokenizer = load_tokenizer(cfg)

    # Load Dataset
    dataset = WorldModelDataset(
        latent_dir="data/latent_sequences",
        action_jsonl="data/actions.jsonl",
        clip_length=8,
        device=device
    )

    # --- Overfitting Setup ---
    # Memorize the first clip (8 frames) to verify model capacity
    print(f"Original dataset size: {len(dataset)}")
    dataset.latent_files = dataset.latent_files[:1] 
    dataset.actions = dataset.actions[:8] 
    print(f"Overfitting dataset size: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Init Models
    d_model = 512
    d_latent = 256
    
    data_builder = DataBuilderWM(d_model, d_latent=d_latent).to(device)
    
    # A smaller World Model for faster overfitting verification
    world_model = WorldModel(
        d_model=d_model,
        d_latent=d_latent,
        num_layers=12,
        num_heads=8,
        n_latents=448 # (16x16 patches for 256x448 image) -> 16*28 = 448
    ).to(device)

    params = list(world_model.parameters()) + list(data_builder.parameters())
    
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6)

    wandb.init(
        project="worldmodel-v1-overfit",
        config={
            "d_model": d_model,
            "d_latent": d_latent,
            "layers": 12,
            "heads": 8,
            "mode": "overfit",
            "loss": "flow_match_x_pred"
        }
    )

    print("Starting training...")
    world_model.train()
    data_builder.train()

    global_step = 0
    MAX_STEPS = 5000

    while global_step < MAX_STEPS:
        for sample in loader:
            latents = sample["latents"]       
            actions = sample["actions"]

            # 1. Build input tokens (adds noise based on random tau)
            wm_input = data_builder(latents, actions)

            # 2. Forward pass: Predict Clean Latents
            pred_z_clean = world_model(wm_input)

            # 3. Loss: X-Prediction Flow Matching
            loss = flow_loss_v2(
                pred_z_clean,
                wm_input["z_clean"],
                wm_input["tau"],
                ramp_weight=True
            )

            # 4. Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            global_step += 1

            # Log
            if global_step % 10 == 0:
                wandb.log({
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "step": global_step
                })
                print(f"[Step {global_step}] Loss: {loss.item():.6f}")
            
            # Visualize
            if global_step % cfg.visualize_interval == 0:
                print(f"Visualizing at step {global_step}...")
                visualize_world_model(
                    world_model, data_builder, sample, tokenizer, global_step, device
                )

            if global_step >= MAX_STEPS:
                break

    print("Training complete!")
    wandb.finish()

if __name__ == "__main__":
    train_overfit()