# Training script v0 (most simple with basic flow-matching)
# python training_script/world_model/short_frames/train_world_model_v0.py
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

# Define constants needed for decoding
PATCH_SIZE = 16
RESIZE = (256, 448)

class TokenizerConfig:
    # ckpt_path = Path("checkpoints/overfit/latest_complete_overfit_mse/best_model.pt")
    ckpt_path = Path("checkpoints/tokenizer/complete_overfit_mse/v1_weights.pt")

    # Model / dataset params (must match training)
    resize = RESIZE
    patch_size = PATCH_SIZE
    clip_length = 8
    input_dim = 3 * patch_size * patch_size
    embed_dim = 512
    latent_dim = 256
    num_heads = 16
    num_layers = 18
    visualize_interval = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

def load_tokenizer(cfg):
    print(f"[Load] Loading checkpoint: {cfg.ckpt_path}")
    model = CausalTokenizer(
        input_dim=cfg.input_dim,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        latent_dim=cfg.latent_dim,
        use_checkpoint=False,
    )

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    state = ckpt["model_state"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    model.to(cfg.device)
    model.eval()
    print("✓ Model loaded")
    return model

@torch.no_grad()
def decode_latents(tokenizer, latents):
    """
    Decode latents back to image using tokenizer decoder.
    latents: (1, N, D_latent) - single frame
    returns: (1, 3, H, W)
    """
    # Project from latent space back to model space
    x = tokenizer.from_latent(latents)  # (1, N, embed_dim)
     
    # Run decoder
    x = tokenizer._run_stack(x, tokenizer.decoder, T=1, N=x.shape[1])
    
    # Project to patch tokens
    patches = tokenizer.output_proj(x)  # (1, N, patch_dim)
    
    # Unpatchify
    patchifier = Patchifier(PATCH_SIZE)
    H, W = RESIZE
    frame = patchifier.unpatchify(patches, (H, W), PATCH_SIZE)[0]  # (3, H, W)
    
    return frame.unsqueeze(0).clamp(0, 1)

@torch.no_grad()
def visualize_world_model(world_model, data_builder, sample, tokenizer, step, device, num_frames=4):
    """
    Simple single-step visualization for debugging world model training.
    
    Shows:
    1. Ground truth (clean) frames
    2. Corrupted frames (what the model sees as input)
    3. Model's prediction (denoised output)
    
    This tests the model's ability to denoise at a FIXED tau level,
    making it easy to track improvement over training.
    
    Args:
        world_model: trained WorldModel instance
        data_builder: DataBuilderWM instance
        sample: dict with 'latents' and 'actions' from dataset
        tokenizer: CausalTokenizer for decoding latents to images
        step: current training step (for logging)
        device: torch device
        num_frames: number of frames to visualize (default 4)
    """
    world_model.eval()
    
    # Extract data from sample
    latents = sample["latents"]  # (T, N, D_latent)
    actions = sample["actions"]  # dict of action tensors
    
    # Add batch dimension if needed
    if latents.dim() == 3:
        latents = latents.unsqueeze(0)  # (1, T, N, D_latent)
        actions = {k: v.unsqueeze(0) for k, v in actions.items()}
    
    latents = latents.to(device)
    actions = {k: v.to(device) for k, v in actions.items()}
    
    B, T, N, D_latent = latents.shape

    # Fixed Tau/D for consistent evaluation
    # Use tau=0.5 (50% signal, 50% noise) for visualization
    # This gives a good balance - not too easy, not impossible
    tau_fixed = torch.full((B, T), 0.5, device=device)
    d_fixed = torch.full((B,), 0.25, device=device)
    
    # Manually create corrupted latents at fixed tau
    # z_corrupted = (1 - tau) * noise + tau * z_clean
    noise = torch.randn_like(latents)
    tau_expanded = tau_fixed.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
    z_corrupted = (1.0 - tau_expanded) * noise + tau_expanded * latents
    
    # Build World Model Input
    # We need to manually construct the input tokens because we want
    # to use our fixed tau/d, not randomly sampled ones
    
    # Project corrupted latents to model dimension
    z_corrupted_proj = data_builder.latent_project(z_corrupted)  # (B, T, N, D_model)
    
    # Get action tokens
    action_tokens = data_builder.action_tokenizer(actions)  # (B, T, Sa, D_model)
    Sa = action_tokens.shape[2]
    
    # Get register tokens
    Sr = data_builder.register_tokens
    register_ids = torch.arange(Sr, device=device)
    reg_base = data_builder.register_embed(register_ids)  # (Sr, D_model)
    reg_base = reg_base.view(1, 1, Sr, data_builder.d_model)
    register_tokens = reg_base.expand(B, T, Sr, data_builder.d_model)
    
    # Build shortcut tokens from fixed (tau, d)
    d_expanded = d_fixed.view(B, 1).expand(B, T)  # (B, T)
    feat = torch.stack([tau_fixed, d_expanded], dim=-1)  # (B, T, 2)
    shortcut_vec = data_builder.shortcut_mlp(feat)  # (B, T, D_model)
    shortcut_vec = shortcut_vec + data_builder.shortcut_slot.view(1, 1, -1)
    shortcut_tokens = shortcut_vec.unsqueeze(2)  # (B, T, 1, D_model)
    
    # Concatenate all tokens
    wm_tokens = torch.cat([
        z_corrupted_proj,  # Corrupted latents
        action_tokens,     # Actions
        register_tokens,   # Registers
        shortcut_tokens    # Shortcut (tau, d)
    ], dim=2)  # (B, T, N+Sa+Sr+1, D_model)
    
    B, T, L_total, D_model = wm_tokens.shape
    wm_input_tokens = wm_tokens.view(B, T * L_total, D_model)
    
    # Create input dict for world model
    wm_input = {
        "wm_input_tokens": wm_input_tokens,
        "tau": tau_fixed,
        "d": d_fixed,
        "z_clean": latents,
        "z_corrupted": z_corrupted,
    }
    
    # Forward pass through world model
    pred_z = world_model(wm_input)  # (B, T, N, D_latent)
    
    # Select frames to visualize
    num_frames = min(num_frames, T)
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)
    
    # Decode and assemble visualization grid
    rows = []
    for t in frame_indices:
        # Ground truth frame
        gt_frame = decode_latents(tokenizer, latents[0, t:t+1])  # (1, 3, H, W)
        
        # Corrupted frame (what model sees)
        corrupted_frame = decode_latents(tokenizer, z_corrupted[0, t:t+1])  # (1, 3, H, W)
        
        # Model's prediction
        pred_frame = decode_latents(tokenizer, pred_z[0, t:t+1])  # (1, 3, H, W)
        
        # Convert to numpy arrays (RGB, 0-255)
        gt_np = (gt_frame[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        corr_np = (corrupted_frame[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pred_np = (pred_frame[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        # Concatenate horizontally: [GT | Corrupted | Prediction]
        row = np.concatenate([gt_np, corr_np, pred_np], axis=1)
        rows.append(row)
    
    # Stack all frames vertically
    final_img = np.concatenate(rows, axis=0)
    
    # Log to wandb
    wandb.log({
        "reconstruction": wandb.Image(
            final_img, 
            caption=f"Step {step} | GT | Corrupted (τ=0.5) | Prediction"
        ),
        "step": step
    })
    
    # Compute and log reconstruction metrics
    with torch.no_grad():
        mse_loss = F.mse_loss(pred_z, latents)
        wandb.log({
            "viz/mse_reconstruction": mse_loss.item(),
            "step": step
        })
    
    world_model.train()

def train_overfit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TokenizerConfig()
    tokenizer = load_tokenizer(cfg)

    # Load Dataset
    dataset = WorldModelDataset(
        latent_dir="data/latent_sequences",
        action_jsonl="data/actions.jsonl",
        clip_length=8,
        device=device
    )

    # This forces the model to memorize exactly 8 frames.
    print(f"Original dataset size: {len(dataset)}")
    dataset.latent_files = dataset.latent_files[:1] 
    dataset.actions = dataset.actions[:8] # Ensure actions match the single clip (8 frames)
    print(f"Overfitting dataset size: {len(dataset)}")
    
    # Disable shuffle so we hammer the same clip repeatedly
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # shuffle=True helps overfitting faster
    # loader = DataLoader(dataset, batch_size=1, shuffle=True)

    d_model = 512
    d_latent = 256
    data_builder = DataBuilderWM(d_model, d_latent=d_latent).to(device)
    world_model = WorldModel(
        d_model=d_model,
        d_latent=d_latent,
        num_layers=16,
        num_heads=8
    ).to(device)

    # Combine parameters for optimization
    params = list(world_model.parameters()) + list(data_builder.parameters())
    
    optimizer = torch.optim.AdamW(
        params,
        lr=1e-4,  # Lower learning rate for stability
        weight_decay=0.01  # Add weight decay for regularization
    )
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=10000, 
        eta_min=1e-6
    )

    wandb.init(
        project="worldmodel-v0-fixed",
        config={
            "d_model": d_model,
            "d_latent": d_latent,
            "num_layers": 16,
            "num_heads": 8,
            "lr": 1e-4,
            "loss_type": "x-prediction"
        }
    )

    world_model.train()
    data_builder.train()

    for step in range(10000):
        for sample in loader:
            latents = sample["latents"]       
            actions = sample["actions"]

            # Build world-model input
            wm_input = data_builder(latents, actions)

            # Forward pass - model predicts CLEAN latents
            pred_z_clean = world_model(wm_input)

            # X-prediction loss: compare prediction to ground truth clean latents
            loss = flow_loss_v2(
                pred_z_clean,
                wm_input["z_clean"],  # Ground truth clean latents
                wm_input["tau"],       # Signal levels
                ramp_weight=True       # Weight by tau
            )

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # Prevent exploding gradients
            optimizer.step()
            scheduler.step()
            
        # Logging
        if step % 10 == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "step": step
            })
            print(f"[step {step}] loss = {loss.item():.6f}")
        
        # Visualization
        if step % cfg.visualize_interval == 0:
            visualize_world_model(
                world_model, data_builder, sample, tokenizer, step, device
            )

    print("Training complete!")

if __name__ == "__main__":
    train_overfit()