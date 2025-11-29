# Training script v0 (most simple with basic flow-matching)
# python training_script/world_model/train_world_model_v0.py
import torch
import torch.nn as nn
import wandb
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier
from world_model.wm_preprocessing.wm_dataset import WorldModelDataset
from world_model.wm_preprocessing.wm_databuilder import DataBuilderWM
from world_model.wm.dynamics_model import WorldModel
from world_model.wm.loss import flow_loss_v1   

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
    latents: (1, N_latents, D_latent) - representing ONE frame's worth of tokens
    returns (1, 3, H, W)
    """
    # 1. Project latents back to embed_dim
    x = tokenizer.from_latent(latents)     # (1, N, embed_dim)

    # 2. Add pos embedding
    seq_len = x.shape[1]
    # Ensure we slice pos_embed correctly (1, seq_len, D)
    x = x + tokenizer.pos_embed[:, :seq_len, :]

    # 3. Flatten for decoder
    x = x.view(1, seq_len, tokenizer.embed_dim)

    # 4. Run decoder
    x = tokenizer._run_stack(x, tokenizer.decoder, T=1, N=seq_len)

    # 5. Unproject to patch tokens
    patches = tokenizer.output_proj(x).view(1, seq_len, -1)

    # 6. Unpatchify → image
    patchifier = Patchifier(PATCH_SIZE)
    H, W = RESIZE
    frame = patchifier.unpatchify(patches, (H, W), PATCH_SIZE)[0]  # (3,H,W)

    return frame.unsqueeze(0)

@torch.no_grad()
def visualize_world_model(world_model, data_builder, sample, tokenizer, step, device, num_frames=4):
    world_model.eval()

    # 1. Prepare Ground Truth
    latents = sample["latents"]      # (B, T, N, D)
    actions = sample["actions"]
    
    B, T_seq, N_latents, D_latent = latents.shape

    # 2. Initialize Autoregressive Canvas
    # Start with pure noise for the entire sequence. We will fill this in frame-by-frame.
    z_current = torch.randn_like(latents)
    
    # Initialize metadata: 
    # tau=0 (Generation) for future frames
    # tau=1 (Context) for past frames (updated in loop)
    tau_map = torch.zeros(B, T_seq, device=device)
    d_map = torch.ones(B, T_seq, device=device) # Predict full step (d=1)

    pred_frames = []
    
    # 3. Autoregressive Rollout Loop
    # We generate Frame t using the history of Frames 0...t-1
    for t in range(T_seq):
        
        # --- A. Construct Inputs for Step t ---
        # Get base tokens (Actions, Registers) from data_builder
        # (We use the structure it creates, then overwrite the dynamic parts)
        wm_inp = data_builder(latents, actions)
        
        # Reshape flat tokens to (B, T, L_total, D) for editing
        wm_tokens = wm_inp["wm_input_tokens"]
        L_total = wm_tokens.shape[1] // T_seq
        wm_tokens = wm_tokens.view(B, T_seq, L_total, -1)
        
        # OVERWRITE 1: Latents (first N tokens)
        # We replace the ground truth with our current canvas (Mixed Noise + Predictions)
        z_current_proj = data_builder.latent_project(z_current)
        wm_tokens[:, :, :N_latents, :] = z_current_proj
        
        # OVERWRITE 2: Shortcut Token (last 1 token)
        # We manually build the token to tell the model: "Past is clean (tau=1), Current is noise (tau=0)"
        feat = torch.stack([tau_map, d_map], dim=-1) # (B, T, 2)
        shortcut_embed = data_builder.shortcut_mlp(feat) + data_builder.shortcut_slot.view(1, 1, -1)
        wm_tokens[:, :, -1:, :] = shortcut_embed.unsqueeze(2)
        
        # Flatten back for Transformer
        wm_inp["wm_input_tokens"] = wm_tokens.view(B, T_seq * L_total, -1)
        # Update metadata (optional, purely for consistency)
        wm_inp["tau"] = tau_map 

        # --- B. Forward Pass ---
        # Predict all frames (Transformer handles causal masking)
        # We only care about the prediction for the current frame 't'
        preds = world_model(wm_inp) # (B, T, N, D)
        pred_t = preds[:, t]        # (B, N, D)
        
        pred_frames.append(pred_t)
        
        # --- C. Update State for Next Step ---
        if t < T_seq - 1:
            # Update Canvas: Feed this prediction back as "History" for the next step
            z_current[:, t] = pred_t
            # Update Conditioning: Mark this frame as "Context" (tau=1) so model attends to it
            tau_map[:, t] = 1.0 

    # 4. Decode and Visualization
    # Stack predictions along time dimension
    pred_z = torch.stack(pred_frames, dim=1) # (B, T, N, D)
    z_clean = latents

    # Remove batch dim if B=1 for easier slicing
    if B == 1:
        pred_z = pred_z[0]
        z_clean = z_clean[0]

    # Select frames to visualize
    idxs = np.linspace(0, T_seq - 1, num_frames, dtype=int)
    rows = []
    
    for t_idx in idxs:
        # Decode Ground Truth
        gt_img = decode_latents(tokenizer, z_clean[t_idx:t_idx+1])
        # Decode Prediction
        pred_img = decode_latents(tokenizer, pred_z[t_idx:t_idx+1])

        # Convert to Numpy Image
        gt_np = (gt_img[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        pred_np = (pred_img[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

        # Concatenate: [GT | Prediction]
        rows.append(np.concatenate([gt_np, pred_np], axis=1))

    final_img = np.concatenate(rows, axis=0)
    
    wandb.log({"wm_reconstruction": wandb.Image(final_img, caption=f"Step {step} (AR)")})
    
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
    data_builder = DataBuilderWM(d_model).to(device)
    world_model = WorldModel(
        d_model=d_model,
        d_latent=d_latent,
        num_layers=16,
        num_heads=8
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(world_model.parameters()) + list(data_builder.parameters()), 
        lr=1e-4,
    )

    wandb.init(
        project="worldmodel-v0",
        config={
            "d_model": d_model,
            "d_latent": d_latent,
            "num_layers": 16,
            "num_heads": 8,
            "lr": 1e-3
        }
    )

    # Training Loop
    print("world_model: training start")
    world_model.train()
    print("data_builder: training start")
    data_builder.train()

    for step in range(10000): # Increased steps to ensure we see visualization
        for sample in loader:

            latents = sample["latents"]       
            actions = sample["actions"]

            # Build world-model input
            wm_input = data_builder(latents, actions)

            # Forward pass
            pred_z = world_model(wm_input)

            # Flow-matching loss (MSE)
            loss = flow_loss_v1(
                pred_z,       
                wm_input["z_clean"],
                wm_input["tau"]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if step % cfg.visualize_interval == 0:
            print(f"[step {step}] loss = {loss.item():.6f}")
            print("Testing visualization...")
            visualize_world_model(world_model, data_builder, sample, tokenizer, step=step, device=device)
        
        wandb.log({"flow_loss": loss.item(), "step": step})

    print("Completed Training")

if __name__ == "__main__":
    train_overfit()