"""
wandb login
PYTHONPATH=. python training_script/train_tokenizer.py
Overview of Training script for tokenizer:
    1. Load video patch data from your preprocessed dataset (TokenizerDataset).
    2. Feed it into the tokenizer (encoder–decoder).
    3. Compute the reconstruction loss (MSE + 0.2×LPIPS) only on masked patches.
    4. Backpropagate, optimize, and log training metrics to wandb.

Goal:
A trained tokenizer whose encoder + tanh bottleneck produce stable, 
compact latents for the world model.
"""
import os
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

from tokenizer.tokenizer_dataset import TokenizerDataset
from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.losses import CombinedLoss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TrainConfig:
    # Data
    data_dir = Path("data")
    resize = (384, 640)
    patch_size = 16
    clip_length = 64
    batch_size = 2  # dataset yields one clip, treat as batch=1
    num_workers = 2

    # Model
    input_dim = 3 * patch_size * patch_size
    embed_dim = 768
    latent_dim = 256
    num_heads = 8
    num_layers = 12

    # Optimization
    lr = 1e-4
    weight_decay = 0.05
    max_epochs = 5
    log_interval = 25
    ckpt_dir = Path("checkpoints")

    # Loss
    alpha = 0.2  # LPIPS weighting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # WandB
    project = "DreamerV4-tokenizer"
    entity = "hiroki-kimiwada-"   # ← remove trailing dash
    run_name = "tokenizer_v4_lpips"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, loss_dict, cfg: TrainConfig):
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.ckpt_dir / f"tokenizer_epoch{epoch:03d}.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss_dict,
    }, path)
    print(f"[Checkpoint] Saved → {path}")
    wandb.save(str(path))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_tokenizer():
    cfg = TrainConfig()

    # --- Initialize wandb safely ---
    try:
        wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            name=cfg.run_name,
            config={k: v for k, v in cfg.__dict__.items() if not k.startswith("__")},
        )
    except Exception as e:
        print(f"[WARN] W&B init failed ({e}); running offline mode.")
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(mode="offline", project=cfg.project, name=cfg.run_name)

    # --- Dataset (no DataLoader; TokenizerDataset is iterable) ---
    dataset = TokenizerDataset(
        video_dir=cfg.data_dir,
        resize=cfg.resize,
        clip_length=cfg.clip_length,
        patch_size=cfg.patch_size,
        mask_prob_range=(0.1, 0.9),
        per_frame_mask_sampling=True,
        mode="random"
    )

    # --- Model ---
    model = CausalTokenizer(
        input_dim=cfg.input_dim,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        latent_dim=cfg.latent_dim,
    ).to(cfg.device)

    # --- Loss & optimizer ---
    criterion = CombinedLoss(alpha=cfg.alpha).to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler()

    wandb.watch(model, criterion, log="all", log_freq=cfg.log_interval)

    print(f"[INFO] Training tokenizer on {cfg.device} for {cfg.max_epochs} epochs")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss, mse_loss, lpips_loss = 0.0, 0.0, 0.0

        progress = tqdm(enumerate(dataset), desc=f"Epoch {epoch}")

        for step, batch in progress:
            patches = batch["patch_tokens"].to(cfg.device)  # (T, N, D)
            mask = batch["mask"].to(cfg.device)              # (T, N)

            # Add batch dimension (B=1 if dataset yields one clip)
            patches = patches.unsqueeze(0)
            mask = mask.unsqueeze(0)

            with autocast():
                recon = model(patches, mask)
                loss, parts = criterion(recon, patches, mask.unsqueeze(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            mse_loss += parts["mse"]
            lpips_loss += parts["lpips"]

            if (step + 1) % cfg.log_interval == 0:
                avg_total = total_loss / cfg.log_interval
                avg_mse = mse_loss / cfg.log_interval
                avg_lpips = lpips_loss / cfg.log_interval

                wandb.log({
                    "train/total_loss": avg_total,
                    "train/mse_loss": avg_mse,
                    "train/lpips_loss": avg_lpips,
                    "epoch": epoch,
                    "step": step + epoch * 1000,  # synthetic step counter
                })

                progress.set_postfix({
                    "total": f"{avg_total:.4f}",
                    "mse": f"{avg_mse:.4f}",
                    "lpips": f"{avg_lpips:.4f}",
                })
                total_loss = mse_loss = lpips_loss = 0.0

        save_checkpoint(model, optimizer, epoch, parts, cfg)
        wandb.log({"epoch_end": epoch})

    wandb.finish()

if __name__ == "__main__":
    train_tokenizer()
