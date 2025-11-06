"""
wandb login
PYTHONPATH=. torchrun --nproc_per_node=2 training_script/train_tokenizer.py
Overview of Training script for tokenizer:
    1. Load video patch data from your preprocessed dataset (TokenizerDatasetDDP).
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
import torch.distributed as dist
from torch import amp
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import wandb

from tokenizer.tokenizer_dataset import TokenizerDatasetDDP
from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.losses import MSELoss

# ---------------------------------------------------------------------------
class TrainConfig:
    data_dir = Path("data")
    resize = (256, 448)
    patch_size = 16
    clip_length = 8
    batch_size = 1
    num_workers = 1
    input_dim = 3 * patch_size * patch_size
    embed_dim = 512
    latent_dim = 256
    num_heads = 8
    num_layers = 12
    lr = 1e-5
    weight_decay = 0.05
    max_epochs = 1
    log_interval = 5
    accumulation_steps = 8
    ckpt_dir = Path("checkpoints")
    alpha = 0.0
    project = "DreamerV4-tokenizer"
    entity = "hiroki-kimiwada-"     
    run_name = "v3_tokenizer_mse_only"

# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, cfg, rank):
    if rank != 0:
        return
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.ckpt_dir / f"tokenizer_epoch{epoch:03d}.pt"
    torch.save({
        "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }, ckpt_path)
    print(f"[Checkpoint] Saved → {ckpt_path}")
    artifact = wandb.Artifact(f"tokenizer-epoch{epoch:03d}", type="model", metadata={"epoch": epoch})
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)

# ---------------------------------------------------------------------------
def main():
    # --- distributed setup ---
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # --- memory safety: limit per-process GPU allocation ---
    torch.cuda.set_per_process_memory_fraction(0.9, device=device)
    torch.backends.cuda.matmul.allow_tf32 = True  # safe perf gain
    torch.backends.cudnn.benchmark = True         # speed up convs

    # --- config ---
    cfg = TrainConfig()

    # --- wandb only on rank 0 ---
    if local_rank == 0:
        wandb.init(project=cfg.project, entity=cfg.entity,
                   name=cfg.run_name, config=vars(cfg))

    # --- data ---
    dataset = TokenizerDatasetDDP(
        video_dir=cfg.data_dir,
        resize=cfg.resize,
        clip_length=cfg.clip_length,
        patch_size=cfg.patch_size,
        mask_prob_range=(0.1, 0.9),
        per_frame_mask_sampling=True,
        mode="random",
    )
    sampler = DistributedSampler(dataset, shuffle=True)

    use_workers = cfg.num_workers > 0
    prefetch = 1 if use_workers else None
    persist  = use_workers  # must be False when num_workers == 0

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        prefetch_factor=prefetch,      # None when num_workers == 0
        persistent_workers=persist,    # False when num_workers == 0
        pin_memory=True
    )

    # --- model / loss / optim ---
    model = CausalTokenizer(
        input_dim=cfg.input_dim,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        latent_dim=cfg.latent_dim,
    ).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = MSELoss().to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler()

    if local_rank == 0:
        wandb.watch(model, criterion, log="all", log_freq=cfg.log_interval)

    print(f"[Rank {local_rank}] ready on {device}")

    # --- training loop ---
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        running_total = running_mse = running_lpips = 0.0

        for step, batch in enumerate(tqdm(loader, disable=(local_rank != 0)), start=1):
            try:
                patches = batch["patch_tokens"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)

                # autocast in FP16
                with amp.autocast(device_type="cuda"):
                    recon = model(patches, mask)
                    loss = criterion(recon, patches, mask.unsqueeze(-1))
                    parts = {'mse': loss.item(), 'lpips': 0.0}

                loss = loss / cfg.accumulation_steps
                scaler.scale(loss).backward()

                # gradient step every accumulation_steps
                if step % cfg.accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                # update metrics
                running_total += loss.item() * cfg.accumulation_steps
                running_mse += parts["mse"]
                running_lpips += parts["lpips"]

                # log
                if local_rank == 0 and step % cfg.log_interval == 0:
                    wandb.log({
                        "train/total_loss": running_total / cfg.log_interval,
                        "train/mse_loss": running_mse / cfg.log_interval,
                        "train/lpips_loss": running_lpips / cfg.log_interval,
                        "epoch": epoch,
                        "step": step,
                    })
                    print(f"[Epoch {epoch} Step {step}] total={running_total / cfg.log_interval:.4f}")
                    running_total = running_mse = running_lpips = 0.0

            # --- graceful OOM handler ---
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[Rank {local_rank}] CUDA OOM detected at step {step}. Freeing memory and exiting gracefully.")
                    # free cached memory
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    import sys
                    sys.exit(1)
                else:
                    raise  # rethrow non-OOM errors

        # --- checkpoint after each epoch (rank 0 only) ---
        if local_rank == 0:
            save_checkpoint(model, optimizer, epoch, cfg, local_rank)
            wandb.log({"epoch_end": epoch})

    # --- cleanup ---
    dist.destroy_process_group()

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
