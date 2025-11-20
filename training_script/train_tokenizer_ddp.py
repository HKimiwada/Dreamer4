"""
# Working Code
PYTHONPATH=. torchrun --nproc_per_node=8 training_script/train_tokenizer_ddp.py
"""
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import numpy as np
import imageio.v2 as imageio

from tokenizer.tokenizer_dataset import TokenizerDatasetDDP
from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.losses import MSELoss, CombinedLoss
from tokenizer.patchify_mask import Patchifier

# ---------------------------------------------------------------------------
class OverfitConfig:
    # Target video
    target_video = "cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4"
    data_dir = Path("data")
    
    # Model architecture
    resize = (256, 448)
    patch_size = 16
    clip_length = 8
    input_dim = 3 * patch_size * patch_size
    embed_dim = 512
    latent_dim = 256
    num_heads = 16
    num_layers = 20
    
    # Training
    batch_size = 1
    num_workers = 0
    lr = 1e-4
    weight_decay = 0.0
    max_epochs = 3000
    log_interval = 5
    
    # Visualization
    visualize_interval = 10
    num_frames_to_viz = 4
    
    # Loss configuration
    use_combined_loss = False
    lpips_weight = 0.2
    lpips_net = "alex"

    # Paths
    ckpt_dir = Path("checkpoints/overfit/latest_complete_overfit_mse")
    viz_dir = Path("visualizations/latest_complete_overfit_mse")
    
    # WandB
    project = "test"
    entity = "hiroki-kimiwada-"
    run_name = "test"

# ---------------------------------------------------------------------------
def setup_ddp():
    """Initialize DDP environment variables."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    """Clean up DDP process group."""
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
def save_best_checkpoint(model, optimizer, epoch, loss, cfg, best_loss):
    """Save checkpoint if loss improves. Called only on rank 0."""
    if loss >= best_loss:
        return best_loss
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.ckpt_dir / "best_model.pt"
    torch.save(
        {
            "model_state": model.module.state_dict(),  # model.module for DDP
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        best_path,
    )
    print(f"✓ [Best Model] Epoch {epoch} | Loss: {loss:.6f} → {best_path}")
    artifact = wandb.Artifact(
        "best-model", type="model", metadata={"epoch": epoch, "loss": float(loss)}
    )
    artifact.add_file(str(best_path))
    wandb.log_artifact(artifact)
    return loss


# ---------------------------------------------------------------------------
@torch.no_grad()
def visualize_reconstruction(model, batch, cfg, epoch, device):
    """Visualize reconstruction (only called by rank 0)."""
    cfg.viz_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    
    patches = batch["patch_tokens"].to(device)
    mask = batch["mask"].to(device)
    recon = model(patches, mask)
    
    print(f"\n[DIAGNOSTIC - Epoch {epoch}]")
    print(f"Input patches shape: {patches.shape}")
    print(f"Recon patches shape: {recon.shape}")
    
    manual_mse = torch.nn.functional.mse_loss(recon, patches)
    print(f"Manual MSE: {manual_mse.item():.6f}")

    patchifier = Patchifier(patch_size=cfg.patch_size)
    H, W = cfg.resize
    num_frames = min(cfg.num_frames_to_viz, patches.shape[1])
    frames_to_viz = np.linspace(0, patches.shape[1] - 1, num_frames, dtype=int)
    images = []

    for t in frames_to_viz:
        orig_frame = patchifier.unpatchify(
            patches[0, t:t+1], (H, W), cfg.patch_size
        )[0].clamp(0, 1)
        recon_frame = patchifier.unpatchify(
            recon[0, t:t+1], (H, W), cfg.patch_size
        )[0].clamp(0, 1)
        mask_frame = mask[0, t].float().view(H // cfg.patch_size, W // cfg.patch_size)
        mask_viz = (
            mask_frame.repeat_interleave(cfg.patch_size, dim=0)
            .repeat_interleave(cfg.patch_size, dim=1)
            .unsqueeze(0)
            .repeat(3, 1, 1)
        )

        orig_np = (orig_frame.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        recon_np = (recon_frame.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_np = (mask_viz.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        combined = np.concatenate([orig_np, mask_np, recon_np], axis=1)
        images.append(combined)

    final_img = np.concatenate(images, axis=0)
    wandb.log({"visualization": wandb.Image(final_img, caption=f"Epoch {epoch}")})
    model.train()


# ---------------------------------------------------------------------------
def main():
    cfg = OverfitConfig()
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        print(f"[DDP] Initialized with {world_size} processes.")
        print(f"[Device] Rank {rank} using {device}")

    # --- wandb ---
    if is_main:
        wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            name=cfg.run_name,
            config=vars(cfg),
        )

    # --- dataset ---
    full_dataset = TokenizerDatasetDDP(
        video_dir=cfg.data_dir,
        resize=cfg.resize,
        clip_length=cfg.clip_length,
        patch_size=cfg.patch_size,
        mask_prob_range=(0.0, 0.0),
        per_frame_mask_sampling=True,
        mode="random",
    )

    target_indices = [
        i for i, s in enumerate(full_dataset.samples)
        if s["video_path"].name == cfg.target_video
    ]
    if not target_indices:
        raise RuntimeError(f"Target video {cfg.target_video} not found in {cfg.data_dir}")

    dataset = Subset(full_dataset, target_indices)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # --- model ---
    model = CausalTokenizer(
        input_dim=cfg.input_dim,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        latent_dim=cfg.latent_dim,
        use_checkpoint=False,
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[Model] {num_params:.2f}M parameters")

    # --- loss & optimizer ---
    if cfg.use_combined_loss:
        criterion = CombinedLoss(
            lpips_weight=cfg.lpips_weight,
            lpips_net=cfg.lpips_net,
            patch_size=cfg.patch_size,
            frame_size=cfg.resize,
        ).to(device)
        if is_main:
            print(f"[Loss] Using Combined Loss")
    else:
        criterion = MSELoss().to(device)
        if is_main:
            print(f"[Loss] Using MSE only")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if is_main:
        wandb.watch(model, log="all", log_freq=cfg.log_interval)

    best_loss = float("inf")

    # --- training loop ---
    for epoch in range(1, cfg.max_epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        if is_main:
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.max_epochs}")
        else:
            pbar = loader

        for batch in pbar:
            patches = batch["patch_tokens"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            recon = model(patches, mask)
            loss = criterion(recon, patches, None)
            if isinstance(loss, tuple):
                loss = loss[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if is_main and num_batches % cfg.log_interval == 0:
                wandb.log({"train/loss": loss.item(), "train/epoch": epoch})

        # Average loss across all processes
        total_loss_tensor = torch.tensor(epoch_loss, device=device)
        total_batches_tensor = torch.tensor(num_batches, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_batches_tensor, op=dist.ReduceOp.SUM)
        avg_epoch_loss = (total_loss_tensor / total_batches_tensor).item()

        if is_main:
            print(f"[Epoch {epoch}] Avg Loss: {avg_epoch_loss:.6f}")
            wandb.log({"epoch/avg_loss": avg_epoch_loss, "epoch/num": epoch})
            best_loss = save_best_checkpoint(model, optimizer, epoch, avg_epoch_loss, cfg, best_loss)

            if epoch % cfg.visualize_interval == 0:
                viz_batch = next(iter(loader))
                visualize_reconstruction(model, viz_batch, cfg, epoch, device)

    if is_main:
        print(f"\n[Training Complete] Best loss: {best_loss:.6f}")
        wandb.finish()
    cleanup_ddp()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
