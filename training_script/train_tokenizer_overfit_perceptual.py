"""
Testing Perceptual Loss (LPIPS: 0.2 weight)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python training_script/train_tokenizer_overfit_perceptual.py : LPIPS weight 0.2, Masking(0.0, 0.0)
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python training_script/train_tokenizer_overfit_perceptual.py : LPIPS weight 0.2, Masking(0.2, 0.5)
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python training_script/train_tokenizer_overfit_perceptual.py : LPIPS weight 0.2, Masking(0.5, 0.8)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python training_script/train_tokenizer_overfit_perceptual.py : LPIPS weight 0.2, Masking(0.0, 0.0)

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python training_script/train_tokenizer_overfit_perceptual.py
"""
import os
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import numpy as np
import imageio.v2 as imageio

from tokenizer.tokenizer_dataset import TokenizerDatasetDDP
from tokenizer.model.encoder_decoder import CausalTokenizer2
from tokenizer.losses import MSELoss, CombinedLoss
from tokenizer.patchify_mask import Patchifier

# ---------------------------------------------------------------------------
class OverfitConfig:
    # Target video
    target_video = "cheeky-cornflower-setter-0a5ba522405b-20220422-133010.mp4"
    data_dir = Path("data")
    
    # Model architecture
    resize = (256, 448)
    patch_size = 16
    clip_length = 8
    input_dim = 3 * patch_size * patch_size
    embed_dim = 512
    latent_dim = 384
    num_heads = 8
    num_layers = 18
    
    # Training
    batch_size = 1
    num_workers = 0  # 0 for overfitting (simpler debugging)
    lr = 1e-4  # higher LR for faster overfitting
    weight_decay = 0.0  # no regularization when overfitting
    max_epochs = 1500
    log_interval = 5
    
    # Visualization
    visualize_interval = 10  # visualize reconstructions every N epochs
    num_frames_to_viz = 4  # how many frames to visualize
    
    # Loss configuration
    use_combined_loss = False  # Set to False to use MSE only
    lpips_weight = 0.2  # Weight for LPIPS (as in Dreamer4 paper)
    lpips_net = 'alex'  # 'alex', 'vgg', or 'squeeze'

    # Paths
    ckpt_dir = Path("checkpoints/overfit/v5_overfit_mse")
    viz_dir = Path("visualizations/v5_overfit_mse")
    
    # WandB
    project = "11_09_Test_LPIPS_Masking"
    entity = "hiroki-kimiwada-"
    run_name = "v5_run"  # Updated run name

# ---------------------------------------------------------------------------
def save_best_checkpoint(model, optimizer, epoch, loss, cfg, best_loss):
    """
    Save checkpoint only if current loss is better than best_loss.
    Overwrites the same file each time.
    """
    if loss >= best_loss:
        return best_loss  # No improvement, don't save
    
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.ckpt_dir / "best_model.pt"
    
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, best_path)
    
    print(f"✓ [Best Model] Epoch {epoch} | Loss: {loss:.6f} → {best_path}")
    
    # Log to wandb (artifact gets updated, not duplicated)
    artifact = wandb.Artifact("best-model", type="model", metadata={"epoch": epoch, "loss": float(loss)})
    artifact.add_file(str(best_path))
    wandb.log_artifact(artifact)
    
    return loss  # Update best_loss

# ---------------------------------------------------------------------------
def visualize_reconstruction(model, batch, cfg, epoch, device):
    """
    Visualize original vs reconstructed frames and log to wandb.
    """
    cfg.viz_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        patches = batch["patch_tokens"].to(device)  # (B, T, N, D)
        mask = batch["mask"].to(device)  # (B, T, N)
        
        # Get reconstruction
        recon = model(patches, mask)  # (B, T, N, D)
        
        # ============================================================
        # HYPOTHESIS 1: Check if reconstruction makes sense
        # ============================================================
        print(f"\n[DIAGNOSTIC - Epoch {epoch}]")
        print(f"Input patches shape: {patches.shape}")
        print(f"Input patches range: [{patches.min():.3f}, {patches.max():.3f}]")
        print(f"Recon patches shape: {recon.shape}")
        print(f"Recon patches range: [{recon.min():.3f}, {recon.max():.3f}]")
        print(f"Recon contains NaN: {torch.isnan(recon).any()}")
        print(f"Recon contains Inf: {torch.isinf(recon).any()}")
        
        # Compute MSE manually to verify
        manual_mse = torch.nn.functional.mse_loss(recon, patches)
        print(f"Manual MSE: {manual_mse.item():.6f}")
        
        # Check if reconstruction is actually different from input
        diff = (recon - patches).abs().mean()
        print(f"Mean absolute difference: {diff.item():.6f}")
        
        # ============================================================
        # HYPOTHESIS 3: Check if output is constant/low variance
        # ============================================================
        recon_mean = recon.mean()
        recon_std = recon.std()
        recon_var = recon.var()
        print(f"\nReconstruction statistics:")
        print(f"  Mean: {recon_mean.item():.6f}")
        print(f"  Std: {recon_std.item():.6f}")
        print(f"  Variance: {recon_var.item():.6f}")
        
        if recon_var < 0.001:
            print("⚠️  WARNING: Reconstruction has very low variance - model might be outputting constant!")
        
        # Check if reconstruction has spatial structure
        # Compare variance within a frame vs across frames
        frame_0_var = recon[0, 0].var()  # First frame variance
        print(f"  Single frame variance: {frame_0_var.item():.6f}")
        
        # ============================================================
        # Take first batch item
        # ============================================================
        patches_orig = patches[0]  # (T, N, D)
        patches_recon = recon[0]  # (T, N, D)
        mask_single = mask[0]  # (T, N)
        
        # ============================================================
        # HYPOTHESIS 2: Check raw patches before unpatchifying
        # ============================================================
        print(f"\nBefore unpatchify - first 10 values of patch 0, frame 0:")
        print(f"  Original: {patches_orig[0, 0, :10].cpu().numpy()}")
        print(f"  Recon:    {patches_recon[0, 0, :10].cpu().numpy()}")
          
        # ============================================================
        # Unpatchify back to images
        # ============================================================
        patchifier = Patchifier(patch_size=cfg.patch_size)
        H, W = cfg.resize
        
        # Take subset of frames for visualization
        num_frames = min(cfg.num_frames_to_viz, patches_orig.shape[0])
        frames_to_viz = np.linspace(0, patches_orig.shape[0] - 1, num_frames, dtype=int)
        
        images = []
        for t in frames_to_viz:
            # Original
            orig_frame = patchifier.unpatchify(
                patches_orig[t:t+1], 
                frame_size=(H, W), 
                patch_size=cfg.patch_size
            )[0].clamp(0,1)  # (3, H, W)
            
            # Reconstruction
            recon_frame = patchifier.unpatchify(
                patches_recon[t:t+1],
                frame_size=(H, W),
                patch_size=cfg.patch_size
            )[0].clamp(0,1)  # (3, H, W)
            
            # ============================================================
            # HYPOTHESIS 2: Check after unpatchifying
            # ============================================================
            if t == frames_to_viz[0]:  # Only for first frame
                print(f"\nAfter unpatchify - frame {t}:")
                print(f"  Original range: [{orig_frame.min():.3f}, {orig_frame.max():.3f}]")
                print(f"  Recon range: [{recon_frame.min():.3f}, {recon_frame.max():.3f}]")
                print(f"  Original top-left 3x3 pixels (R channel):")
                print(f"    {orig_frame[0, :3, :3].cpu().numpy()}")
                print(f"  Recon top-left 3x3 pixels (R channel):")
                print(f"    {recon_frame[0, :3, :3].cpu().numpy()}")
            
            # Mask visualization (where was it masked?)
            mask_frame = mask_single[t]  # (N,)
            mask_viz = mask_frame.float().view(H // cfg.patch_size, W // cfg.patch_size)
            mask_viz = mask_viz.repeat_interleave(cfg.patch_size, dim=0).repeat_interleave(cfg.patch_size, dim=1)
            mask_viz = mask_viz.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
            
            # Convert to numpy and stack horizontally
            orig_np = (orig_frame.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            recon_np = (recon_frame.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            mask_np = (mask_viz.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            
            # Stack: [original | mask | reconstruction]
            combined = np.concatenate([orig_np, mask_np, recon_np], axis=1)
            images.append(combined)
        
        # Stack all frames vertically
        final_img = np.concatenate(images, axis=0)
        
        # Save to disk
        # viz_path = cfg.viz_dir / f"epoch{epoch:03d}.png"
        # imageio.imwrite(viz_path, final_img)
        # print(f"\n[Visualization] Saved → {viz_path}")
        # print("="*60 + "\n")
        
        # Log to wandb
        wandb.log({
            "visualization": wandb.Image(final_img, caption=f"Epoch {epoch}"),
            "epoch": epoch
        })
    
    model.train()

# ---------------------------------------------------------------------------
def main():
    # --- setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    
    cfg = OverfitConfig()
    
    # --- wandb ---
    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=cfg.run_name,
        config=vars(cfg)
    )
    
    # --- load full dataset ---
    full_dataset = TokenizerDatasetDDP(
        video_dir=cfg.data_dir,
        resize=cfg.resize,
        clip_length=cfg.clip_length,
        patch_size=cfg.patch_size,
        mask_prob_range=(0.0, 0.0),
        per_frame_mask_sampling=True,
        mode="random",
    )
    
    # --- filter to target video ---
    target_indices = []
    for idx, sample_info in enumerate(full_dataset.samples):
        video_path = sample_info["video_path"]
        if video_path.name == cfg.target_video:
            target_indices.append(idx)
    
    if not target_indices:
        raise RuntimeError(f"Target video {cfg.target_video} not found in {cfg.data_dir}")
    
    print(f"[Data] Found {len(target_indices)} clips from target video: {cfg.target_video}")
    
    # Create subset with only target video
    dataset = Subset(full_dataset, target_indices)
    
    # DataLoader (no DDP, simple iteration)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # --- model ---
    model = CausalTokenizer2(
        input_dim=cfg.input_dim,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        latent_dim=cfg.latent_dim,
        use_checkpoint=False,  # disable for overfitting (faster)
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] {num_params:.2f}M parameters")
    
    # --- loss & optimizer ---
    if cfg.use_combined_loss:
        criterion = CombinedLoss(
            lpips_weight=cfg.lpips_weight,
            lpips_net=cfg.lpips_net,
            patch_size=cfg.patch_size,
            frame_size=cfg.resize
        ).to(device)
        print(f"[Loss] Using Combined Loss (MSE + {cfg.lpips_weight} * LPIPS)")
    else:
        criterion = MSELoss().to(device)
        print(f"[Loss] Using MSE only")
    
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Watch model in wandb
    wandb.watch(model, log="all", log_freq=cfg.log_interval)
    
    # --- training loop ---
    best_loss = float('inf')
    
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_lpips = 0.0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.max_epochs}")
        for step, batch in enumerate(pbar, start=1):
            patches = batch["patch_tokens"].to(device)  # (B, T, N, D)
            mask = batch["mask"].to(device)  # (B, T, N)
            
            # Forward
            recon = model(patches, mask)
            
            # Compute loss (handle both MSE and CombinedLoss signatures)
            loss_output = criterion(recon, patches, None)
            
            if isinstance(loss_output, tuple):
                # CombinedLoss returns (loss, loss_dict)
                loss, loss_dict = loss_output
                batch_mse = loss_dict['mse']
                batch_lpips = loss_dict['lpips']
                epoch_mse += batch_mse
                epoch_lpips += batch_lpips
            else:
                # MSELoss returns scalar only
                loss = loss_output
                batch_mse = loss.item()
                batch_lpips = 0.0
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional but good practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            if cfg.use_combined_loss:
                pbar.set_postfix({
                    "loss": loss.item(),
                    "mse": batch_mse,
                    "lpips": batch_lpips
                })
            else:
                pbar.set_postfix({"loss": loss.item()})
            
            # Log to wandb
            if step % cfg.log_interval == 0:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/epoch": epoch,
                    "train/step": step,
                }
                if cfg.use_combined_loss:
                    log_dict.update({
                        "train/mse": batch_mse,
                        "train/lpips": batch_lpips,
                    })
                wandb.log(log_dict)
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        
        # Log epoch averages
        log_dict = {
            "epoch/avg_loss": avg_epoch_loss,
            "epoch/num": epoch,
        }
        
        if cfg.use_combined_loss:
            avg_mse = epoch_mse / num_batches
            avg_lpips = epoch_lpips / num_batches
            log_dict.update({
                "epoch/avg_mse": avg_mse,
                "epoch/avg_lpips": avg_lpips,
            })
            print(f"[Epoch {epoch}] Avg Loss: {avg_epoch_loss:.6f} | MSE: {avg_mse:.6f} | LPIPS: {avg_lpips:.6f}")
        else:
            print(f"[Epoch {epoch}] Avg Loss: {avg_epoch_loss:.6f}")
        
        wandb.log(log_dict)
        
        # --- save checkpoint if improved ---
        #best_loss = save_best_checkpoint(model, optimizer, epoch, avg_epoch_loss, cfg, best_loss)
        
        # --- visualize reconstruction ---
        if epoch % cfg.visualize_interval == 0:
            # Get one batch for visualization
            viz_batch = next(iter(loader))
            visualize_reconstruction(model, viz_batch, cfg, epoch, device)
    
    print(f"\n[Training Complete] Best loss: {best_loss:.6f}")
    print(f"Best model saved at: {cfg.ckpt_dir / 'best_model.pt'}")
    wandb.finish()

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()