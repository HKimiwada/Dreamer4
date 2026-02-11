"""
DreamerV4 Atari World Model Training — v3 (Multi-Step Rollout)
==============================================================
PYTHONPATH=. uv run torchrun --nproc_per_node=8 training_script/world_model/atari/v3_train_world_model_atari.py

KEY CHANGES FROM v2 (single-step):
-----------------------------------
1. MULTI-STEP AUTOREGRESSIVE ROLLOUT (the core fix)
   - After single-step warm-up phase, we unroll the model K steps 
   - At each step, the model's OWN prediction becomes the next input
   - Loss is computed at EVERY step against ground truth
   - This forces the model to: use actions, track state, maintain consistency

2. PROPER SHORTCUT SCHEDULE (Dreamer4-aligned)
   - d is sampled as powers-of-2 / k_max (not uniform random)
   - tau is sampled on a grid aligned to d (not continuous uniform)
   - This matches the paper's shortcut forcing formulation

3. CURRICULUM TRAINING
   - Phase 1 (0-20% of training): Single-step only (warm-up)
   - Phase 2 (20-100%): Gradually increase rollout length K from 2 → max_K
   - This stabilizes early training before adding the harder objective

4. PROPER CHECKPOINTING
   - Best model saved on running-average loss, not per-batch minimum
   
5. GRADIENT MANAGEMENT
   - Truncated BPTT: detach older steps to limit memory
   - Gradient clipping to prevent explosion from long rollouts
"""

import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import wandb
import numpy as np

from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier
from world_model.wm.dynamics_model_atari import WorldModel


# ===========================================================================
# Configuration
# ===========================================================================
class AtariWMConfig:
    # Dataset paths
    latent_path = Path("data/atari/latent_sequences/video_full_frames.pt")
    actions_jsonl = Path("data/atari/raw/actions.jsonl")
    tokenizer_ckpt = Path("checkpoints/atari/tokenizer_v3/best_model.pt")

    # Model architecture
    resize = (64, 64)
    patch_size = 8
    n_latents = 64
    input_dim = 3 * patch_size * patch_size
    latent_dim = 256
    embed_dim = 512
    action_dim = 4
    num_layers = 12
    num_heads = 8
    Sa = 1
    Sr = 8

    # Training params
    batch_size = 8              # Reduced from 16 — rollout uses more memory
    clip_length = 64
    stride = 32
    lr = 2e-4
    max_steps = 100_000
    warmup_steps = 5_000        # LR warmup
    visualize_interval = 500
    log_interval = 10
    grad_clip = 1.0

    # ---- NEW: Multi-step rollout config ----
    k_max = 64                  # Shortcut schedule granularity
    rollout_max_K = 8           # Maximum rollout steps at end of training
    rollout_start_frac = 0.2    # Start rollout training at 20% of max_steps
    rollout_ramp_frac = 0.5     # Reach max K at 50% of max_steps
    context_frames = 4          # Seed frames from ground truth before rollout
    rollout_loss_weight = 1.0   # Weight for rollout loss vs single-step loss
    # ----------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path("checkpoints/world_model/latest_atari")
    project = "latest_feburary_wm_atari"
    entity = "hiroki-kimiwada-"


# ===========================================================================
# Dataset (unchanged)
# ===========================================================================
class AtariWorldModelDataset(Dataset):
    def __init__(self, cfg):
        print(f"[Dataset] Loading latents from {cfg.latent_path}...")
        data = torch.load(cfg.latent_path, map_location="cpu")
        self.latents = data["z"]

        print(f"[Dataset] Loading actions from {cfg.actions_jsonl}...")
        self.actions = []
        with open(cfg.actions_jsonl, "r") as f:
            for line in f:
                entry = json.loads(line)
                self.actions.append(entry["action"])

        self.actions = torch.tensor(self.actions, dtype=torch.long)[:len(self.latents)]
        self.clip_length = cfg.clip_length
        self.stride = cfg.stride
        self.start_indices = list(range(0, len(self.latents) - self.clip_length, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.clip_length
        return {
            "latents": self.latents[start:end],
            "actions": self.actions[start:end],
            "start_idx": start,
        }


# ===========================================================================
# Data Builder (unchanged architecture, cleaner interface)
# ===========================================================================
class AtariDataBuilder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.embed_dim)
        self.action_embed = nn.Embedding(cfg.action_dim, cfg.embed_dim)
        self.register_embed = nn.Embedding(cfg.Sr, cfg.embed_dim)
        self.shortcut_slot = nn.Parameter(torch.randn(cfg.embed_dim))
        self.shortcut_mlp = nn.Sequential(
            nn.Linear(2, cfg.embed_dim), nn.SiLU(), nn.Linear(cfg.embed_dim, cfg.embed_dim)
        )

    def forward(self, latents, actions, tau, d):
        """
        Args:
            latents: (B, T, N, D_latent) — corrupted latents
            actions:  (B, T) — discrete actions
            tau:      (B, T) — signal level per frame
            d:        (B,)   — step size
        Returns:
            tokens: (B, T * L_total, E) — flattened token sequence for transformer
        """
        B, T, N, D = latents.shape
        E = self.cfg.embed_dim

        z_tokens = self.latent_proj(latents)                                     # (B,T,N,E)
        a_tokens = self.action_embed(actions).unsqueeze(2)                       # (B,T,1,E)
        reg_ids = torch.arange(self.cfg.Sr, device=latents.device)
        reg_tokens = self.register_embed(reg_ids).view(1, 1, self.cfg.Sr, E).expand(B, T, -1, -1)

        feat = torch.stack([tau, d.view(B, 1).expand(B, T)], dim=-1)            # (B,T,2)
        s_tokens = (self.shortcut_mlp(feat) + self.shortcut_slot).unsqueeze(2)   # (B,T,1,E)

        tokens = torch.cat([z_tokens, a_tokens, reg_tokens, s_tokens], dim=2)   # (B,T,L,E)
        return tokens.view(B, T * tokens.shape[2], E)


# ===========================================================================
# KEY CHANGE #2: Proper Shortcut Schedule
# ===========================================================================
def sample_shortcut_schedule(B, T, k_max, device):
    """
    Dreamer4-aligned shortcut schedule.
    
    d is sampled as powers-of-2 / k_max:  {1/64, 2/64, 4/64, ..., 1}
    tau is sampled on a grid aligned to d: {0, d, 2d, ..., 1-d}
    
    This ensures the model sees a structured curriculum of denoising difficulties,
    not arbitrary continuous noise levels.
    
    Args:
        B: batch size
        T: sequence length
        k_max: finest granularity (e.g., 64)
        device: torch device
    
    Returns:
        tau: (B, T) — signal levels on the d-aligned grid
        d:   (B,)   — step sizes (powers of 2 / k_max)
    """
    max_pow = int(math.log2(k_max))

    # Sample d = 2^p / k_max for random integer p in [0, max_pow]
    pow_idx = torch.randint(0, max_pow + 1, (B,), device=device)
    d = (2.0 ** pow_idx.float()) / k_max  # (B,)

    # Sample tau on the grid {0, d, 2d, ..., 1-d} for each batch element
    tau = torch.zeros(B, T, device=device)
    for b in range(B):
        d_val = d[b].item()
        num_steps = int(1.0 / d_val)
        # Random grid-aligned step index per frame
        step_idx = torch.randint(0, num_steps, (T,), device=device)
        tau[b] = step_idx.float() * d_val

    return tau, d


def corrupt_latents(z_clean, tau):
    """
    Flow matching corruption: z_corr = (1 - tau) * noise + tau * z_clean
    tau=0 → pure noise, tau=1 → clean
    """
    noise = torch.randn_like(z_clean)
    tau_exp = tau.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
    z_corr = (1.0 - tau_exp) * noise + tau_exp * z_clean
    return z_corr, noise


# ===========================================================================
# Loss Functions
# ===========================================================================
def flow_loss(pred_z, z_clean, tau):
    """
    Weighted MSE loss for flow matching.
    Weight: (1 - tau + 0.1) — prioritizes denoising noisy frames while
    keeping a floor so clean frames still contribute.
    """
    sq_error = (pred_z - z_clean).pow(2)  # (B, T, N, D)
    w = (1.0 - tau + 0.1).unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
    return (w * sq_error).mean()


# ===========================================================================
# KEY CHANGE #1: Multi-Step Autoregressive Rollout
# ===========================================================================
def compute_rollout_loss(
    wm, builder, z_gt, actions, time_offsets, cfg, K, device
):
    """
    THE CORE CHANGE: Multi-step autoregressive rollout with gradient.
    
    Instead of just doing single-step denoising, we:
    1. Seed with `context_frames` of ground truth
    2. For K steps, predict the next frame using the model's OWN previous predictions
    3. Compute loss at EACH step against ground truth
    4. Gradients flow through the entire rollout (with truncation for memory)
    
    This forces the model to:
    - Use actions (ignoring them compounds error over K steps)
    - Track state (block destruction, score) because inconsistency creates loss
    - Produce temporally coherent predictions
    
    Args:
        wm: world model (unwrapped from DDP)
        builder: data builder (unwrapped from DDP)
        z_gt: (B, T, N, D) ground truth latents for the full clip
        actions: (B, T) actions for the full clip
        time_offsets: (B,) global time indices
        cfg: config
        K: number of rollout steps this iteration
        device: torch device
    
    Returns:
        rollout_loss: scalar loss averaged over all rollout steps
    """
    B, T, N, D = z_gt.shape
    ctx = cfg.context_frames

    # We need at least context + K frames in the clip
    if ctx + K > T:
        K = T - ctx
    if K <= 0:
        return torch.tensor(0.0, device=device)

    # Start with ground truth context frames
    z_current = z_gt[:, :ctx].clone()  # (B, ctx, N, D)

    total_loss = torch.tensor(0.0, device=device)
    num_steps = 0

    for step in range(K):
        target_t = ctx + step  # Index of the frame we're predicting

        # Build the full window the model sees: [context | predicted so far | noisy target]
        # The model always operates on a fixed-size window. We construct a window
        # ending at target_t, padded/filled as needed.
        
        # Window: last (ctx + step) frames of z_current, plus a noisy version of target
        # For simplicity: use a sliding window of size min(ctx + step + 1, T)
        window_len = min(ctx + step + 1, T)
        
        # Pad z_current with a noisy estimate for the target frame
        # tau=0 means pure noise (hardest prediction task)
        noise_target = torch.randn(B, 1, N, D, device=device)
        z_window = torch.cat([z_current, noise_target], dim=1)[:, -window_len:]  # (B, window_len, N, D)
        
        # Actions for this window
        a_window = actions[:, target_t - window_len + 1 : target_t + 1]  # (B, window_len)
        
        # Shortcut schedule: tau=0 for target frame (pure prediction), tau=1 for context
        tau_window = torch.ones(B, window_len, device=device)   # Known frames: tau=1
        tau_window[:, -1] = 0.0                                  # Target frame: tau=0 (pure noise)
        d_window = torch.ones(B, device=device)                  # d=1 (single-step prediction)
        
        # Time offsets adjusted for the window position
        t_offsets = time_offsets + (target_t - window_len + 1)
        
        # Forward pass
        tokens = builder(z_window, a_window, tau_window, d_window)
        wm_input = {"wm_input_tokens": tokens, "tau": tau_window, "d": d_window}
        pred_z = wm(wm_input, time_offsets=t_offsets)  # (B, window_len, N, D)
        
        # Loss on the TARGET frame only (the last frame in the window)
        pred_target = pred_z[:, -1]    # (B, N, D)
        gt_target = z_gt[:, target_t]  # (B, N, D)
        step_loss = F.mse_loss(pred_target, gt_target)
        total_loss = total_loss + step_loss
        num_steps += 1
        
        # KEY: Feed prediction back as input for next step
        # Detach every `truncate_every` steps to limit memory (truncated BPTT)
        pred_target_3d = pred_target.unsqueeze(1)  # (B, 1, N, D)
        
        # Detach periodically to manage memory (every 4 steps)
        if step > 0 and step % 4 == 0:
            z_current = torch.cat([z_current.detach(), pred_target_3d], dim=1)
        else:
            z_current = torch.cat([z_current, pred_target_3d], dim=1)

    return total_loss / max(num_steps, 1)


# ===========================================================================
# KEY CHANGE #3: Curriculum — compute current rollout length K
# ===========================================================================
def get_current_rollout_K(global_step, cfg):
    """
    Curriculum schedule for rollout length K.
    
    - Before rollout_start_frac:    K = 0 (single-step only)
    - rollout_start → rollout_ramp: K linearly increases from 2 → rollout_max_K
    - After rollout_ramp_frac:      K = rollout_max_K
    
    This prevents destabilizing early training with long rollouts
    before the model has learned basic denoising.
    """
    start_step = int(cfg.max_steps * cfg.rollout_start_frac)
    ramp_step = int(cfg.max_steps * cfg.rollout_ramp_frac)

    if global_step < start_step:
        return 0  # Pure single-step phase
    elif global_step >= ramp_step:
        return cfg.rollout_max_K
    else:
        # Linear ramp from 2 to max_K
        progress = (global_step - start_step) / max(ramp_step - start_step, 1)
        return int(2 + progress * (cfg.rollout_max_K - 2))


# ===========================================================================
# LR Scheduler with Warmup
# ===========================================================================
def get_lr(step, cfg):
    """Linear warmup then cosine decay."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    # Cosine decay after warmup
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ===========================================================================
# Visualization (unchanged from v4, with local positional context fix)
# ===========================================================================
@torch.no_grad()
def visualize_step(wm, builder, tokenizer, batch, cfg, step, device):
    wm_eval = wm.module if hasattr(wm, "module") else wm
    builder_eval = builder.module if hasattr(builder, "module") else builder
    wm_eval.eval(); builder_eval.eval(); tokenizer.eval()

    latents = batch["latents"][:1].to(device)
    actions = batch["actions"][:1].to(device)
    start_indices = batch["start_idx"][:1].to(device)
    B, T, N, D = latents.shape

    tau = torch.full((B, T), 0.5, device=device)
    d = torch.full((B,), 0.25, device=device)
    z_corr, _ = corrupt_latents(latents, tau)

    tokens = builder_eval(z_corr, actions, tau, d)
    wm_input = {"wm_input_tokens": tokens, "tau": tau, "d": d}
    pred_z = wm_eval(wm_input, time_offsets=start_indices)

    def decode_local(z_seq):
        B_v, T_v, N_v, D_v = z_seq.shape
        x = tokenizer.from_latent(z_seq)
        x = x.view(B_v, T_v * N_v, tokenizer.embed_dim)
        x = x + tokenizer.pos_embed[:, :T_v * N_v, :]
        x = tokenizer._run_stack(x, tokenizer.decoder, T=T_v, N=N_v)
        x = x.view(B_v, T_v, N_v, tokenizer.embed_dim)
        patches = tokenizer.output_proj(x)
        frames = Patchifier(cfg.patch_size).unpatchify(patches.squeeze(0), cfg.resize, cfg.patch_size)
        return frames[-4:]

    gt_f = decode_local(latents)
    pr_f = decode_local(pred_z)

    rows = []
    for i in range(4):
        combined = torch.cat([gt_f[i].clamp(0, 1), pr_f[i].clamp(0, 1)], dim=-1)
        img_np = (combined.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rows.append(img_np)

    final_grid = np.concatenate(rows, axis=0)
    wandb.log({
        "reconstruction": wandb.Image(final_grid, caption=f"Step {step} (L:GT R:Pred)"),
        "step": step,
    })
    wm_eval.train(); builder_eval.train()


# ===========================================================================
# DDP Setup
# ===========================================================================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


# ===========================================================================
# Main Training Loop
# ===========================================================================
def main():
    cfg = AtariWMConfig()
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        wandb.init(project=cfg.project, entity=cfg.entity, config=vars(cfg))
        cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Models ---
    tokenizer = CausalTokenizer(
        input_dim=cfg.input_dim, embed_dim=256, num_heads=8, num_layers=8, latent_dim=256
    )
    tk_ckpt = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
    tokenizer.load_state_dict({k.replace("module.", ""): v for k, v in tk_ckpt["model_state"].items()})
    tokenizer.to(device).eval()

    builder = AtariDataBuilder(cfg).to(device)
    wm = WorldModel(
        d_model=cfg.embed_dim, d_latent=cfg.latent_dim, num_layers=cfg.num_layers,
        num_heads=cfg.num_heads, n_latents=cfg.n_latents, Sr=cfg.Sr, use_checkpoint=True,
    ).to(device)

    wm = DDP(wm, device_ids=[local_rank])
    builder = DDP(builder, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        list(wm.parameters()) + list(builder.parameters()),
        lr=cfg.lr, weight_decay=0.01,
    )
    scaler = GradScaler()

    # --- Data ---
    dataset = AtariWorldModelDataset(cfg)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # --- Training State ---
    global_step = 0
    best_avg_loss = float("inf")
    loss_accumulator = 0.0
    loss_count = 0
    epoch = 0

    # --- Training Loop ---
    while global_step < cfg.max_steps:
        sampler.set_epoch(epoch)

        for batch in tqdm(loader, disable=not is_main, desc=f"Epoch {epoch}"):
            if global_step >= cfg.max_steps:
                break

            # ---- LR Schedule ----
            lr = get_lr(global_step, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            latents = batch["latents"].to(device)          # (B, T, N, D)
            actions = batch["actions"].to(device)           # (B, T)
            time_offsets = batch["start_idx"].to(device)    # (B,)
            B, T, N, D = latents.shape

            # ===========================================================
            # PART A: Single-step flow matching loss (always active)
            # ===========================================================
            # KEY CHANGE: Use proper shortcut schedule instead of uniform
            tau, d = sample_shortcut_schedule(B, T, cfg.k_max, device)
            z_corr, noise = corrupt_latents(latents, tau)

            with autocast(device_type="cuda", dtype=torch.float16):
                tokens = builder(z_corr, actions, tau, d)
                wm_input = {"wm_input_tokens": tokens, "tau": tau, "d": d}
                pred_z = wm(wm_input, time_offsets=time_offsets)
                single_step_loss = flow_loss(pred_z, latents, tau)

            # ===========================================================
            # PART B: Multi-step rollout loss (activated by curriculum)
            # ===========================================================
            K = get_current_rollout_K(global_step, cfg)
            
            if K > 0:
                # Unwrap DDP for direct access during rollout
                wm_inner = wm.module if hasattr(wm, "module") else wm
                builder_inner = builder.module if hasattr(builder, "module") else builder

                with autocast(device_type="cuda", dtype=torch.float16):
                    rollout_loss = compute_rollout_loss(
                        wm_inner, builder_inner, latents, actions,
                        time_offsets, cfg, K, device,
                    )
                
                total_loss = single_step_loss + cfg.rollout_loss_weight * rollout_loss
            else:
                rollout_loss = torch.tensor(0.0, device=device)
                total_loss = single_step_loss

            # ===========================================================
            # Backward + Optimize
            # ===========================================================
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()

            # Gradient clipping — important for rollout stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(wm.parameters()) + list(builder.parameters()), cfg.grad_clip
            )

            scaler.step(optimizer)
            scaler.update()

            # ===========================================================
            # Logging
            # ===========================================================
            global_step += 1
            loss_accumulator += total_loss.item()
            loss_count += 1

            if is_main and global_step % cfg.log_interval == 0:
                avg_loss = loss_accumulator / loss_count
                wandb.log({
                    "loss/total": total_loss.item(),
                    "loss/single_step": single_step_loss.item(),
                    "loss/rollout": rollout_loss.item(),
                    "loss/running_avg": avg_loss,
                    "schedule/lr": lr,
                    "schedule/rollout_K": K,
                    "step": global_step,
                })

            # KEY CHANGE: Save on running average, not per-batch minimum
            if is_main and global_step % 100 == 0:
                avg_loss = loss_accumulator / max(loss_count, 1)
                if avg_loss < best_avg_loss:
                    best_avg_loss = avg_loss
                    torch.save(wm.module.state_dict(), cfg.ckpt_dir / "best_wm.pt")
                    torch.save(builder.module.state_dict(), cfg.ckpt_dir / "best_builder.pt")
                    print(f"✓ New best model at step {global_step}, avg_loss={avg_loss:.6f}")
                loss_accumulator = 0.0
                loss_count = 0

            # Periodic checkpoint (not just best)
            if is_main and global_step % 10_000 == 0:
                torch.save(wm.module.state_dict(), cfg.ckpt_dir / f"wm_step{global_step}.pt")

            if is_main and global_step % cfg.visualize_interval == 0:
                visualize_step(wm, builder, tokenizer, batch, cfg, global_step, device)

        epoch += 1

    if is_main:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()