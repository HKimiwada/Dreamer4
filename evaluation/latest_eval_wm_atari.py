"""
Drop-in Evaluation Script (FIXED) — DreamerV4 Atari WM v3
========================================================

Fixes your current eval bugs:
- Off-by-one skip that made rollout/action metrics all null
- Action sanity loop break logic that could exit before collecting anything
- Adds explicit counters + asserts so you never silently get nulls again
- Saves a few rollout videos + action counterfactual videos

Run:
  PYTHONPATH=. uv run python evaluation/latest_eval_wm_atari.py

Outputs:
  evaluation/results/latest_atari_eval/
    metrics.json
    plots/*.png
    videos/
      rollout_gt_vs_pred_00.mp4
      action_sanity_00.mp4
      ...
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

import cv2

from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier
from world_model.wm.dynamics_model_atari import WorldModel
from training_script.world_model.atari.v3_train_world_model_atari import AtariDataBuilder


# -----------------------------
# Config
# -----------------------------
@dataclass
class EvalConfig:
    # Checkpoints / data
    tokenizer_ckpt: Path = Path("checkpoints/atari/tokenizer_v3/best_model.pt")
    wm_ckpt: Path = Path("checkpoints/world_model/latest_atari/best_wm.pt")
    builder_ckpt: Path = Path("checkpoints/world_model/latest_atari/best_builder.pt")
    latent_path: Path = Path("data/atari/latent_sequences/video_full_frames.pt")
    actions_jsonl: Path = Path("data/atari/raw/actions.jsonl")

    # Output
    out_dir: Path = Path("evaluation/results/latest_atari_eval")

    # Must match training
    resize: Tuple[int, int] = (64, 64)
    patch_size: int = 8
    n_latents: int = 64
    latent_dim: int = 256
    embed_dim: int = 512
    tokenizer_dim: int = 256
    num_layers: int = 12
    num_heads: int = 8
    action_dim: int = 4
    Sr: int = 8
    Sa: int = 1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset slicing (matches training clip sampling)
    clip_length: int = 64
    stride: int = 32

    # Eval settings
    num_eval_clips: int = 64
    batch_size: int = 8
    num_workers: int = 2

    context_frames: int = 4
    rollout_horizons: List[int] = None
    one_step_taus: List[float] = None

    # Rendering
    fps: int = 10
    num_video_examples: int = 4
    latent_clamp: float = 5.0

    # Action ablations
    action_zero_id: int = 0

    def __post_init__(self):
        if self.rollout_horizons is None:
            # 60 is OK because ctx=4 and T=64 => predict frames [4..63] = 60 steps.
            self.rollout_horizons = [1, 2, 4, 8, 16, 32, 48, 60]
        if self.one_step_taus is None:
            self.one_step_taus = [0.0, 0.25, 0.5, 0.75]


# -----------------------------
# Dataset
# -----------------------------
class AtariWorldModelDataset(Dataset):
    def __init__(self, cfg: EvalConfig):
        print(f"[Dataset] Loading latents from {cfg.latent_path} ...")
        data = torch.load(cfg.latent_path, map_location="cpu")
        self.latents = data["z"]  # (num_frames, N, D)

        print(f"[Dataset] Loading actions from {cfg.actions_jsonl} ...")
        actions = []
        with open(cfg.actions_jsonl, "r") as f:
            for line in f:
                actions.append(json.loads(line)["action"])
        self.actions = torch.tensor(actions, dtype=torch.long)[: len(self.latents)]

        self.clip_length = cfg.clip_length
        self.stride = cfg.stride
        self.start_indices = list(range(0, len(self.latents) - self.clip_length, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.clip_length
        return {
            "latents": self.latents[start:end],  # (T,N,D)
            "actions": self.actions[start:end],  # (T,)
            "start_idx": start,                  # int
        }


# -----------------------------
# Model loading
# -----------------------------
def load_models(cfg: EvalConfig):
    device = torch.device(cfg.device)
    print(f"[Models] Loading on {device} ...")

    tokenizer = CausalTokenizer(
        input_dim=3 * cfg.patch_size * cfg.patch_size,
        embed_dim=cfg.tokenizer_dim,
        num_heads=8,
        num_layers=8,
        latent_dim=cfg.latent_dim,
        use_checkpoint=False,
    )
    tok_state = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
    tokenizer.load_state_dict({k.replace("module.", ""): v for k, v in tok_state["model_state"].items()})
    tokenizer.to(device).eval()

    builder = AtariDataBuilder(cfg)
    b_state = torch.load(cfg.builder_ckpt, map_location="cpu")
    builder.load_state_dict({k.replace("module.", ""): v for k, v in b_state.items()})
    builder.to(device).eval()

    wm = WorldModel(
        d_model=cfg.embed_dim,
        d_latent=cfg.latent_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        n_latents=cfg.n_latents,
        Sr=cfg.Sr,
        use_checkpoint=False,
    )
    wm_state = torch.load(cfg.wm_ckpt, map_location="cpu")
    wm.load_state_dict({k.replace("module.", ""): v for k, v in wm_state.items()})
    wm.to(device).eval()

    return tokenizer, builder, wm


# -----------------------------
# Utils
# -----------------------------
def corrupt_latents(z_clean: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(z_clean)
    tau_exp = tau.unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)
    return (1.0 - tau_exp) * noise + tau_exp * z_clean


@torch.no_grad()
def decode_latents(tokenizer: CausalTokenizer, latents: torch.Tensor, cfg: EvalConfig) -> torch.Tensor:
    """
    latents: (B,T,N,D) -> (B,T,H,W,3) float in [0,1]
    Uses local positional context fix.
    """
    B, T, N, D = latents.shape
    patchifier = Patchifier(cfg.patch_size)

    latents = torch.clamp(latents, -cfg.latent_clamp, cfg.latent_clamp)

    x = tokenizer.from_latent(latents)  # (B,T,N,E)
    x = x.view(B, T * N, tokenizer.embed_dim)
    x = x + tokenizer.pos_embed[:, : T * N, :]
    x = tokenizer._run_stack(x, tokenizer.decoder, T=T, N=N)
    x = x.view(B, T, N, tokenizer.embed_dim)
    patches = tokenizer.output_proj(x)

    frames = []
    for b in range(B):
        fr = patchifier.unpatchify(patches[b], cfg.resize, cfg.patch_size)  # (T,3,H,W)
        frames.append(fr)
    frames = torch.stack(frames, dim=0)  # (B,T,3,H,W)
    frames = frames.permute(0, 1, 3, 4, 2).contiguous()  # (B,T,H,W,3)
    return frames.clamp(0.0, 1.0)


def psnr_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=(-1, -2, -3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def save_video_rgb(frames_uint8: np.ndarray, path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames_uint8.shape
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(T):
        out.write(cv2.cvtColor(frames_uint8[t], cv2.COLOR_RGB2BGR))
    out.release()


# -----------------------------
# Evaluations
# -----------------------------
@torch.no_grad()
def one_step_eval(cfg: EvalConfig, tokenizer, builder, wm, loader) -> Dict:
    device = torch.device(cfg.device)
    results = {f"tau_{t}": {"latent_mse": [], "psnr": []} for t in cfg.one_step_taus}

    used = 0
    for batch in tqdm(loader, desc="Eval: one-step", leave=False):
        z_gt = batch["latents"].to(device)           # (B,T,N,D)
        a = batch["actions"].to(device)              # (B,T)
        start_idx = batch["start_idx"].to(device)    # (B,)
        B, T, N, D = z_gt.shape

        # Random target frame (must have at least 1 history frame)
        target_t = torch.randint(cfg.context_frames, T, (1,)).item()
        window_len = min(cfg.context_frames + 1, target_t + 1)
        win_start = target_t - window_len + 1

        z_hist = z_gt[:, win_start:target_t]                 # (B, window_len-1, N, D)
        z_target_clean = z_gt[:, target_t:target_t + 1]      # (B,1,N,D)
        a_window = a[:, win_start:target_t + 1]              # (B,window_len)
        offsets = start_idx + win_start
        d_window = torch.ones(B, device=device)

        for tau_val in cfg.one_step_taus:
            tau_window = torch.ones(B, window_len, device=device)
            tau_window[:, -1] = tau_val

            # corrupt ONLY the target frame at desired tau
            z_target_corr = corrupt_latents(z_target_clean, tau_window[:, -1:].clone())
            z_window = torch.cat([z_hist, z_target_corr], dim=1)

            tokens = builder(z_window, a_window, tau_window, d_window)
            pred_seq = wm({"wm_input_tokens": tokens, "tau": tau_window, "d": d_window}, time_offsets=offsets)
            pred_target = pred_seq[:, -1]  # (B,N,D)

            mse = (pred_target - z_target_clean.squeeze(1)).pow(2).mean(dim=(1, 2))
            results[f"tau_{tau_val}"]["latent_mse"].extend(mse.detach().cpu().tolist())

            pred_frame = decode_latents(tokenizer, pred_target.unsqueeze(1), cfg)[:, 0]
            gt_frame = decode_latents(tokenizer, z_target_clean, cfg)[:, 0]
            ps = psnr_torch(pred_frame, gt_frame)
            results[f"tau_{tau_val}"]["psnr"].extend(ps.detach().cpu().tolist())

        used += B

    assert used > 0, "No one-step samples evaluated — something is wrong with the loader."
    summary = {}
    for k, v in results.items():
        summary[k] = {
            "latent_mse_mean": float(np.mean(v["latent_mse"])) if len(v["latent_mse"]) else None,
            "latent_mse_std": float(np.std(v["latent_mse"])) if len(v["latent_mse"]) else None,
            "psnr_mean": float(np.mean(v["psnr"])) if len(v["psnr"]) else None,
            "psnr_std": float(np.std(v["psnr"])) if len(v["psnr"]) else None,
        }
    return summary


@torch.no_grad()
def rollout_eval(cfg: EvalConfig, tokenizer, builder, wm, loader) -> Dict:
    """
    Closed-loop rollout:
    - seed with GT context
    - predict next with tau_target=0, feed back
    - report latent MSE and PSNR vs horizon
    Also saves a few GT|Pred videos.
    """
    device = torch.device(cfg.device)
    ctx = cfg.context_frames
    max_h = max(cfg.rollout_horizons)

    horizon_mse = {h: [] for h in cfg.rollout_horizons}
    horizon_psnr = {h: [] for h in cfg.rollout_horizons}

    vids_dir = cfg.out_dir / "videos"
    vids_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    used = 0
    for batch in tqdm(loader, desc="Eval: rollout", leave=False):
        z_gt = batch["latents"].to(device)        # (B,T,N,D)
        a = batch["actions"].to(device)           # (B,T)
        start_idx = batch["start_idx"].to(device) # (B,)
        B, T, N, D = z_gt.shape

        # FIXED OFF-BY-ONE: allow ctx+max_h == T
        if ctx + max_h > T:
            continue

        z_current = z_gt[:, :ctx].clone()
        preds = []

        for step in range(max_h):
            target_t = ctx + step
            window_len = min(ctx + step + 1, T)

            noise_target = torch.randn(B, 1, N, D, device=device)
            z_window = torch.cat([z_current, noise_target], dim=1)[:, -window_len:]
            a_window = a[:, target_t - window_len + 1: target_t + 1]

            tau_window = torch.ones(B, window_len, device=device)
            tau_window[:, -1] = 0.0
            d_window = torch.ones(B, device=device)

            offsets = start_idx + (target_t - window_len + 1)

            tokens = builder(z_window, a_window, tau_window, d_window)
            pred_seq = wm({"wm_input_tokens": tokens, "tau": tau_window, "d": d_window}, time_offsets=offsets)
            pred_target = torch.clamp(pred_seq[:, -1], -cfg.latent_clamp, cfg.latent_clamp)

            preds.append(pred_target)
            z_current = torch.cat([z_current, pred_target.unsqueeze(1)], dim=1)

        pred_stack = torch.stack(preds, dim=1)     # (B,max_h,N,D)
        gt_stack = z_gt[:, ctx:ctx + max_h]        # (B,max_h,N,D)

        # Metrics
        for h in cfg.rollout_horizons:
            pred_h = pred_stack[:, :h]
            gt_h = gt_stack[:, :h]

            mse = (pred_h - gt_h).pow(2).mean(dim=(1, 2, 3))
            horizon_mse[h].extend(mse.detach().cpu().tolist())

            pred_last = pred_h[:, -1:].contiguous()
            gt_last = gt_h[:, -1:].contiguous()
            pred_frame = decode_latents(tokenizer, pred_last, cfg)[:, 0]
            gt_frame = decode_latents(tokenizer, gt_last, cfg)[:, 0]
            ps = psnr_torch(pred_frame, gt_frame)
            horizon_psnr[h].extend(ps.detach().cpu().tolist())

        # Save a few videos: GT | Pred (side-by-side)
        if saved < cfg.num_video_examples:
            b = 0
            z_pred_full = torch.cat([z_gt[b:b+1, :ctx], pred_stack[b:b+1]], dim=1)  # (1,ctx+max_h,...)
            z_gt_full = z_gt[b:b+1, :ctx + max_h]

            frames_gt = decode_latents(tokenizer, z_gt_full, cfg)[0].cpu().numpy()
            frames_pr = decode_latents(tokenizer, z_pred_full, cfg)[0].cpu().numpy()

            gt_u = (frames_gt * 255.0).astype(np.uint8)
            pr_u = (frames_pr * 255.0).astype(np.uint8)
            grid = np.concatenate([gt_u, pr_u], axis=2)  # width concat

            save_video_rgb(grid, vids_dir / f"rollout_gt_vs_pred_{saved:02d}.mp4", cfg.fps)
            saved += 1

        used += B

    assert used > 0, (
        f"No rollout samples evaluated. Check clip_length={cfg.clip_length}, "
        f"context_frames={ctx}, max_horizon={max_h}."
    )

    summary = {
        "latent_mse_by_horizon": {str(h): float(np.mean(horizon_mse[h])) if len(horizon_mse[h]) else None
                                 for h in cfg.rollout_horizons},
        "psnr_by_horizon": {str(h): float(np.mean(horizon_psnr[h])) if len(horizon_psnr[h]) else None
                            for h in cfg.rollout_horizons},
        "videos_saved": saved,
        "video_layout": "GT | PRED",
    }
    return summary


@torch.no_grad()
def action_sanity_eval(cfg: EvalConfig, tokenizer, builder, wm, loader) -> Dict:
    """
    Rollout with:
      - true actions
      - zero actions
      - random actions
    Saves videos: GT | TRUE | ZERO | RAND
    Reports divergence in latent space wrt TRUE rollout.
    """
    device = torch.device(cfg.device)
    ctx = cfg.context_frames
    max_h = max(cfg.rollout_horizons)

    vids_dir = cfg.out_dir / "videos"
    vids_dir.mkdir(parents=True, exist_ok=True)

    divergences_zero = []
    divergences_rand = []
    saved = 0
    used = 0

    def rollout(z_gt: torch.Tensor, a_seq: torch.Tensor, start_idx: torch.Tensor) -> torch.Tensor:
        B, T, N, D = z_gt.shape
        z_cur = z_gt[:, :ctx].clone()
        preds = []
        for step in range(max_h):
            target_t = ctx + step
            window_len = min(ctx + step + 1, T)

            noise_target = torch.randn(B, 1, N, D, device=device)
            z_window = torch.cat([z_cur, noise_target], dim=1)[:, -window_len:]
            a_window = a_seq[:, target_t - window_len + 1: target_t + 1]

            tau_window = torch.ones(B, window_len, device=device)
            tau_window[:, -1] = 0.0
            d_window = torch.ones(B, device=device)

            offsets = start_idx + (target_t - window_len + 1)

            tokens = builder(z_window, a_window, tau_window, d_window)
            pred_seq = wm({"wm_input_tokens": tokens, "tau": tau_window, "d": d_window}, time_offsets=offsets)
            pred = torch.clamp(pred_seq[:, -1], -cfg.latent_clamp, cfg.latent_clamp)
            preds.append(pred)
            z_cur = torch.cat([z_cur, pred.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)  # (B,max_h,N,D)

    for batch in tqdm(loader, desc="Eval: action sanity", leave=False):
        z_gt = batch["latents"].to(device)
        a_true = batch["actions"].to(device)
        start_idx = batch["start_idx"].to(device)
        B, T, N, D = z_gt.shape

        # FIXED OFF-BY-ONE: allow ctx+max_h == T
        if ctx + max_h > T:
            continue

        a_zero = torch.full_like(a_true, cfg.action_zero_id)
        a_rand = torch.randint(0, cfg.action_dim, a_true.shape, device=device)

        pred_true = rollout(z_gt, a_true, start_idx)
        pred_zero = rollout(z_gt, a_zero, start_idx)
        pred_rand = rollout(z_gt, a_rand, start_idx)

        div_zero = (pred_zero - pred_true).pow(2).mean(dim=(1, 2, 3))
        div_rand = (pred_rand - pred_true).pow(2).mean(dim=(1, 2, 3))
        divergences_zero.extend(div_zero.detach().cpu().tolist())
        divergences_rand.extend(div_rand.detach().cpu().tolist())

        # Save videos
        if saved < cfg.num_video_examples:
            b = 0
            z_gt_full = z_gt[b:b+1, :ctx + max_h]
            z_true_full = torch.cat([z_gt[b:b+1, :ctx], pred_true[b:b+1]], dim=1)
            z_zero_full = torch.cat([z_gt[b:b+1, :ctx], pred_zero[b:b+1]], dim=1)
            z_rand_full = torch.cat([z_gt[b:b+1, :ctx], pred_rand[b:b+1]], dim=1)

            frames_gt = decode_latents(tokenizer, z_gt_full, cfg)[0].cpu().numpy()
            frames_tr = decode_latents(tokenizer, z_true_full, cfg)[0].cpu().numpy()
            frames_ze = decode_latents(tokenizer, z_zero_full, cfg)[0].cpu().numpy()
            frames_ra = decode_latents(tokenizer, z_rand_full, cfg)[0].cpu().numpy()

            gt_u = (frames_gt * 255.0).astype(np.uint8)
            tr_u = (frames_tr * 255.0).astype(np.uint8)
            ze_u = (frames_ze * 255.0).astype(np.uint8)
            ra_u = (frames_ra * 255.0).astype(np.uint8)

            grid = np.concatenate([gt_u, tr_u, ze_u, ra_u], axis=2)
            save_video_rgb(grid, vids_dir / f"action_sanity_{saved:02d}.mp4", cfg.fps)
            saved += 1

        used += B

        # Stop once we've got enough stats + videos
        if used >= cfg.num_eval_clips and saved >= cfg.num_video_examples:
            break

    assert used > 0, "No action sanity samples evaluated — horizon/clip_length mismatch."
    summary = {
        "divergence_zero_mean": float(np.mean(divergences_zero)) if len(divergences_zero) else None,
        "divergence_zero_std": float(np.std(divergences_zero)) if len(divergences_zero) else None,
        "divergence_random_mean": float(np.mean(divergences_rand)) if len(divergences_rand) else None,
        "divergence_random_std": float(np.std(divergences_rand)) if len(divergences_rand) else None,
        "videos_saved": saved,
        "video_layout": "GT | TRUE_ACTIONS | ZERO_ACTIONS | RANDOM_ACTIONS",
    }
    return summary


def try_plot(cfg: EvalConfig, rollout_summary: Dict):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Plot] matplotlib not available, skipping plots.")
        return

    plots_dir = cfg.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mse = rollout_summary["latent_mse_by_horizon"]
    psnr = rollout_summary["psnr_by_horizon"]

    hs = [int(h) for h in mse.keys()]
    mse_vals = [mse[str(h)] for h in hs]
    psnr_vals = [psnr[str(h)] for h in hs]

    plt.figure()
    plt.plot(hs, mse_vals, marker="o")
    plt.xlabel("Horizon (steps)")
    plt.ylabel("Latent MSE (mean up to horizon)")
    plt.title("Closed-loop Rollout Error vs Horizon")
    plt.grid(True)
    plt.savefig(plots_dir / "rollout_latent_mse.png")
    plt.close()

    plt.figure()
    plt.plot(hs, psnr_vals, marker="o")
    plt.xlabel("Horizon (steps)")
    plt.ylabel("PSNR (dB) on last frame at horizon")
    plt.title("Closed-loop Rollout PSNR vs Horizon")
    plt.grid(True)
    plt.savefig(plots_dir / "rollout_psnr.png")
    plt.close()

    print(f"[Plot] Saved plots to {plots_dir}")


def main():
    cfg = EvalConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, builder, wm = load_models(cfg)

    dataset = AtariWorldModelDataset(cfg)

    # Deterministic spread across dataset for speed
    n = min(cfg.num_eval_clips, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num=n).astype(int).tolist()
    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    metrics = {}
    metrics["one_step"] = one_step_eval(cfg, tokenizer, builder, wm, loader)
    metrics["rollout"] = rollout_eval(cfg, tokenizer, builder, wm, loader)
    metrics["action_sanity"] = action_sanity_eval(cfg, tokenizer, builder, wm, loader)

    out_path = cfg.out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Done] Wrote metrics to {out_path}")

    try_plot(cfg, metrics["rollout"])

    print("\nInterpretation (fast):")
    print("- one_step PSNR ~ mid-30s is decent.")
    print("- rollout PSNR should decay with horizon but not instantly crater by ~8.")
    print("- action_sanity divergences should be > 0. If ~0, model ignores actions.")
    print(f"- videos saved under: {cfg.out_dir / 'videos'}")


if __name__ == "__main__":
    main()