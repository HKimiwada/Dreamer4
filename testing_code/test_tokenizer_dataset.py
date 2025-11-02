# python testing_code/test_tokenizer_dataset.py
from pathlib import Path
import torch
from tokenizer.tokenizer_dataset import TokenizerDataset

video_dir = Path("data")  # your downloaded .mp4s
ds = TokenizerDataset(
    video_dir=video_dir,
    target_fps=20.0,
    resize=(384, 640),
    max_frames_loader=None,   # let loader read full video (it will still slice to 64)
    clip_length=64,
    stride=64,
    mode="sequential",        # use "random" during actual training
    patch_size=16,
    mask_prob_range=(0.0, 0.9),
    per_frame_mask_sampling=True,
)

it = iter(ds)
for _ in range(2):
    batch = next(it)
    x = batch["patch_tokens"]  # (T, N, D)
    m = batch["mask"]          # (T, N)
    meta = batch["meta"]

    print("patch_tokens:", x.shape, x.dtype, f"[{x.min().item():.3f}, {x.max().item():.3f}]")
    print("mask:", m.shape, m.dtype, "masked_frac≈", m.float().mean().item())
    print("meta:", {k: meta[k] for k in ("video_id", "start", "end", "T", "N", "D")})
