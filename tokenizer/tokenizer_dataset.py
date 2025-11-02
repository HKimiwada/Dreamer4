"""
End-to-End data source for training the Dreamer 4 causal tokenizer.

TokenizerDataset — Puts together Stage A (VideoLoader), Stage B (TemporalSlicer),
and Stage C (Patchifier + MaskGenerator) into a single iterable that yields training examples.
"""
from typing import Iterator, Dict, Any, Optional
from pathlib import Path
import torch

from tokenizer.dataset import VideoLoader          # Stage A
from tokenizer.temporal_slicer import TemporalSlicer  # Stage B
from tokenizer.patchify_mask import Patchifier     # Stage C (your Patchifier)
from tokenizer.patchify_mask import MaskGenerator        # Stage C (mask only)

class TokenizerDataset:
    def __init__(
        self,
        video_dir: Path,
        target_fps: float = 20.0,
        resize = (384, 640),
        max_frames_loader: Optional[int] = None,
        clip_length: int = 64,
        stride: Optional[int] = None,
        mode: str = "random",               # "random" for training, "sequential" for pre-cache
        drop_last: bool = True,
        patch_size: int = 16,
        mask_prob_range = (0.0, 0.9),
        per_frame_mask_sampling: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.loader = VideoLoader(video_dir=video_dir, target_fps=target_fps, resize=resize, max_frames=max_frames_loader)
        self.slicer = TemporalSlicer(self.loader, clip_length=clip_length, stride=stride, mode=mode, drop_last=drop_last)
        self.patchifier = Patchifier(patch_size=patch_size)
        self.masker = MaskGenerator(mask_prob_range=mask_prob_range, per_frame_sampling=per_frame_mask_sampling)
        self.device = device if device is not None else torch.device("cpu")

        # Derived sizes
        H, W = resize
        assert H % patch_size == 0 and W % patch_size == 0
        self.num_patches_per_frame = (H // patch_size) * (W // patch_size)
        self.frame_size = (H, W)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Yields dictionaries with:
          - 'patch_tokens': (T, N, D) float32 in [0,1]
          - 'mask': (T, N) bool
          - 'meta': dict with video_id, start, end, etc.
        """
        for sample in self.slicer.generate_all():
            frames: torch.Tensor = sample["frames"].to(self.device)  # (T, 3, H, W), [0,1]
            meta: Dict[str, Any] = sample["metadata"]

            # Stage C: patchify
            patches = self.patchifier(frames)  # (T, N, D)

            T, N, D = patches.shape
            assert N == self.num_patches_per_frame, f"expected {self.num_patches_per_frame}, got {N}"

            # Stage C: mask (no replacement here; model will inject learned mask token)
            mask = self.masker(T=T, N=N, device=self.device)  # (T, N) bool

            yield {
                "patch_tokens": patches,   # (T, N, D) in [0,1]
                "mask": mask,             # (T, N) bool
                "meta": {
                    **meta,
                    "T": T,
                    "N": N,
                    "D": D,
                    "frame_size": self.frame_size,
                    "patch_size": self.patchifier.patch_size,
                },
            }
