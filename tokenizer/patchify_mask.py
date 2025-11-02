"""
Data preprocessing pipeline: convert raw VPT gameplay into tensor clips that can be used to train causal tokenizer. 
Overview:
    1. Load raw VPT gameplay data from disk and convert to tensors -> dataset.VideoLoader
    2. Stadardize tensors (resize, normalize, clip into sequences): clip into sequence -> temporal_slicer.TemporalSlicer
    3. Patchify and mask frames for masked-autoencoding training.
    4. Store or Stream batches efficiently for tokenizer.

Classes:
    Patchifier: Converts a batch of video frames (T, 3, H, W) into flattened patch tokens (T, N, D).
        Converts tensors for each frame into non-overlapping patches (splits one frame into 960 patches) and flattens them.
    
"""
from typing import Tuple
import torch
import torch.nn.functional as F
from einops import rearrange

from typing import Tuple, Optional
import torch

class Patchifier:
    def __init__(self, patch_size: int = 16):
        """
        Args:
            patch_size: size of each square patch (pixels).
                        16 is used in Dreamer 4 for 384×640 frames.
        """
        self.patch_size = patch_size

    # --------------------------------------------------------------

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: torch.Tensor of shape (T, 3, H, W),
                    pixel values already in [0, 1].

        Returns:
            patches: torch.Tensor of shape (T, N, D)
                     where N = (H/ps)*(W/ps),  D = 3*ps*ps
        """
        T, C, H, W = frames.shape
        ps = self.patch_size

        # sanity checks
        if H % ps != 0 or W % ps != 0:
            raise ValueError(f"Frame size ({H}, {W}) not divisible by patch size {ps}")

        # split each frame into non-overlapping 16×16 patches
        # unfold: (T, C, H, W) → (T, C, num_patches, ps, ps)
        patches = frames.unfold(2, ps, ps).unfold(3, ps, ps)
        # rearrange to (T, N, C, ps, ps)
        patches = rearrange(patches, "t c nh nw ps1 ps2 -> t (nh nw) c ps1 ps2")
        # flatten each patch to vector D = C × ps × ps
        patches = rearrange(patches, "t n c h w -> t n (c h w)")
        return patches

    # --------------------------------------------------------------

    @staticmethod
    def unpatchify(patches: torch.Tensor, frame_size: Tuple[int, int], patch_size: int = 16) -> torch.Tensor:
        """
        Reconstruct frames from flattened patch tokens (for debugging or visualization).

        Args:
            patches: (T, N, D)
            frame_size: (H, W)
            patch_size: same as used in patchify()

        Returns:
            frames: torch.Tensor (T, 3, H, W)
        """
        T, N, D = patches.shape
        H, W = frame_size
        ps = patch_size
        C = 3
        num_h = H // ps
        num_w = W // ps
        # reshape back to patches
        patches = rearrange(patches, "t (nh nw) (c h w) -> t c nh nw h w",
                            nh=num_h, nw=num_w, c=C, h=ps, w=ps)
        # merge patches into full frames
        frames = rearrange(patches, "t c nh nw h w -> t c (nh h) (nw w)")
        return frames

class MaskGenerator:
    def __init__(
        self,
        mask_prob_range: Tuple[float, float] = (0.0, 0.9),
        per_frame_sampling: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            mask_prob_range: (lo, hi), sample p ~ Uniform(lo, hi)
            per_frame_sampling: if True, sample a separate p for each frame;
                                if False, one p per clip (applied to all frames).
            seed: optional RNG seed for reproducibility.
        """
        self.lo, self.hi = mask_prob_range
        assert 0.0 <= self.lo <= self.hi <= 1.0
        self.per_frame_sampling = per_frame_sampling
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    @torch.no_grad()
    def __call__(self, T: int, N: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Produce a binary mask of shape (T, N), where 1 = masked, 0 = visible.
        """
        if self.per_frame_sampling:
            # different p for each frame
            p = torch.empty(T, device=device).uniform_(self.lo, self.hi, generator=self.generator)
            # for each frame t, sample N Bernoulli(p[t])
            mask = torch.bernoulli(p.unsqueeze(1).expand(T, N), generator=self.generator)
        else:
            # one p for the whole clip
            p = float(torch.empty(1, device=device).uniform_(self.lo, self.hi, generator=self.generator).item())
            mask = torch.bernoulli(torch.full((T, N), p, device=device), generator=self.generator)

        return mask.to(dtype=torch.bool)