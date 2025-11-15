"""
End-to-End data source for training the Dreamer 4 causal tokenizer.

TokenizerDataset — Puts together Stage A (VideoLoader), Stage B (TemporalSlicer),
and Stage C (Patchifier + MaskGenerator) into a single iterable that yields training examples.
"""
from typing import Iterator, Dict, Any, Optional
from pathlib import Path
import torch
import shutil
import tempfile

from tokenizer.dataset import VideoLoader          # Stage A
from tokenizer.temporal_slicer import TemporalSlicer  # Stage B
from tokenizer.patchify_mask import Patchifier     # Stage C (your Patchifier)
from tokenizer.patchify_mask import MaskGenerator        # Stage C (mask only)
from torch.utils.data import Dataset

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

    def __len__(self):
        return len(self.samples)

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

class TokenizerDatasetWM:
    """
    Version of TokenizerDataset that loads **exactly one video file**.

    Internally creates a temporary directory and copies the video into it,
    because VideoLoader only accepts `video_dir` and loads all *.mp4 in that directory.
    """

    def __init__(
        self,
        video_file: Path,
        target_fps: float = 20.0,
        resize=(384, 640),
        max_frames_loader: Optional[int] = None,
        clip_length: int = 8,
        stride: Optional[int] = None,
        mode: str = "sequential",
        drop_last: bool = False,
        patch_size: int = 16,
        mask_prob_range=(0.0, 0.0),
        per_frame_mask_sampling: bool = False,
        device: Optional[torch.device] = None,
    ):
        video_file = Path(video_file)
        assert video_file.exists(), f"Video file not found: {video_file}"

        # ---------------------------------------------------------
        # Create a temporary folder containing ONLY this video
        # ---------------------------------------------------------
        self.temp_dir = Path(tempfile.mkdtemp(prefix="tokds_"))
        self.single_video_path = self.temp_dir / video_file.name
        shutil.copy(video_file, self.single_video_path)

        # ---------------------------------------------------------
        # Load using the existing VideoLoader(video_dir=...)
        # ---------------------------------------------------------
        self.loader = VideoLoader(
            video_dir=self.temp_dir,
            target_fps=target_fps,
            resize=resize,
            max_frames=max_frames_loader,
        )

        self.slicer = TemporalSlicer(
            self.loader,
            clip_length=clip_length,
            stride=stride,
            mode=mode,
            drop_last=drop_last,
        )

        self.patchifier = Patchifier(patch_size=patch_size)
        self.masker = MaskGenerator(
            mask_prob_range=mask_prob_range,
            per_frame_sampling=per_frame_mask_sampling
        )

        self.device = device if device is not None else torch.device("cpu")

        H, W = resize
        assert H % patch_size == 0 and W % patch_size == 0
        self.num_patches_per_frame = (H // patch_size) * (W // patch_size)
        self.frame_size = (H, W)

        print(f"[TokenizerDatasetWM] Using temp dir {self.temp_dir}")
        print(f"[TokenizerDatasetWM] Single video: {self.single_video_path}")

    def __iter__(self):
        for sample in self.slicer.generate_all():
            frames = sample["frames"].to(self.device)
            meta = sample["metadata"]

            patches = self.patchifier(frames)
            T, N, D = patches.shape
            assert N == self.num_patches_per_frame

            mask = self.masker(T=T, N=N, device=self.device)

            yield {
                "patch_tokens": patches,
                "mask": mask,
                "meta": {
                    **meta,
                    "T": T,
                    "N": N,
                    "D": D,
                    "frame_size": self.frame_size,
                },
            }

class TokenizerDatasetDDP(Dataset):
    """
    DDP-compatible dataset for DreamerV4 tokenizer training.
    Each __getitem__ dynamically loads one clip, patchifies, and masks.
    """

    def __init__(
        self,
        video_dir: Path,
        target_fps: float = 20.0,
        resize=(384, 640),
        max_frames_loader: Optional[int] = None,
        clip_length: int = 64,
        stride: Optional[int] = None,
        mode: str = "random",
        drop_last: bool = True,
        patch_size: int = 16,
        mask_prob_range=(0.0, 0.9),
        per_frame_mask_sampling: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.video_dir = video_dir
        self.target_fps = target_fps
        self.resize = resize
        self.clip_length = clip_length
        self.stride = stride
        self.mode = mode
        self.drop_last = drop_last
        self.patch_size = patch_size
        self.device = device if device is not None else torch.device("cpu")

        # Core components
        self.loader = VideoLoader(video_dir=video_dir, target_fps=target_fps, resize=resize, max_frames=max_frames_loader)
        self.slicer = TemporalSlicer(self.loader, clip_length=clip_length, stride=stride, mode=mode, drop_last=drop_last)
        self.patchifier = Patchifier(patch_size=patch_size)
        self.masker = MaskGenerator(mask_prob_range=mask_prob_range, per_frame_sampling=per_frame_mask_sampling)

        # Instead of loading frames, only store slice metadata
        self.samples = []
        for vid_path in self.loader.video_paths:
            self.samples.append({"video_path": vid_path})

        # Derived sizes
        H, W = resize
        assert H % patch_size == 0 and W % patch_size == 0
        self.num_patches_per_frame = (H // patch_size) * (W // patch_size)
        self.frame_size = (H, W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]
        path = sample_info["video_path"]

        # --- Lazy load frames from disk ---
        frames, meta = self.loader.__getitem__(idx)
        frames = frames.to(self.device)

        # --- Slice clip to fixed length ---
        # TemporalSlicer returns generator; take one random clip
        for clip in self.slicer.slice_video(frames, meta):
            frames = clip["frames"]
            meta = clip["metadata"]
            break  # only first clip per video for simplicity

        # --- Patchify and mask ---
        patches = self.patchifier(frames)
        T, N, D = patches.shape
        mask = self.masker(T=T, N=N, device=self.device)

        return {
            "patch_tokens": patches,
            "mask": mask,
            "meta": {
                **meta,
                "T": T,
                "N": N,
                "D": D,
                "frame_size": self.frame_size,
                "patch_size": self.patchifier.patch_size,
                "video_path": str(path)
            },
        }
