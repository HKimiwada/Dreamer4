"""
Data preprocessing pipeline: convert raw VPT gameplay into tensor clips that can be used to train causal tokenizer. 
Overview:
    1. Load raw VPT gameplay data from disk.
    2. Stadardize data (resize, normalize, clip into sequences)
    3. Patchify and mask frames for masked-autoencoding training.
    4. Store or Stream batches efficiently for tokenizer.

Classes:
    VideoLoader: Load and preprocess video clips from raw .mp4 files.
"""
import os
from pathlib import Path
import numpy as np
import torch
from decord import VideoReader, cpu
import torchvision.transforms.functional as F  # optional
from PIL import Image

class VideoLoader:
    """
    Input raw mp4 video files from a directory, sample frames at target fps, resize frames, and return as tensor clips.
    [Frame, 3 (number of channels RGB), Height, Width]

    Video 1:
        Frames shape: torch.Size([64, 3, 384, 640])
        Metadata: {'video_id': 'cheeky-cornflower-setter-8106e54c1f1d-20220420-205328', 'orig_frame_count': 64, 'clip_length': 64}
    Testing loader with max_frames=64 for video index 2
    """

    def __init__(self, 
                video_dir: Path, 
                target_fps: float = 20.0, 
                resize: tuple = (384, 640), 
                max_frames: int = None):
        """
        :param video_dir: Path to directory containing .mp4 video files.
        :param target_fps: Desired frames per second to sample / downsample to.
        :param resize: (H, W) tuple to resize each frame to.
        :param max_frames: Optional maximum number of frames to return per video. (64 and 192 frames were used in the paper)
        """
        self.video_dir = video_dir
        self.target_fps = target_fps
        self.resize = resize
        self.max_frames = max_frames
        
        self.video_paths = list(self.video_dir.glob("*.mp4"))
        if not self.video_paths:
            raise RuntimeError(f"No .mp4 files found in {self.video_dir}")
    
    def __len__(self):
        return len(self.video_paths)
    
    def _read_video(self, path: Path):
        vr = VideoReader(str(path), ctx=cpu(0))
        # total frames
        total_frames = len(vr)
        # original fps
        orig_fps = vr.get_avg_fps()  # may not always be available
        # compute step to sample to target_fps
        if orig_fps and orig_fps > 0:
            step = max(1, int(round(orig_fps / self.target_fps)))
        else:
            step = 1
        
        indices = list(range(0, total_frames, step))
        if self.max_frames:
            indices = indices[:self.max_frames]
        
        # fetch batch of frames
        frames = vr.get_batch(indices).asnumpy()  # shape (T, H_orig, W_orig, 3)
        return frames  # uint8 or uint as per library
    
    def _process_frames(self, frames: np.ndarray):
        """
        :param frames: numpy array shape (T, H_orig, W_orig, 3) -> T = Number of frames, H = Frame Height, W = Frame Width, C = Number of color channels
        :returns: torch.Tensor shape (T, 3, H, W), float32 in [0, 1] 
        """
        T, H0, W0, C = frames.shape
        out_frames = []
        for i in range(T):
            img = Image.fromarray(frames[i])
            img = img.convert("RGB")
            img = img.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
            arr = F.to_tensor(img)  # shape (3, H, W), float32 [0,1]
            out_frames.append(arr)
        out = torch.stack(out_frames, dim=0)  # shape (T,3,H,W)
        return out
    
    def __getitem__(self, idx: int):
        """
        Returns:
          frames_tensor: torch.Tensor (T, 3, H, W)
          metadata: dict with keys: video_id, orig_fps, total_frames, clip_length
        """
        path = self.video_paths[idx]
        video_id = path.stem
        frames_np = self._read_video(path)
        frames_t = self._process_frames(frames_np)
        metadata = {
            "video_id": video_id,
            "orig_frame_count": frames_np.shape[0],
            "clip_length": frames_t.shape[0]
        }
        return frames_t, metadata

if __name__ == "__main__":
    pass