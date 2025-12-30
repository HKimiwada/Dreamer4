# python world_model/wm_preprocessing/latent_tokenizer.py
# Freeze tokenizer & Create latent dataset
# Generate latent sequences z1,...,zT for input videos. 
# World Model = Causal ViT trained on these WM with short-cut forcing. 
# Latent Token: (T, N, D) -> T is the number of frames per clip, N is the number of tokens per frame, D is the dimension of the latent token
# Currently: (8, 448, 256)
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
import os
from pathlib import Path

from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier
from tokenizer.tokenizer_dataset import TokenizerDatasetWM

import tqdm

# ---------------------------------------------------------------------------
class AtariConfig:
    # Dataset paths
    data_dir = Path("data/atari/raw")
    ckpt_path = Path("checkpoints/atari/tokenizer_v2/best_model.pt")
    output_dir = Path("data/atari/latent_sequences")
    
    # Model architecture (Optimized for 64x64)
    resize = (64, 64)       
    patch_size = 8         
    input_dim = 3 * patch_size * patch_size
    embed_dim = 256
    latent_dim = 256
    num_heads = 8           
    num_layers = 8         
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
class AtariLatentDataset(torch.utils.data.Dataset):
    """Simple dataset to iterate through your 100k .pt frames."""
    def __init__(self, data_dir, patch_size=16):
        self.data_dir = Path(data_dir)
        self.frame_paths = sorted(list(self.data_dir.glob("frame_*.pt")))
        self.patchifier = Patchifier(patch_size=patch_size)
        
        if not self.frame_paths:
            raise RuntimeError(f"No .pt frames found in {data_dir}")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        # Load (C, H, W) byte tensor and normalize
        f = torch.load(self.frame_paths[idx], map_location="cpu").float() / 255.0
        # Convert to patches (1, N, D)
        patches = self.patchifier(f.unsqueeze(0)) 
        return patches.squeeze(0) # (N, D)

# ---------------------------------------------------------------------------
class TokenizerWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = CausalTokenizer(
            input_dim=cfg.input_dim,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            latent_dim=cfg.latent_dim,
            use_checkpoint=False,
        )
        
        # Load weights and strip 'module.' if trained with DDP
        ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        
        self.model.load_state_dict(state)
        self.model.to(cfg.device).eval()
        
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_latents(self, patches):
        """patches: (B, T, N, D) -> returns (B, T, N, latent_dim)"""
        B, T, N, D = patches.shape
        x = self.model.input_proj(patches)
        x = x.view(B, T * N, self.model.embed_dim)
        x = x + self.model.pos_embed[:, :T*N, :]
        x = self.model._run_stack(x, self.model.encoder, T, N)
        x = x.view(B, T, N, self.model.embed_dim)
        return self.model.to_latent(x)

    def export_all_latents(self, cfg):
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        dataset = AtariLatentDataset(cfg.data_dir, cfg.patch_size)
        # Process in batches for speed (e.g., 64 frames at once)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        
        all_z = []
        print(f"[Export] Processing {len(dataset)} frames...")

        for batch in tqdm.tqdm(loader):
            # batch is (B, N, D). Add temporal dimension T=1 for the encoder
            patches = batch.unsqueeze(1).to(cfg.device) # (B, 1, N, D)
            z = self.encode_latents(patches) # (B, 1, N, latent_dim)
            all_z.append(z.squeeze(1).cpu()) # Store as (B, N, latent_dim)

        # Concatenate everything into one long sequence
        full_seq = torch.cat(all_z, dim=0) # (100000, N, latent_dim)
        out_path = cfg.output_dir / "video_full_frames.pt"
        torch.save({"z": full_seq, "length": full_seq.shape[0]}, out_path)
        print(f"✓ Saved latent sequence: {out_path} | Shape: {full_seq.shape}")

if __name__ == "__main__":
    config = AtariConfig()
    wrapper = TokenizerWrapper(config)
    wrapper.export_all_latents(config)