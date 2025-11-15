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

from tokenizer.model.encoder_decoder import CausalTokenizer2
from tokenizer.patchify_mask import Patchifier
from tokenizer.tokenizer_dataset import TokenizerDatasetWM

class InferenceConfig:
    video_file = "v1_video.mp4"

    data_dir = Path("data")
    ckpt_path = Path("checkpoints/tokenizer/complete_overfit_mse/v1_weights.pt")

    # Model / dataset params (must match training)
    resize = (256, 448)
    patch_size = 16
    clip_length = 8
    input_dim = 3 * patch_size * patch_size
    embed_dim = 512
    latent_dim = 256
    num_heads = 16
    num_layers = 18
    device = "cuda" if torch.cuda.is_available() else "cpu"

class TokenizerWrapper(nn.Module):
    """
    Wrap tokenizer to expose latents.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model = CausalTokenizer2(
            input_dim=cfg.input_dim,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            latent_dim=cfg.latent_dim,
            use_checkpoint=False,
        )
        ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
        state = ckpt["model_state"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to("cuda").eval()
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_latents(self, patches):
        """
        patches: (B, T, N, D_in)
        returns latents: (B, T, N, latent_dim)
        """
        B, T, N, D_in = patches.shape

        # project to embed_dim
        x = self.model.input_proj(patches)  # (B, T, N, embed_dim)

        # flatten time and patches
        x = x.view(B, T * N, self.model.embed_dim)

        # add positional embeddings
        seq_len = T * N
        x = x + self.model.pos_embed[:, :seq_len, :]

        # run encoder stack
        x = self.model._run_stack(x, self.model.encoder, T, N)

        # reshape for bottleneck
        x = x.view(B, T, N, self.model.embed_dim)

        z = self.model.to_latent(x)

        return z

    def load_video_dataset(self):
        print(f"[Dataset] Loading ONLY video: {self.cfg.video_file}")
        dataset = TokenizerDatasetWM(
            video_file=self.cfg.data_dir / self.cfg.video_file,
            target_fps=20.0,
            resize=self.cfg.resize,
            clip_length=self.cfg.clip_length,
            stride=self.cfg.clip_length,
            mode="sequential",
            drop_last=False,
            patch_size=self.cfg.patch_size,
            mask_prob_range=(0.0, 0.0),
            per_frame_mask_sampling=False,
            device=torch.device("cpu"),
        )
        print("✓ Dataset ready (single video, sequential clips)")
        return dataset
    
    def export_latents(self, cfg, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        dataset = self.load_video_dataset()
        patchifier = Patchifier(cfg.patch_size)

        print("[Export] Starting latent generation...")

        for clip_idx, sample in enumerate(dataset):
            patches = sample["patch_tokens"].unsqueeze(0).to(cfg.device)  # (1, T, N, D)
            
            # Run tokenizer encoder directly on patches
            z = self.encode_latents(patches)  # (1, T, N, latent_dim)
            z = z.squeeze(0).cpu()

            out_path = Path(output_dir) / f"video_{clip_idx:05d}.pt"
            torch.save({"z": z, "length": z.shape[0]}, out_path)
            print(f"[Saved] {out_path} | shape={tuple(z.shape)}")

    
if __name__ == "__main__":
    cfg = InferenceConfig()
    output_dir = "data/latent_sequences/"
    latent_creator = TokenizerWrapper(cfg)
    latent_creator.export_latents(cfg, output_dir)

