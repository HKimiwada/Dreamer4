# python world_model/wm_preprocessing/latent_tokenizer.py
# Freeze tokenizer & Create latent dataset
# Generate latent sequences z1,...,zT for input videos. 
# World Model = Causal ViT trained on these WM with short-cut forcing. 
# Latent Token: (T, N, D) -> T is the number of frames per clip, N is the number of tokens per frame, D is the dimension of the latent token
# Currently: (8, 448, 256)
# world_model/wm_preprocessing/latent_tokenizer_atari.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm

from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier

# ---------------------------------------------------------------------------
class AtariConfig:
    data_dir = Path("data/atari/raw")
    ckpt_path = Path("checkpoints/atari/tokenizer_v3/best_model.pt")
    output_dir = Path("data/atari/latent_sequences")
    
    resize = (64, 64)       
    patch_size = 8         
    input_dim = 3 * patch_size * patch_size
    embed_dim = 256
    latent_dim = 256
    num_heads = 8           
    num_layers = 8         
    clip_length = 64 
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
class AtariLatentDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, patch_size=8):
        self.data_dir = Path(data_dir)
        self.frame_paths = sorted(list(self.data_dir.glob("frame_*.pt")))
        self.patchifier = Patchifier(patch_size=patch_size)
        
    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        f = torch.load(self.frame_paths[idx], map_location="cpu").float() / 255.0
        return self.patchifier(f.unsqueeze(0)).squeeze(0)

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
        
        print(f"Loading Tokenizer V3 weights from {cfg.ckpt_path}...")
        ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(cfg.device).eval()

    @torch.no_grad()
    def encode_latents(self, patches):
        """Helper to run the full encoding pipeline with temporal context."""
        B, T, N, D = patches.shape
        # 1. Project to embedding space
        x = self.model.input_proj(patches)
        x = x.view(B, T * N, self.model.embed_dim)
        
        # 2. Add Positional Embeddings (Correct slicing for T*N)
        x = x + self.model.pos_embed[:, :T * N, :]
        
        # 3. Run Encoder Transformer Stack
        x = self.model._run_stack(x, self.model.encoder, T, N)
        
        # 4. To Latent Bottleneck
        x = x.view(B, T, N, self.model.embed_dim)
        return self.model.to_latent(x)

    @torch.no_grad()
    def export_all_latents(self, cfg):
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        dataset = AtariLatentDataset(cfg.data_dir, cfg.patch_size)
        
        all_z = []
        L = len(dataset)
        C = cfg.clip_length
        
        print(f"[Export] Encoding {L} frames using SLIDING WINDOW (T={C})")
        print("This ensures maximum temporal context for every frame.")

        # 1. Handle the first window (0 to 63)
        # We store all 64 latents from this initial pass.
        first_frames = [dataset[i] for i in range(C)]
        patches = torch.stack(first_frames).unsqueeze(0).to(cfg.device)
        z_initial = self.encode_latents(patches) # (1, 64, N, Latent_Dim)
        all_z.append(z_initial.squeeze(0).cpu()) # Store all 64

        # 2. Sliding Window for remaining frames
        # For every frame from 64 onwards, we look back 63 frames.
        # We only keep the very last latent produced.
        for i in tqdm.tqdm(range(C, L), desc="Sliding Context"):
            # Load the 64-frame window ending at i
            # Optimization note: In a real production environment, you'd use a 
            # rolling buffer here, but for 100k frames, this is safer.
            window = [dataset[j] for j in range(i - C + 1, i + 1)]
            patches = torch.stack(window).unsqueeze(0).to(cfg.device)
            
            z_window = self.encode_latents(patches) # (1, 64, N, Latent_Dim)
            
            # We ONLY want the latent for the current frame (the last one in the window)
            # This frame has the "perfect" context of the previous 63 frames.
            all_z.append(z_window[:, -1, :, :].cpu()) 

        # 3. Save
        full_seq = torch.cat(all_z, dim=0) 
        out_path = cfg.output_dir / "video_full_frames.pt"
        torch.save({"z": full_seq, "length": full_seq.shape[0]}, out_path)
        print(f"✓ Success! Noise-free latents saved to: {out_path}")
        print(f"Final Shape: {full_seq.shape}")

if __name__ == "__main__":
    config = AtariConfig()
    wrapper = TokenizerWrapper(config)
    wrapper.export_all_latents(config)