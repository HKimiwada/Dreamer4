import os
import json
import torch
import torch.nn as nn
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
from world_model.wm.loss import flow_loss_v2

# ---------------------------------------------------------------------------
class AtariWMConfig:
    # Dataset paths
    latent_path = Path("data/atari/latent_sequences/video_full_frames.pt")
    actions_jsonl = Path("data/atari/raw/actions.jsonl")
    tokenizer_ckpt = Path("checkpoints/atari/tokenizer_v2/best_model.pt")
    
    # Model architecture (Must match your latent shape: [100000, 64, 256])
    resize = (64, 64)
    patch_size = 8         # 64 / 8 = 8x8 patches = 64 tokens
    n_latents = 64
    input_dim = 3 * patch_size * patch_size
    latent_dim = 256
    embed_dim = 256
    action_dim = 4         # Breakout: No-op, Fire, Right, Left
    num_layers = 8
    num_heads = 8
    Sa = 1                 # 1 token for the discrete action
    Sr = 8                 # 8 register tokens (standard for this architecture)

    # Training Params
    batch_size = 4         # Per-GPU (Increase if VRAM allows)
    clip_length = 64       # Window size for the transformer
    stride = 32            # Slidind window stride
    lr = 2e-4
    max_steps = 100000
    warmup_steps = 2000
    visualize_interval = 500
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path("checkpoints/world_model/atari_v1")

# ---------------------------------------------------------------------------
class AtariWorldModelDataset(Dataset):
    def __init__(self, cfg):
        print(f"[Dataset] Loading latents from {cfg.latent_path}...")
        data = torch.load(cfg.latent_path, map_location="cpu")
        self.latents = data["z"]  # Expected [100000, 64, 256]
        
        print(f"[Dataset] Loading actions from {cfg.actions_jsonl}...")
        self.actions, self.rewards, self.terminals = [], [], []
        with open(cfg.actions_jsonl, "r") as f:
            for line in f:
                entry = json.loads(line)
                self.actions.append(entry["action"])
                self.rewards.append(entry["reward"])
                self.terminals.append(float(entry["is_terminal"]))
        
        self.actions = torch.tensor(self.actions, dtype=torch.long)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float)
        self.terminals = torch.tensor(self.terminals, dtype=torch.float)
        
        self.clip_length = cfg.clip_length
        self.stride = cfg.stride
        
        # Calculate possible start indices for sliding windows
        self.start_indices = list(range(0, len(self.latents) - self.clip_length, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.clip_length
        return {
            "latents": self.latents[start:end],
            "actions": self.actions[start:end],
            "rewards": self.rewards[start:end],
            "is_terminal": self.terminals[start:end],
            "start_idx": start
        }

# ---------------------------------------------------------------------------
class AtariDataBuilder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.embed_dim)
        self.action_embed = nn.Embedding(cfg.action_dim, cfg.embed_dim)
        
        # Registers: Learnable tokens that give the transformer "thinking space"
        self.register_embed = nn.Embedding(cfg.Sr, cfg.embed_dim)
        
        self.shortcut_slot = nn.Parameter(torch.randn(cfg.embed_dim))
        self.shortcut_mlp = nn.Sequential(
            nn.Linear(2, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim)
        )

    def forward(self, latents, actions, tau, d):
        B, T, N, D = latents.shape
        E = self.cfg.embed_dim
        
        # 1. Visual tokens (B, T, 64, E)
        z_tokens = self.latent_proj(latents) 
        
        # 2. Action tokens (B, T, 1, E)
        a_tokens = self.action_embed(actions).unsqueeze(2) 
        
        # 3. Register tokens (B, T, 8, E)
        reg_ids = torch.arange(self.cfg.Sr, device=latents.device)
        reg_tokens = self.register_embed(reg_ids).view(1, 1, self.cfg.Sr, E).expand(B, T, -1, -1)
        
        # 4. Shortcut tokens (B, T, 1, E)
        feat = torch.stack([tau, d.view(B, 1).expand(B, T)], dim=-1)
        s_tokens = (self.shortcut_mlp(feat) + self.shortcut_slot).unsqueeze(2)
        
        # TOTAL TOKENS: 64 (z) + 1 (a) + 8 (reg) + 1 (s) = 74 tokens per timestep
        tokens = torch.cat([z_tokens, a_tokens, reg_tokens, s_tokens], dim=2) 
        return tokens.view(B, T * (N + self.cfg.Sa + self.cfg.Sr + 1), E)

# ---------------------------------------------------------------------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()

@torch.no_grad()
def visualize_step(wm, builder, tokenizer, batch, cfg, step, device):
    """
    Corrected visualization logic for Atari Breakout.
    Runs the full Decoder pipeline to convert latents to pixels.
    """
    # 1. Support both DDP-wrapped and raw models
    wm_eval = wm.module if hasattr(wm, "module") else wm
    builder_eval = builder.module if hasattr(builder, "module") else builder
    
    wm_eval.eval()
    builder_eval.eval()
    tokenizer.eval()

    # 2. Prepare Data
    latents = batch["latents"].to(device)  # (B, T, N, D_latent)
    actions = batch["actions"].to(device)  # (B, T)
    start_idx = batch["start_idx"][0].item()
    B, T, N, D = latents.shape

    # 3. Get World Model Prediction (Fixed tau=0.5 for diagnostic consistency)
    tau = torch.full((B, T), 0.5, device=device)
    d = torch.full((B,), 0.25, device=device)
    noise = torch.randn_like(latents)
    z_corr = (1.0 - tau.unsqueeze(-1).unsqueeze(-1)) * noise + tau.unsqueeze(-1).unsqueeze(-1) * latents

    tokens = builder_eval(z_corr, actions, tau, d)
    pred_z = wm_eval({"wm_input_tokens": tokens, "tau": tau, "d": d, "z_clean": latents, "z_corrupted": z_corr}, 
                     time_offset=start_idx)

    # 4. Helper: Decodes latents using the Tokenizer's Transformer Decoder
    def decode_latents_to_frames(z_sequence):
        """
        Processes a sequence of latents through the full tokenizer pipeline.
        z_sequence shape: (T_sub, N, D_latent)
        """
        T_sub = z_sequence.shape[0]
        # (T, N, D) -> (1, T, N, D) to satisfy tokenizer batch expectations
        z_in = z_sequence.unsqueeze(0)

        # Step A: Project to embedding dimension (B, T, N, E)
        x = tokenizer.from_latent(z_in) 
        
        # Step B: Flatten for the transformer blocks
        x = x.view(1, T_sub * N, tokenizer.embed_dim)
        
        # Step C: Run through the Decoder Transformer blocks
        # This is essential to recover spatial/temporal structure
        x = tokenizer._run_stack(x, tokenizer.decoder, T=T_sub, N=N)
        
        # Step D: Project back to pixel space (B, T, N, D_pixel)
        x = x.view(1, T_sub, N, tokenizer.embed_dim)
        patches = tokenizer.output_proj(x) # Size 192 for 8x8 patches

        # Step E: Reconstruct pixels using Patchifier
        patchifier = Patchifier(cfg.patch_size)
        frames = []
        for t in range(T_sub):
            # unpatchify expects (1, N, D) -> returns (1, C, H, W)
            f = patchifier.unpatchify(patches[:, t], cfg.resize, cfg.patch_size)
            frames.append(f[0].clamp(0, 1)) # Ensure range is [0, 1]
        return torch.stack(frames)

    # Visualize the first 4 frames of the sequence
    gt_frames = decode_latents_to_frames(latents[0, :4])
    pred_frames = decode_latents_to_frames(pred_z[0, :4])

    # 5. Build Side-by-Side Comparison Grid
    rows = []
    for i in range(gt_frames.shape[0]):
        # Concatenate horizontally: [Ground Truth | Prediction]
        combined = torch.cat([gt_frames[i], pred_frames[i]], dim=-1)
        # Convert [C, H, W] -> [H, W, C] and scale to 255 for WandB
        img_np = (combined.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rows.append(img_np)

    # Stack the 4 frames vertically for a long image
    final_grid = np.concatenate(rows, axis=0)

    wandb.log({
        "world_model_reconstruction": wandb.Image(final_grid, caption=f"Step {step} | Left: Ground Truth, Right: Predicted"),
        "step": step
    })

    wm_eval.train()
    builder_eval.train()
# ---------------------------------------------------------------------------
def main():
    cfg = AtariWMConfig()
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        wandb.init(project="dreamer4-atari-wm", config=vars(cfg))
        cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load Tokenizer for visualization
    tokenizer = CausalTokenizer(input_dim=cfg.input_dim, embed_dim=256, num_heads=8, num_layers=8, latent_dim=256)
    tk_ckpt = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in tk_ckpt["model_state"].items()}
    tokenizer.load_state_dict(state_dict)
    tokenizer.to(device).eval()

    # Dataset
    dataset = AtariWorldModelDataset(cfg)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # Models
    builder = AtariDataBuilder(cfg).to(device)
    wm = WorldModel(d_model=cfg.embed_dim, d_latent=cfg.latent_dim, num_layers=cfg.num_layers, 
                    num_heads=cfg.num_heads, n_latents=cfg.n_latents, Sr=cfg.Sr, use_checkpoint=True).to(device)
    
    wm = DDP(wm, device_ids=[local_rank])
    builder = DDP(builder, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(list(wm.parameters()) + list(builder.parameters()), lr=cfg.lr)
    scaler = GradScaler()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(100):
        sampler.set_epoch(epoch)
        for batch in tqdm(loader, disable=not is_main):
            latents = batch["latents"].to(device)
            actions = batch["actions"].to(device)
            B, T, N, D = latents.shape

            # Sample Noise Levels (Tau and D)
            tau = torch.rand(B, T, device=device)
            d = torch.rand(B, device=device)
            noise = torch.randn_like(latents)
            z_corr = (1.0 - tau.unsqueeze(-1).unsqueeze(-1)) * noise + tau.unsqueeze(-1).unsqueeze(-1) * latents

            with autocast(device_type="cuda", dtype=torch.float16):
                tokens = builder(z_corr, actions, tau, d)
                pred_z = wm({"wm_input_tokens": tokens, "tau": tau, "d": d, "z_clean": latents, "z_corrupted": z_corr}, 
                             time_offset=batch["start_idx"][0].item())
                
                loss = flow_loss_v2(pred_z, latents, tau, ramp_weight=True)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if is_main and global_step % 10 == 0:
                wandb.log({"loss": loss.item(), "step": global_step})
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(wm.module.state_dict(), cfg.ckpt_dir / "best_wm.pt")

            if is_main and global_step % cfg.visualize_interval == 0:
                visualize_step(wm.module, builder.module, tokenizer, batch, cfg, global_step, device)

if __name__ == "__main__":
    main()