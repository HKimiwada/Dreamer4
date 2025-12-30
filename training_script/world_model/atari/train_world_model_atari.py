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
    
    # Model architecture
    resize = (64, 64)
    patch_size = 8         
    n_latents = 64
    input_dim = 3 * patch_size * patch_size
    latent_dim = 256
    embed_dim = 512        
    action_dim = 4         
    num_layers = 12        
    num_heads = 8
    Sa = 1                 
    Sr = 8                 

    # Training Params
    batch_size = 16        
    clip_length = 64       
    stride = 32            
    lr = 2e-4
    max_steps = 100000     
    warmup_steps = 5000    
    visualize_interval = 500
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path("checkpoints/world_model/atari_v2")

# ---------------------------------------------------------------------------
class AtariWorldModelDataset(Dataset):
    def __init__(self, cfg):
        print(f"[Dataset] Loading latents from {cfg.latent_path}...")
        data = torch.load(cfg.latent_path, map_location="cpu")
        self.latents = data["z"]  
        
        print(f"[Dataset] Loading actions from {cfg.actions_jsonl}...")
        self.actions = []
        with open(cfg.actions_jsonl, "r") as f:
            for line in f:
                entry = json.loads(line)
                self.actions.append(entry["action"])
        
        self.actions = torch.tensor(self.actions, dtype=torch.long)
        self.clip_length = cfg.clip_length
        self.stride = cfg.stride
        self.start_indices = list(range(0, len(self.latents) - self.clip_length, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.clip_length
        return {
            "latents": self.latents[start:end],
            "actions": self.actions[start:end],
            "start_idx": start
        }

# ---------------------------------------------------------------------------
class AtariDataBuilder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.embed_dim)
        self.action_embed = nn.Embedding(cfg.action_dim, cfg.embed_dim)
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
        z_tokens = self.latent_proj(latents) 
        a_tokens = self.action_embed(actions).unsqueeze(2) 
        reg_ids = torch.arange(self.cfg.Sr, device=latents.device)
        reg_tokens = self.register_embed(reg_ids).view(1, 1, self.cfg.Sr, E).expand(B, T, -1, -1)
        feat = torch.stack([tau, d.view(B, 1).expand(B, T)], dim=-1)
        s_tokens = (self.shortcut_mlp(feat) + self.shortcut_slot).unsqueeze(2)
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
    wm_eval = wm.module if hasattr(wm, "module") else wm
    builder_eval = builder.module if hasattr(builder, "module") else builder
    wm_eval.eval(); builder_eval.eval(); tokenizer.eval()

    latents = batch["latents"].to(device)
    actions = batch["actions"].to(device)
    B, T, N, D = latents.shape
    tau = torch.full((B, T), 0.5, device=device)
    d = torch.full((B,), 0.25, device=device)
    noise = torch.randn_like(latents)
    z_corr = (1.0 - tau.unsqueeze(-1).unsqueeze(-1)) * noise + tau.unsqueeze(-1).unsqueeze(-1) * latents

    # Corrected: Build input dictionary and pass start_idx as a Tensor
    tokens = builder_eval(z_corr, actions, tau, d)
    wm_input = {"wm_input_tokens": tokens, "tau": tau, "d": d, "z_clean": latents, "z_corrupted": z_corr}
    # Use only the first item in batch for viz, but maintain tensor shape (1,)
    pred_z = wm_eval(wm_input, time_offsets=batch["start_idx"][:1].to(device))

    def decode(z_seq):
        # IMPORTANT: Tokenizer decoder must match trained clip_length
        T_sub = z_seq.shape[0]; z_in = z_seq.unsqueeze(0)
        x = tokenizer.from_latent(z_in) 
        x = x.view(1, T_sub * N, tokenizer.embed_dim)
        x = tokenizer._run_stack(x, tokenizer.decoder, T=T_sub, N=N)
        x = x.view(1, T_sub, N, tokenizer.embed_dim)
        patches = tokenizer.output_proj(x)
        return Patchifier(cfg.patch_size).unpatchify(patches.squeeze(0), cfg.resize, cfg.patch_size)

    # Visualize first 4 frames of the sequence
    gt_f = decode(latents[0, :4]); pr_f = decode(pred_z[0, :4])
    rows = [np.concatenate([(torch.cat([gt_f[i], pr_f[i]], dim=-1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)], axis=1) for i in range(4)]
    final_grid = np.concatenate(rows, axis=0)
    wandb.log({"reconstruction": wandb.Image(final_grid, caption=f"Step {step} (L:GT, R:Pred)"), "step": step})
    wm_eval.train(); builder_eval.train()

# ---------------------------------------------------------------------------
def main():
    cfg = AtariWMConfig()
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        wandb.init(project="dreamer4-atari-wm", config=vars(cfg))
        cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CausalTokenizer(input_dim=cfg.input_dim, embed_dim=256, num_heads=8, num_layers=8, latent_dim=256)
    tk_ckpt = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
    tokenizer.load_state_dict({k.replace("module.", ""): v for k, v in tk_ckpt["model_state"].items()})
    tokenizer.to(device).eval()

    dataset = AtariWorldModelDataset(cfg)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    builder = AtariDataBuilder(cfg).to(device)
    wm = WorldModel(d_model=cfg.embed_dim, d_latent=cfg.latent_dim, num_layers=cfg.num_layers, 
                    num_heads=cfg.num_heads, n_latents=cfg.n_latents, Sr=cfg.Sr, use_checkpoint=True).to(device)
    
    wm = DDP(wm, device_ids=[local_rank]); builder = DDP(builder, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(list(wm.parameters()) + list(builder.parameters()), lr=cfg.lr)
    scaler = GradScaler(); global_step = 0; best_loss = float('inf'); epoch = 0

    while global_step < cfg.max_steps:
        sampler.set_epoch(epoch)
        for batch in tqdm(loader, disable=not is_main, desc=f"Epoch {epoch}"):
            if global_step >= cfg.max_steps: break
            
            latents = batch["latents"].to(device)
            actions = batch["actions"].to(device)
            tau = torch.rand(latents.shape[0], latents.shape[1], device=device)
            d = torch.rand(latents.shape[0], device=device)
            noise = torch.randn_like(latents)
            z_corr = (1.0 - tau.unsqueeze(-1).unsqueeze(-1)) * noise + tau.unsqueeze(-1).unsqueeze(-1) * latents

            with autocast(device_type="cuda", dtype=torch.float16):
                tokens = builder(z_corr, actions, tau, d)
                # Corrected: Define wm_input dictionary
                wm_input = {"wm_input_tokens": tokens, "tau": tau, "d": d, "z_clean": latents, "z_corrupted": z_corr}
                # Corrected: Pass the entire batch of start_indices
                pred_z = wm(wm_input, time_offsets=batch["start_idx"].to(device))
                loss = flow_loss_v2(pred_z, latents, tau, ramp_weight=True)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

            global_step += 1
            if is_main and global_step % 10 == 0:
                wandb.log({"loss": loss.item(), "step": global_step, "epoch": epoch})
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(wm.module.state_dict(), cfg.ckpt_dir / "best_wm.pt")

            if is_main and global_step % cfg.visualize_interval == 0:
                visualize_step(wm, builder, tokenizer, batch, cfg, global_step, device)
        epoch += 1

    if is_main: wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()