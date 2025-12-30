# world_model/wm/dynamics_model.py
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from world_model.wm.transformer_blocks_wm import BlockCausalTransformer, RMSNorm

class WorldModel(nn.Module):
    def __init__(
        self,
        d_model: int,     # embed_dim (e.g., 512)
        d_latent: int,    # latent_dim (e.g., 256)
        num_layers: int,
        num_heads: int,
        clip_length: int = 64, # Maximum temporal window (e.g., 64 frames)
        n_latents: int = 64,   # 8x8 patches = 64 tokens
        Sa: int = 1,           # Action tokens (Atari = 1)
        Sr: int = 8,           # Register tokens
        use_checkpoint: bool = True 
    ):
        super().__init__()
        # NEW: Calculate n_total dynamically: latents + actions + registers + 1 shortcut
        self.n_latents = n_latents
        self.n_total = n_latents + Sa + Sr + 1 
        
        self.d_model = d_model
        self.d_latent = d_latent
        
        # Positional Embeddings
        self.time_embed = nn.Embedding(200000, d_model) # Large enough for 100k+ steps
        self.slot_embed = nn.Embedding(self.n_total, d_model)
        
        self.use_checkpoint = use_checkpoint
        
        # Transformer Backbone
        transformer_blocks = []
        for i in range(num_layers):
            # Every 4th block is a 'Temporal' block (standard Dreamer4 architecture)
            causal_time = (i % 4 == 3)
            transformer_blocks.append(BlockCausalTransformer(d_model, num_heads, causal_time))
        self.transformer_blocks = nn.ModuleList(transformer_blocks)
        
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, d_latent)
        
    def forward(self, data_input_wm, time_offsets):
        # time_offsets: tensor of shape (B,) 
        wm_input_tokens = data_input_wm["wm_input_tokens"] 
        B, seq_len, dim = wm_input_tokens.shape
        N_total = self.n_total
        T = seq_len // N_total
      
        x = wm_input_tokens.view(B, T, N_total, self.d_model)
        
        # --- FIX: Per-sample temporal indices ---
        # (B, 1) + (1, T) -> (B, T)
        t_indices = time_offsets.unsqueeze(1) + torch.arange(T, device=x.device).unsqueeze(0)
        time_emb = self.time_embed(t_indices) # (B, T, D)

        s_idx = torch.arange(N_total, device=x.device) 
        slot_emb = self.slot_embed(s_idx) # (N_total, D)

        # Apply embeddings (B, T, D) broadcasts to (B, T, N_total, D)
        x = x + time_emb.unsqueeze(2) + slot_emb[None, None, :, :]
        # ----------------------------------------

        x = x.view(B, T * N_total, self.d_model)
        for block in self.transformer_blocks:
            x = checkpoint.checkpoint(block, x, T, N_total, use_reentrant=False) if self.training else block(x, T, N_total)

        x = self.final_norm(x).view(B, T, N_total, self.d_model)
        return self.output_head(x[:, :, :self.n_latents, :])

# ---------------------------------------------------------------------------
# Corrected Test Main for Atari Settings
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    T = 8 
    
    # 1. Initialize World Model
    model = WorldModel(
        d_model=512,
        d_latent=256,
        num_layers=4,
        num_heads=8,
        n_latents=64,
        Sa=1,
        Sr=8
    ).to(device)
    
    # 2. Mock Input (Matches Builder output)
    n_total = 64 + 1 + 8 + 1 # 74
    mock_tokens = torch.randn(batch_size, T * n_total, 512).to(device)
    input_data = {"wm_input_tokens": mock_tokens}
    
    # --- FIX: Create a tensor for time_offsets matching the batch size ---
    # In training, this would be batch["start_idx"]
    time_offsets = torch.tensor([100, 5000], device=device) 
    
    print(f"Testing Atari World Model:")
    print(f"Tokens per timestep: {n_total}")
    print(f"Batch Size: {batch_size} | Time Steps: {T}")
    
    # Pass as the correct keyword 'time_offsets'
    output = model(input_data, time_offsets=time_offsets)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, T, 64, 256)
    print("✓ World Model Reshape and Forward Successful!")

if __name__ == "__main__":
    main()