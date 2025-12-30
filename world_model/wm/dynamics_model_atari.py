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
        
    def forward(self, data_input_wm, time_offset=0):
        wm_input_tokens = data_input_wm["wm_input_tokens"] # (B, T*N_total, D)
        
        B, seq_len, dim = wm_input_tokens.shape
        N_total = self.n_total
        T = seq_len // N_total # Calculate number of timesteps in this window
      
        # Reshape to (B, T, N_total, D)
        # If this fails, N_total does not match the builder output!
        x = wm_input_tokens.view(B, T, N_total, self.d_model)
        
        # 1. Temporal Positional Encoding (Absolute time in the environment)
        t_idx = torch.arange(start=time_offset, end=time_offset + T, device=x.device) 
        time_emb = self.time_embed(t_idx) # (T, D)

        # 2. Slot Positional Encoding (Token index within one timestep)
        s_idx = torch.arange(N_total, device=x.device) 
        slot_emb = self.slot_embed(s_idx) # (N_total, D)

        # Apply embeddings
        x = x + time_emb[None, :, None, :] + slot_emb[None, None, :, :]

        # 3. Flatten for Transformer Blocks
        x = x.view(B, T * N_total, self.d_model)

        for block in self.transformer_blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(block, x, T, N_total, use_reentrant=False)
            else:
                x = block(x, T, N_total)

        x = self.final_norm(x)

        # 4. Extract visual latent predictions
        x = x.view(B, T, N_total, self.d_model)
        latents_only = x[:, :, :self.n_latents, :] # Take only the N_latents tokens

        return self.output_head(latents_only) # (B, T, n_latents, d_latent)

# ---------------------------------------------------------------------------
# Updated Test Main for Atari Settings
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Atari Settings: 8x8 patches, 1 action, 8 registers
    n_latents = 64
    Sa = 1
    Sr = 8
    d_model = 512
    d_latent = 256
    T = 8 # Test with 8 frames
    
    # 1. Initialize World Model
    model = WorldModel(
        d_model=d_model,
        d_latent=d_latent,
        num_layers=4,
        num_heads=8,
        n_latents=n_latents,
        Sa=Sa,
        Sr=Sr
    ).to(device)
    
    # 2. Mock Input (Matches Builder output)
    n_total = n_latents + Sa + Sr + 1 # 74
    mock_tokens = torch.randn(2, T * n_total, d_model).to(device)
    input_data = {"wm_input_tokens": mock_tokens}
    
    print(f"Testing Atari World Model:")
    print(f"Tokens per timestep: {n_total}")
    print(f"Input shape: {mock_tokens.shape}")
    
    output = model(input_data, time_offset=100)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, T, n_latents, d_latent)
    print("✓ World Model Reshape and Forward Successful!")

if __name__ == "__main__":
    main()