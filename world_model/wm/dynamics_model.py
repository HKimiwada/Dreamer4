# Code for Dynamics Model (the backbone of World Model)
# python world_model/wm/dynamics_model.py
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from world_model.wm.transformer_blocks_wm import BlockCausalTransformer, RMSNorm
from world_model.wm_preprocessing.wm_dataset import WorldModelDataset
from world_model.wm_preprocessing.wm_databuilder import DataBuilderWM

class WorldModel(nn.Module):
    """
    Predict the clean latent tokens for each frame, given:
        - Corrupted latents
        - Actions
        - Register tokens
        - Shortcut token (τ, d)
        - Causal history (previous timesteps)

    Outputs of DataBuilderWM
    {
        "wm_input_tokens": (B, T * L_total, D_model),
        "tau":             (B, T),
        "d":               (B,),
        "z_clean":         (B, T, N, D_latent),
        "z_corrupted":     (B, T, N, D_latent),
    }
    """
    def __init__(
        self,
        d_model: int, # Should be same a dim from DataBuilderWM
        d_latent: int,
        num_layers: int,
        num_heads: int,
        clip_length: int = 8, # Clip length of input video
        n_total: int = 461, # Total number of wm_input_tokens (corupted_latent + action_token + register_token + shortcut_token)
        n_latents: int = 448, # Total number of latent tokens (from tokenizer training configs)
    ):
        super().__init__()
        self.n_total = n_total
        self.n_latents = n_latents
        self.d_model = d_model
        self.d_latent = d_latent
        self.time_embed = nn.Embedding(clip_length, d_model)  # Temporal positional embedding
        self.slot_embed = nn.Embedding(n_total, d_model) # Token-type positional embedding
        
        transformer_blocks = []
        for i in range(num_layers):
            causal_time = (i % 4 == 3)
            transformer_blocks.append(BlockCausalTransformer(d_model, num_heads, causal_time))
        self.transformer_blocks = nn.ModuleList(transformer_blocks)
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, d_latent)
        
    def forward(self, data_input_wm): # data_input_wm is the raw output from DataBuilderWM
        wm_input_tokens = data_input_wm["wm_input_tokens"]
        # Handle unbatched vs batched 
        unbatched = (wm_input_tokens.dim() == 2)
        if unbatched:
            wm_input_tokens = wm_input_tokens.unsqueeze(0)

        B, seq_len, dim = wm_input_tokens.shape
        assert dim == self.d_model, f"WorldModel expected d_model={self.d_model}, but got wm_input_tokens dim={dim}"
        N_total = self.n_total
        T = seq_len // N_total
      
        x = wm_input_tokens.view(B, T, N_total, self.d_model)
        
        # Positional Encoding
        # time index: 0..T-1
        t_idx = torch.arange(T, device=x.device)          # (T,)
        time_emb = self.time_embed(t_idx)                 # (T, D) -> Creating embedding for particular timestamp

        # slot index: 0..N_total-1
        s_idx = torch.arange(N_total, device=x.device)    # (N_total,)
        slot_emb = self.slot_embed(s_idx)                 # (N_total, D)

        # broadcast:
        # time_emb → (1, T, 1, D)
        # slot_emb → (1, 1, N_total, D)
        x = x + time_emb[None, :, None, :] + slot_emb[None, None, :, :]

        # flatten back for transformer
        x = x.view(B, T * N_total, self.d_model)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, T, N_total)

        # RMS Normalization
        x = self.final_norm(x)

        # Extracting latents
        x = x.view(B, T, self.n_total, self.d_model)
        latents_only = x[:, :, :self.n_latents, :]

        # Predict uncorrupted latents
        pred_z_clean = self.output_head(latents_only)

        if unbatched:
            pred_z_clean = pred_z_clean.squeeze(0)

        return pred_z_clean

def main():
    dataset = WorldModelDataset(
        latent_dir="data/latent_sequences",
        action_jsonl="data/actions.jsonl",
        clip_length=8,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Test first item
    sample = dataset[0]
    lat = sample["latents"]
    act = sample["actions"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_builder = DataBuilderWM(640).to(device) # Assuming dimension of WM is 640
    input_wm = data_builder(lat, act)
    print("Input to World Model:\n")
    print(input_wm)
    print("Shape of Input to WM:\n")
    print(input_wm["wm_input_tokens"].shape)
    print()
    print("Testing World Model:\n")
    world_model = WorldModel(640, 256, 16, 8).to(device)
    output = world_model(input_wm)
    print(output)
    print("Shape of WM Output:\n")
    print(output.shape)

if __name__ == "__main__":
    # Testing with mock data
    main()