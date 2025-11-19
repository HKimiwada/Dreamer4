# Data Builder to prepare data for input into World Model (WM).
# PYTHONPATH=. python world_model/wm/wm_databuilder.py
# Input: latent tokens, numerically-converted actions (from wm_dataset)
# Output: (latent tokens, tokenized actions, register tokens, short cut token) for each timestep
import math
import torch
import torch.nn as nn
from world_model.wm_preprocessing.action_tokenizer import ActionTokenizer
from world_model.wm_preprocessing.wm_dataset import WorldModelDataset

class DataBuilderWM(nn.Module):
    """
    Builds Dreamer4-style input sequences for the world model transformer.

    Inputs (forward method):
        latents: (T, N, D_latent) or (B, T, N, D_latent)
        actions: dict with shapes (T, ...) or (B, T, ...)

    Outputs:
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
        d_model: int,
        d_latent: int = 256,
        register_tokens: int = 8,
        k_max: int = 64,  # finest number of diffusion steps
    ):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.register_tokens = register_tokens
        self.k_max = k_max

        # project tokenizer latents → world model dim
        self.latent_project = nn.Linear(d_latent, d_model)

        # action tokenizer (handles batched + unbatched)
        self.action_tokenizer = ActionTokenizer(d_model)

        # register tokens: Sr learnable vectors (acts like scratch pad)
        self.register_embed = nn.Embedding(register_tokens, d_model)

        # shortcut token: encode (tau, d) → D_model
        self.shortcut_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Shortcut slot embedding to mark token type
        self.shortcut_slot = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.shortcut_slot, std=0.02)

        # precompute allowed step sizes d ∈ {1/k_max, 2/k_max, 4/k_max, ..., 1}
        max_pow = int(math.log2(k_max))
        d_values = [ (2**i) / k_max for i in range(max_pow + 1) ]  # powers of 2 / k_max
        self.register_buffer("d_values", torch.tensor(d_values, dtype=torch.float32))

    def forward(self, latents, actions):
        """
        latents: (T, N, D_latent) or (B, T, N, D_latent)
        actions: dict with (T, ...) or (B, T, ...)

        Returns dict with wm_input_tokens, tau, d, z_clean, z_corrupted.
        """
        device = latents.device

        # Handle unbatched vs batched 
        unbatched = (latents.dim() == 3)  # (T,N,D)
        if unbatched:
            latents = latents.unsqueeze(0)              # (1,T,N,D)
            actions = {k: v.unsqueeze(0) for k, v in actions.items()}

        B, T, N, D_lat = latents.shape
        assert D_lat == self.d_latent, f"Expected latent dim {self.d_latent}, got {D_lat}"

        # Clean latents in original latent space 
        z_clean = latents  # (B,T,N,D_latent)

        # Project latents into model space 
        z_proj = self.latent_project(z_clean)  # (B,T,N,D_model)

        # Tokenize actions → (B,T,Sa,D_model) 
        action_tokens = self.action_tokenizer(actions)  # (B,T,Sa,D_model)
        Sa = action_tokens.shape[2]

        # Register tokens: (B,T,Sr,D_model) 
        Sr = self.register_tokens
        register_ids = torch.arange(Sr, device=device)              # (Sr,)
        reg_base = self.register_embed(register_ids)                # (Sr,D)
        reg_base = reg_base.view(1, 1, Sr, self.d_model)           # (1,1,Sr,D)
        register_tokens = reg_base.expand(B, T, Sr, self.d_model)  # (B,T,Sr,D)

        # Sample step sizes d for each sequence in batch 
        num_d = self.d_values.shape[0]
        d_idx = torch.randint(0, num_d, size=(B,), device=device)   # (B,)
        d = self.d_values[d_idx]                                   # (B,)

        # Sample tau_t per (B,T) on grid {0, d, 2d, ..., 1} 
        # For simplicity: sample n in [0, 1/d] per (B,T), tau = n * d[b]
        tau = torch.zeros(B, T, device=device)
        for b in range(B):
            max_n = int(1.0 / float(d[b]))  # number of steps
            n_t = torch.randint(0, max_n + 1, size=(T,), device=device)
            tau[b] = n_t * d[b]

        # Corrupt latents: z_tau = (1 - tau)*noise + tau*z_clean 
        noise = torch.randn_like(z_clean)
        tau_bt = tau.view(B, T, 1, 1)                          # (B,T,1,1)
        z_corrupted = (1.0 - tau_bt) * noise + tau_bt * z_clean  # (B,T,N,D_latent)
        z_corrupted_proj = self.latent_project(z_corrupted)     # (B,T,N,D_model)

        # Build shortcut tokens from (tau,d) 
        # Features: [tau_t, d_b] for each (b,t)
        d_bt = d.view(B, 1).expand(B, T)  # (B,T)
        feat = torch.stack([tau, d_bt], dim=-1)   # (B,T,2)
        shortcut_vec = self.shortcut_mlp(feat)    # (B,T,D_model)
        shortcut_vec = shortcut_vec + self.shortcut_slot.view(1, 1, -1)  # add type embedding
        shortcut_tokens = shortcut_vec.unsqueeze(2)   # (B,T,1,D_model)

        # ---- Concatenate all tokens per timestep ----
        wm_tokens = torch.cat(
            [z_corrupted_proj, action_tokens, register_tokens, shortcut_tokens],
            dim=2
        )  # (B,T, N+Sa+Sr+1, D_model)

        B, T, L_total, Dm = wm_tokens.shape

        # Flatten time + token dims for transformer 
        wm_input_tokens = wm_tokens.view(B, T * L_total, Dm)  # (B, T*L_total, D_model)

        out = {
            "wm_input_tokens": wm_input_tokens,
            "tau":             tau,            # (B,T)
            "d":               d,              # (B,)
            "z_clean":         z_clean,        # (B,T,N,D_latent)
            "z_corrupted":     z_corrupted,    # (B,T,N,D_latent)
        }

        if unbatched:
            # squeeze batch dim back out for convenience
            out["wm_input_tokens"] = out["wm_input_tokens"].squeeze(0)
            out["tau"]             = out["tau"].squeeze(0)
            out["d"]               = out["d"].squeeze(0)
            out["z_clean"]         = out["z_clean"].squeeze(0)
            out["z_corrupted"]     = out["z_corrupted"].squeeze(0)

        return out


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

if __name__ == "__main__":
    main()

