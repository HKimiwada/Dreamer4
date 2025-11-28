# Implement Shortcut-Forcing Loss (flow matching + bootstrap).
# python world_model/wm/loss.py
import torch
import torch.nn as nn
from world_model.wm.transformer_blocks_wm import BlockCausalTransformer, RMSNorm
from world_model.wm_preprocessing.wm_dataset import WorldModelDataset
from world_model.wm_preprocessing.wm_databuilder import DataBuilderWM
from world_model.wm.dynamics_model import WorldModel

def flow_loss_v1(pred_z_clean, z_clean, tau):
    """
    Simple flow loss w/no bootstrapping.
    pred_z_clean: (B, T, N_latents, D_latent)
    z_clean:      (B, T, N, D_latent)
    tau:          (B, T)
    """
    if pred_z_clean.dim() == 3:
        print("Unsqueezing pred_z_clean\n")
        pred_z_clean = pred_z_clean.unsqueeze(0)
    if z_clean.dim() == 3:
        print("Unsqueezing z_clean\n")
        z_clean = z_clean.unsqueeze(0)
  
    # 1. Align shapes (N_latents must match)
    B, T, N_latents, D_latent = pred_z_clean.shape
    
    if z_clean.shape[2] != N_latents:
        # slice z_clean to match latent tokens
        z_clean = z_clean[:, :, :N_latents, :]

    # 2. Compute squared error
    diff = pred_z_clean - z_clean         # (B, T, N_latents, D_latent)
    sq_error = diff.pow(2)

    # 3. Ramp weighting based on tau (used for large scale training tasks to stabilize training)
    # tau: (B,T) → expand to (B,T,1,1)
    # w = (0.1 + 0.9 * tau).unsqueeze(-1).unsqueeze(-1)

    # 3. Uniform weighting to force generation learning (for overfitting on small frames)
    print("Using uniform weighting for flow loss.\n")
    w = torch.ones_like(tau).unsqueeze(-1).unsqueeze(-1)

    weighted_error = w * sq_error

    # 4. Mean over all axes
    loss = weighted_error.mean()

    return loss

def test():
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
    pred = world_model(input_wm)
    print(pred)
    print("Shape of WM Output:\n")
    print(pred.shape)
    print("Calculating Loss:\n")
    z_clean = input_wm["z_clean"]
    tau = input_wm["tau"]
    loss = flow_loss_v1(pred, z_clean, tau)
    print("Flow Loss:", loss.item())

if __name__ == "__main__":
    # Testing with mock data
    test()