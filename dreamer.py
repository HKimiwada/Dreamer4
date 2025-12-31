import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from tokenizer.model.encoder_decoder import CausalTokenizer
from world_model.wm.dynamics_model_atari import WorldModel
from tokenizer.patchify_mask import Patchifier
from training_script.world_model.atari.latest_train_world_model_atari import AtariWMConfig, AtariDataBuilder

@torch.no_grad()
def generate_dream_video(num_frames=200):
    cfg = AtariWMConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load weights
    tokenizer = CausalTokenizer(input_dim=cfg.input_dim, embed_dim=256, num_heads=8, num_layers=8, latent_dim=256)
    tk_ckpt = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
    tokenizer.load_state_dict({k.replace("module.", ""): v for k, v in tk_ckpt["model_state"].items()})
    tokenizer.to(device).eval()

    wm = WorldModel(d_model=cfg.embed_dim, d_latent=cfg.latent_dim, num_layers=cfg.num_layers, 
                    num_heads=cfg.num_heads, n_latents=cfg.n_latents, Sr=cfg.Sr, use_checkpoint=False)
    wm_ckpt = torch.load(cfg.ckpt_dir / "best_wm.pt", map_location="cpu")
    wm.load_state_dict({k.replace("module.", ""): v for k, v in wm_ckpt.items()})
    wm.to(device).eval()
    
    builder = AtariDataBuilder(cfg).to(device).eval()

    # 2. Prepare initial state (First 64 frames to prime temporal memory)
    print("Priming model memory with initial sequence...")
    latents_data = torch.load(cfg.latent_path, map_location="cpu")["z"]
    current_latents = latents_data[:64].unsqueeze(0).to(device) # (1, 64, 64, 256)
    
    dream_frames = []
    patchifier = Patchifier(cfg.patch_size)

    # 3. Dreaming Loop
    print(f"Dreaming for {num_frames} frames...")
    for i in tqdm(range(num_frames)):
        # Define actions: Here we just move the paddle randomly (action 0-3)
        # In a real test, you could provide a specific action sequence
        actions = torch.randint(0, cfg.action_dim, (1, 64), device=device)
        
        # In Dreaming, we set tau=1.0 (Pure Prediction / No Ground Truth Signal)
        tau = torch.ones((1, 64), device=device)
        d = torch.zeros((1,), device=device) # No drift during inference
        
        # Build tokens and predict
        tokens = builder(current_latents, actions, tau, d)
        wm_input = {"wm_input_tokens": tokens, "tau": tau, "d": d, "z_clean": current_latents, "z_corrupted": current_latents}
        
        # We assume the dream starts at global index 0 for temporal embeddings
        # The model predicts the NEXT latent state for the entire window
        pred_z = wm(wm_input, time_offsets=torch.tensor([0], device=device)) 
        
        # Extract the last frame of the prediction to add to our dream
        next_latent = pred_z[:, -1:, :, :] 
        
        # Decode the frame for the video
        # We use local context 0-63 as established in our diagnostic
        x = tokenizer.from_latent(next_latent)
        x = x.view(1, 1 * 64, tokenizer.embed_dim)
        x = x + tokenizer.pos_embed[:, 63*64 : 64*64, :] # Use the last position's embedding
        x = tokenizer._run_stack(x, tokenizer.decoder, T=1, N=64)
        x = x.view(1, 1, 64, tokenizer.embed_dim)
        patches = tokenizer.output_proj(x)
        frame = patchifier.unpatchify(patches.squeeze(0), cfg.resize, cfg.patch_size)[0]
        
        img = (frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        dream_frames.append(img)
        
        # Update current_latents: Slide the window (Drop oldest, add newest prediction)
        current_latents = torch.cat([current_latents[:, 1:, :, :], next_latent], dim=1)

    # 4. Save to Video
    out_path = "latest_atari_dream.mp4"
    height, width, _ = dream_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, 20, (width, height))

    for f in dream_frames:
        video.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    video.release()
    print(f"✓ Dream video saved to {out_path}")

if __name__ == "__main__":
    generate_dream_video()