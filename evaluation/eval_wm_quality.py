"""
Evaluates PSNR + FVD metrics to measure world model performance.
Methodology:
1. Loads target sequence (initial frames)
2. Let model dream for 30 frames using the actual recorded actions from the same video.
3. Compare the "dreamed" frames to the actual frames using PSNR + FVD.
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm

from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier
from world_model.wm.dynamics_model_atari import WorldModel
from training_script.world_model.atari.train_world_model_atari import AtariDataBuilder

# --- Configuration (Aligned with Atari v4 Training) ---
class AtariInferenceConfig:
    tokenizer_ckpt = Path("checkpoints/atari/tokenizer_v3/best_model.pt")
    wm_ckpt_path = Path("checkpoints/world_model/atari_v6/best_wm.pt")
    latent_path = Path("data/atari/latent_sequences/video_full_frames.pt")
    actions_jsonl = Path("data/atari/raw/actions.jsonl")
    output_dir = Path("inference/results/world_model/atari")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    resize = (64, 64)
    patch_size = 8
    n_latents = 64         
    latent_dim = 256
    
    embed_dim = 512        # World Model (v4)
    tokenizer_dim = 256    # Tokenizer (v3)
    num_layers = 12        # World Model (v4)
    
    action_dim = 4
    Sr = 8                 
    Sa = 1                 
    
    num_ode_steps = 50     
    gen_length = 256        
    fps = 10

# --- Helper Functions ---
def load_atari_models(cfg):
    print(f"Loading Atari models on {cfg.device}...")
    
    # 1. Tokenizer (256-dim as per training v3)
    tokenizer = CausalTokenizer(
        input_dim=3 * cfg.patch_size * cfg.patch_size,
        embed_dim=cfg.tokenizer_dim, 
        num_heads=8,
        num_layers=8,
        latent_dim=cfg.latent_dim,
        use_checkpoint=False,
    )
    tok_state = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in tok_state["model_state"].items()}
    tokenizer.load_state_dict(state_dict)
    tokenizer.to(cfg.device).eval()
    
    # 2. World Model & Builder (512-dim, 12 layers as per training v4)
    world_model = WorldModel(
        d_model=cfg.embed_dim,
        d_latent=cfg.latent_dim,
        num_layers=cfg.num_layers,
        num_heads=8,
        n_latents=cfg.n_latents,
        Sr=cfg.Sr,
        use_checkpoint=False 
    )
    builder = AtariDataBuilder(cfg)
    
    wm_state = torch.load(cfg.wm_ckpt_path, map_location="cpu")
    wm_state = {k.replace("module.", ""): v for k, v in wm_state.items()}
    world_model.load_state_dict(wm_state)
    
    world_model.to(cfg.device).eval()
    builder.to(cfg.device).eval()
    
    return tokenizer, world_model, builder

@torch.no_grad()
def get_eval_data(step_start, length, data_dir="data/atari/raw"):
    data_path = Path(data_dir)
    gt_frames = []
    actions = []
    
    # Load actions from JSONL
    with open(data_path / "actions.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i < step_start: continue
            if i >= step_start + length: break
            entry = json.loads(line)
            actions.append(entry["action"])
            
    # Load raw frames as Ground Truth
    for i in range(step_start, step_start + length):
        frame = torch.load(data_path / f"frame_{i:06d}.pt").permute(1, 2, 0).numpy()
        gt_frames.append(frame) # uint8 [0, 255]
        
    return np.array(gt_frames), torch.tensor(actions)

@torch.no_grad()
def decode_latents_full_context(tokenizer, latents, cfg):
    """Refined decoding with latent range protection."""
    T, N, D = latents.shape
    device = latents.device
    patchifier = Patchifier(cfg.patch_size)
    
    # 1. LATENT CLAMPING: Prevents graininess caused by out-of-bounds WM predictions
    # This keeps the WM output in the same 'neighborhood' as the Tokenizer Encoder.
    latents = torch.clamp(latents, -5.0, 5.0) 
    
    z = latents.unsqueeze(0)
    x = tokenizer.from_latent(z) 
    x = x.view(1, T * N, tokenizer.embed_dim)
    
    # Matches WANDB local context fix
    x = x + tokenizer.pos_embed[:, :T * N, :]
    
    x = tokenizer._run_stack(x, tokenizer.decoder, T=T, N=N)
    x = x.view(1, T, N, tokenizer.embed_dim)
    patches = tokenizer.output_proj(x)
    
    frames = patchifier.unpatchify(patches[0], cfg.resize, cfg.patch_size)
    
    all_frames = []
    for t in range(T):
        # Apply slight clipping here to ensure valid RGB range
        img_np = (frames[t].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        all_frames.append(img_np)
            
    return np.array(all_frames)

def calculate_psnr(target_frames, predicted_frames):
    """
    Computes average PSNR between two sequences of images.
    target_frames: (T, H, W, C) numpy array (uint8)
    predicted_frames: (T, H, W, C) numpy array (uint8)
    """
    psnrs = []
    for t in range(len(target_frames)):
        # Convert to float to avoid overflow during subtraction
        target = target_frames[t].astype(np.float32)
        prediction = predicted_frames[t].astype(np.float32)
        
        mse = np.mean((target - prediction) ** 2)
        
        if mse < 1e-10:
            psnrs.append(100.0)  # Perfect match
        else:
            # 255.0 is the max peak value for 8-bit images
            psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
            psnrs.append(psnr)
            
    return np.mean(psnrs), psnrs

# --- Missing Helper: Tokenizer Reconstruction ---
@torch.no_grad()
def decode_gt_sequence(tokenizer, gt_frames, cfg):
    """
    Passes raw frames through the Tokenizer (Encoder -> Decoder).
    Ensures dimensions are correctly handled for the Transformer blocks.
    """
    device = cfg.device
    T = len(gt_frames)
    N = cfg.n_latents
    patchifier = Patchifier(cfg.patch_size)
    
    # 1. Convert numpy (T, H, W, C) -> torch (T, C, H, W)
    frames_torch = torch.from_numpy(gt_frames).permute(0, 3, 1, 2).float() / 255.0
    frames_torch = frames_torch.to(device)
    
    # 2. Patchify: (T, C, H, W) -> (T, N, D_patch)
    patches = patchifier(frames_torch) 
    
    # 3. Project & Add Positional Embeddings
    # x starts as (T, N, D_proj)
    x = tokenizer.input_proj(patches)
    
    # IMPORTANT: The Transformer blocks expect a Batch dimension (B, T*N, D)
    # We unsqueeze to add Batch=1: (1, T, N, D)
    x = x.unsqueeze(0) 
    
    # Flatten T and N for the stack: (1, T*N, D)
    x_flat = x.view(1, T * N, -1)
    
    # Apply positional embeddings across the whole sequence length
    x_flat = x_flat + tokenizer.pos_embed[:, :T * N, :]
    
    # 4. Encoder Stack
    # Now T=64 and N=64 are passed correctly to the factorized layers
    encoded = tokenizer._run_stack(x_flat, tokenizer.encoder, T=T, N=N)
    
    # 5. Bottleneck to Latents
    latents = tokenizer.to_latent(encoded) # (1, T*N, D_latent)
    
    # 6. Prepare for Decoder
    # Reshape back to (T, N, D_latent) to match decode_latents_full_context expectations
    latents_for_decoder = latents.view(T, N, -1)
    
    return decode_latents_full_context(tokenizer, latents_for_decoder, cfg)

# --- Updated evaluate_world_model_psnr ---
@torch.no_grad()
def evaluate_world_model_psnr(cfg, wm, builder, tokenizer, step_start=0, actions=None):
    device = cfg.device
    T_len = cfg.gen_length
    
    # 1. Fetch data
    gt_frames, gt_actions = get_eval_data(step_start, T_len) 
    eval_actions = actions if actions is not None else gt_actions
    actions_tensor = eval_actions.unsqueeze(0).to(device)

    # 2. MATCHING INITIALIZATION: Encode the first frame exactly like decode_gt_sequence does
    # This ensures panel 3 & 4 start EXACTLY where panel 2 starts.
    first_frame = gt_frames[:1] # Take only the first frame
    
    # Use the logic from decode_gt_sequence for just one frame
    frames_torch = torch.from_numpy(first_frame).permute(0, 3, 1, 2).float() / 255.0
    frames_torch = frames_torch.to(device)
    patchifier = Patchifier(cfg.patch_size)
    patches = patchifier(frames_torch) 
    
    x = tokenizer.input_proj(patches)
    x = x.unsqueeze(0) # (1, 1, N, D)
    x_flat = x.view(1, 1 * cfg.n_latents, -1)
    x_flat = x_flat + tokenizer.pos_embed[:, :cfg.n_latents, :]
    
    encoded = tokenizer._run_stack(x_flat, tokenizer.encoder, T=1, N=cfg.n_latents)
    seed_z = tokenizer.to_latent(encoded).view(1, 1, cfg.n_latents, cfg.latent_dim)

    # 3. Initialize ODE Sequence with the Seed
    # Important: Ensure the rest of the sequence is noise so the ODE has something to solve
    z = torch.randn(1, T_len, cfg.n_latents, cfg.latent_dim, device=device)
    z[:, 0] = seed_z

    # 4. ODE Solver
    timesteps = torch.linspace(0, 1, cfg.num_ode_steps + 1, device=device)
    dt = 1.0 / cfg.num_ode_steps
    
    for i in tqdm(range(cfg.num_ode_steps), desc="Solving ODE"):
        t_curr = timesteps[i]
        tau = torch.full((1, T_len), t_curr.item(), device=device)
        d_map = torch.full((1,), 1.0/T_len, device=device) 
        
        tokens = builder(z, actions_tensor, tau, d_map)
        wm_input = {"wm_input_tokens": tokens, "tau": tau, "d": d_map}
        
        # Ensure time_offsets matches the step_start to use correct positional context
        time_offsets = torch.full((1,), step_start, dtype=torch.long, device=device)
        pred_z_clean = wm(wm_input, time_offsets=time_offsets)
        pred_z_clean = torch.clamp(pred_z_clean, -5.0, 5.0)
        
        v_pred = (pred_z_clean - z) / (1.0 - t_curr + 1e-5)
        z[:, 1:] = z[:, 1:] + (v_pred[:, 1:] * dt)

    # Panel 3/4 Final Frames
    predicted_frames = decode_latents_full_context(tokenizer, z.squeeze(0), cfg)
    avg_psnr, psnr_list = calculate_psnr(gt_frames, predicted_frames)

    return {
        "avg_psnr": avg_psnr,
        "psnr_list": psnr_list,
        "gt_frames": gt_frames,
        "predicted_frames": predicted_frames
    }

def main():
    cfg = AtariInferenceConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer, wm, builder = load_atari_models(cfg)

    # 1. Get raw data and actions
    print("Loading Ground Truth and Actions...")
    gt_frames, gt_actions = get_eval_data(step_start=0, length=cfg.gen_length)
    
    # 2. Generate the Tokenizer Reconstruction (The baseline)
    print("Generating Tokenizer Reconstruction...")
    reconstructed_gt = decode_gt_sequence(tokenizer, gt_frames, cfg)

    # 3. Dream A: Real Actions
    print("Running Conditioned Dream (Real Actions)...")
    eval_real = evaluate_world_model_psnr(cfg, wm, builder, tokenizer, step_start=0, actions=gt_actions)
    dream_real = eval_real["predicted_frames"]

    # 4. Dream B: Zero Actions (The Counterfactual)
    print("Running Counterfactual Dream (Zero Actions)...")
    zero_actions = torch.zeros_like(gt_actions)
    eval_zero = evaluate_world_model_psnr(cfg, wm, builder, tokenizer, step_start=0, actions=zero_actions)
    dream_zero = eval_zero["predicted_frames"]

    # 5. Save Quad-Comparison: [RAW | RECON | REAL-DREAM | ZERO-DREAM]
    save_path = cfg.output_dir / "v7_lateset_counterfactual_test.mp4"
    H, W, _ = gt_frames[0].shape
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), cfg.fps, (W * 4, H))
    
    print(f"Saving comparison video to {save_path}...")
    for f_raw, f_rec, f_real, f_zero in zip(gt_frames, reconstructed_gt, dream_real, dream_zero):
        combined = np.hstack([f_raw, f_rec, f_real, f_zero])
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    out.release()
    
    print(f"Test Complete. Compare the 3rd and 4th panels to see action sensitivity.")
   
if __name__ == "__main__":
    main()