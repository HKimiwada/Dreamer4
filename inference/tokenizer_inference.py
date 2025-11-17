# CUDA_VISIBLE_DEVICES=0 python inference/tokenizer_inference.py
import torch
import imageio
import numpy as np
from pathlib import Path

from tokenizer.model.encoder_decoder import CausalTokenizer2
from tokenizer.patchify_mask import Patchifier
from tokenizer.tokenizer_dataset import TokenizerDatasetWM

# ----------------------------------------------------------------------------
class InferenceConfig:
    video_file = "cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4"

    data_dir = Path("data")
    ckpt_path = Path("checkpoints/overfit/latest_complete_overfit_mse/best_model.pt")

    # Model / dataset params (must match training)
    resize = (256, 448)
    patch_size = 16
    clip_length = 8
    input_dim = 3 * patch_size * patch_size
    embed_dim = 512
    latent_dim = 256
    num_heads = 16
    num_layers = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    out_video = "inference/results/v2_reconstructed_output.mp4"
    fps = 30

# ----------------------------------------------------------------------------
def load_model(cfg):
    print(f"[Load] Loading checkpoint: {cfg.ckpt_path}")
    model = CausalTokenizer2(
        input_dim=cfg.input_dim,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        latent_dim=cfg.latent_dim,
        use_checkpoint=False,
    )

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    state = ckpt["model_state"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    model.to(cfg.device)
    model.eval()
    print("✓ Model loaded")
    return model

# ----------------------------------------------------------------------------
def load_video_dataset(cfg):
    print(f"[Dataset] Loading: {cfg.video_file}")

    dataset = TokenizerDatasetWM(
        video_file=cfg.data_dir / cfg.video_file,
        target_fps=20.0,
        resize=cfg.resize,
        clip_length=cfg.clip_length,
        stride=cfg.clip_length,      # non-overlapping sequential clips
        mode="sequential",
        drop_last=False,
        patch_size=cfg.patch_size,
        mask_prob_range=(0.0, 0.0),  # no masking at inference
        per_frame_mask_sampling=False,
        device=torch.device("cpu"),
    )

    print("✓ Dataset ready (iterator mode, sequential clips)")
    return dataset

# ----------------------------------------------------------------------------
@torch.no_grad()
def reconstruct_video(model, dataset, cfg):
    patchifier = Patchifier(cfg.patch_size)
    reconstructed_frames = []

    print(f"[Reconstruct] Starting inference...")

    # Iterate over ALL sequential clips
    for sample in dataset:
        patches = sample["patch_tokens"].unsqueeze(0).to(cfg.device)
        mask = sample["mask"].unsqueeze(0).to(cfg.device)

        recon = model(patches, mask).clamp(0, 1)

        T = recon.shape[1]
        H, W = cfg.resize

        # Unpatchify per frame
        for t in range(T):
            frame = patchifier.unpatchify(
                recon[0, t:t+1], (H, W), cfg.patch_size
            )[0]
            frame = frame.clamp(0, 1)
            frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            reconstructed_frames.append(frame_np)

    print(f"✓ Reconstructed {len(reconstructed_frames)} frames")
    return reconstructed_frames

# ----------------------------------------------------------------------------
def save_video(frames, cfg):
    Path(cfg.out_video).parent.mkdir(parents=True, exist_ok=True)

    print(f"[Save] Writing {len(frames)} frames → {cfg.out_video}")
    imageio.mimsave(cfg.out_video, frames, fps=cfg.fps)
    print("✓ Done")

# ----------------------------------------------------------------------------
def main():
    cfg = InferenceConfig()

    model = load_model(cfg)
    dataset = load_video_dataset(cfg)
    frames = reconstruct_video(model, dataset, cfg)
    save_video(frames, cfg)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()