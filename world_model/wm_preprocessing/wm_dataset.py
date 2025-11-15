# Combining latent token + action description to create input to WM.
# python world_model/data/wm_dataset.py
import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np

# ---------------------------------------------------------------------
# Keyboard mapping for 23 binary keys (consistent with VPT + Dreamer4)
# ---------------------------------------------------------------------
KEY_LIST = [
    "key.keyboard.w", "key.keyboard.a", "key.keyboard.s", "key.keyboard.d",
    "key.keyboard.space", "key.keyboard.left.shift", "key.keyboard.left.control",
    "key.keyboard.e", "key.keyboard.q",
    "key.keyboard.1", "key.keyboard.2", "key.keyboard.3",
    "key.keyboard.4", "key.keyboard.5", "key.keyboard.6",
    "key.keyboard.7", "key.keyboard.8", "key.keyboard.9",
    "key.keyboard.f",
    "key.keyboard.r",
    "key.keyboard.tab",
    "key.keyboard.left.alt",
    "key.keyboard.right.alt"
]

# ---------------------------------------------------------------------
# μ-law discretization for mouse dx/dy → integer 0..10
# ---------------------------------------------------------------------
def mulaw_encode(x, mu=10):
    """ Convert continuous mouse delta into discretized 0..mu """
    x = np.sign(x) * np.log1p(mu * abs(x)) / np.log1p(mu)
    x = int(np.clip((x + 1) / 2 * mu, 0, mu))
    return x

# ---------------------------------------------------------------------
# Single-frame action encoder returning SPLIT COMPONENTS
# ---------------------------------------------------------------------
def encode_action_components(a):
    """ Encode a raw JSON action into structured Dreamer4-style components. """

    mouse = a["mouse"]
    keys  = a["keyboard"]["keys"]

    # --------------------
    # Mouse dx/dy → 121-way categorical via μ-law
    # --------------------
    dx_bin = mulaw_encode(mouse["dx"])
    dy_bin = mulaw_encode(mouse["dy"])
    mouse_cat = dx_bin * 11 + dy_bin   # int ∈ [0..120]

    # --------------------
    # Scroll wheel (map -∞..0..∞ to 0,1,2)
    # --------------------
    dwheel = mouse["dwheel"]
    if   dwheel > 0: scroll = 2
    elif dwheel < 0: scroll = 0
    else:            scroll = 1

    # --------------------
    # Mouse buttons (3 binary flags)
    # JSON has "buttons": [0] or [] etc.
    # --------------------
    buttons = [
        1.0 if 0 in mouse.get("buttons", []) else 0.0,   # left
        1.0 if 1 in mouse.get("buttons", []) else 0.0,   # right
        1.0 if 2 in mouse.get("buttons", []) else 0.0,   # middle
    ]

    # --------------------
    # Keyboard keys → 23-dim binary vector
    # --------------------
    key_vec = torch.zeros(len(KEY_LIST), dtype=torch.float32)
    keyset = set(keys)
    for i, k in enumerate(KEY_LIST):
        if k in keyset:
            key_vec[i] = 1.0

    # --------------------
    # Camera angles
    # --------------------
    yaw   = float(a["yaw"])
    pitch = float(a["pitch"])
    yaw_pitch = torch.tensor([yaw, pitch], dtype=torch.float32)

    # --------------------
    # Hotbar selection (0..8)
    # --------------------
    hotbar = float(a.get("hotbar", 0))

    # --------------------
    # GUI states
    # --------------------
    gui = torch.tensor([
        1.0 if a["isGuiOpen"] else 0.0,
        1.0 if a["isGuiInventory"] else 0.0,
    ], dtype=torch.float32)

    # --------------------
    # Return split format
    # --------------------
    return {
        "mouse_cat": torch.tensor(mouse_cat, dtype=torch.long),   # (,) categorical
        "scroll":    torch.tensor(scroll, dtype=torch.long),      # (,) categorical
        "yaw_pitch": yaw_pitch,                                   # (2,)
        "hotbar":    torch.tensor(hotbar, dtype=torch.long),      # (,) categorical
        "gui":       gui,                                         # (2,)
        "buttons":   torch.tensor(buttons, dtype=torch.float32),  # (3,)
        "keys":      key_vec,                                     # (23,)
    }

# ---------------------------------------------------------------------
# The Dataset for Dreamer4-style Dynamics Training
# ---------------------------------------------------------------------
class WorldModelDataset(Dataset):
    """
    Returns:
        {
            "latents":      (T, N, D)
            "actions": {
                "mouse_cat": (T,)      long
                "scroll":    (T,)      long
                "yaw_pitch": (T,2)     float
                "hotbar":    (T,)      long
                "gui":       (T,2)     float
                "buttons":   (T,3)     float
                "keys":      (T,23)    float
            }
        }
    """

    def __init__(self, latent_dir, action_jsonl, clip_length=8, device="cuda"):
        self.device = device
        self.clip_length = clip_length

        # Load latent clips
        self.latent_files = sorted(Path(latent_dir).glob("*.pt"))
        print(f"[WM Dataset] Found {len(self.latent_files)} latent clips.")

        # Load actions (one per raw frame)
        self.actions = []
        with open(action_jsonl, "r") as f:
            for line in f:
                self.actions.append(json.loads(line))
        print(f"[WM Dataset] Loaded {len(self.actions)} frame actions.")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        # ------------------------
        # Load latent tokens
        # ------------------------
        entry = torch.load(self.latent_files[idx], map_location="cpu")
        latents = entry["z"]                     # shape (T, N, D)
        T = latents.shape[0]

        # ------------------------
        # Fetch matching raw actions
        # ------------------------
        start = idx * self.clip_length
        end   = start + self.clip_length

        clip_actions = self.actions[start:end]
        if len(clip_actions) != T:
            raise RuntimeError(
                f"Action length mismatch for clip {idx}: expected {T}, got {len(clip_actions)}"
            )

        # ------------------------
        # Encode actions into split components
        # ------------------------
        mouse_cat_list = []
        scroll_list    = []
        yaw_pitch_list = []
        hotbar_list    = []
        gui_list       = []
        buttons_list   = []
        keys_list      = []

        for a in clip_actions:
            enc = encode_action_components(a)

            mouse_cat_list.append(enc["mouse_cat"])
            scroll_list.append(enc["scroll"])
            yaw_pitch_list.append(enc["yaw_pitch"])
            hotbar_list.append(enc["hotbar"])
            gui_list.append(enc["gui"])
            buttons_list.append(enc["buttons"])
            keys_list.append(enc["keys"])

        # Stack into tensors
        actions = {
            "mouse_cat": torch.stack(mouse_cat_list, dim=0).to(self.device),   # (T,)
            "scroll":    torch.stack(scroll_list, dim=0).to(self.device),      # (T,)
            "yaw_pitch": torch.stack(yaw_pitch_list, dim=0).to(self.device),   # (T,2)
            "hotbar":    torch.stack(hotbar_list, dim=0).to(self.device),      # (T,)
            "gui":       torch.stack(gui_list, dim=0).to(self.device),         # (T,2)
            "buttons":   torch.stack(buttons_list, dim=0).to(self.device),     # (T,3)
            "keys":      torch.stack(keys_list, dim=0).to(self.device),        # (T,23)
        }

        # Move latents to target device
        latents = latents.to(self.device)

        return {
            "latents": latents,   # (T, N, D)
            "actions": actions,
        }

def main():
    dataset = WorldModelDataset(
        latent_dir="data/latent_sequences",
        action_jsonl="data/actions.jsonl",
        clip_length=8,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nTotal clips:", len(dataset))

    # Test first item
    sample = dataset[0]
    lat = sample["latents"]
    act = sample["actions"]

    print("\n--- FIRST SAMPLE ---")
    print("latents:", lat.shape)      # expect (T, N, D)
    print("mouse_cat:", act["mouse_cat"].shape)
    print("scroll:", act["scroll"].shape)
    print("yaw_pitch:", act["yaw_pitch"].shape)
    print("hotbar:", act["hotbar"].shape)
    print("gui:", act["gui"].shape)
    print("buttons:", act["buttons"].shape)
    print("keys:", act["keys"].shape)

    print("\nLatents device:", lat.device)
    print("Buttons example:", act["buttons"][0])

if __name__ == "__main__":
    main()
