# ActionTokenizer: converts raw action components (mouse, keyboard, scroll, hotbar) into transformer tokens.
# Tokenizer trained along-side world model.
import torch
import torch.nn as nn

class ActionTokenizer(nn.module):
    """
    Dreamer4-style simplified multi-token action encoder.

    INPUT: actions dict of batched tensors:
        mouse_cat: (B, T) long
        scroll:    (B, T) long
        buttons:   (B, T, 3) float
        keys:      (B, T, 23) float
        yaw_pitch: (B, T, 2) float
        gui:       (B, T, 2) float
        hotbar:    (B, T) long

    OUTPUT:
        action_tokens: (B, T, Sa=4, D_model)
    """
    def __init__(
        self, d_model: int, 
        mouse_cat_vocab: int = 121, 
        scroll_vocab: int = 3, 
        hotbar_vocab: int = 9,
        num_keys: int = 23,
        num_buttons: int =3,
    )
        super().__init__()
        self.Sa = 4
        self.d_model = d_model

        # ----- Token 0: mouse movement (categorical) -----
        self.mouse_embed = nn.Embedding(mouse_cat_vocab, d_model)

        # ----- Token 1: scroll (categorical) + buttons (3 floats) -----
        self.scroll_embed = nn.Embedding(scroll_vocab, d_model)
        self.buttons_proj = nn.Linear(num_buttons, d_model)

        # ----- Token 2: keyboard multi-hot (23 dims) -----
        self.keys_proj = nn.Linear(num_keys, d_model)

        # ----- Token 3: yaw_pitch(2) + gui(2) + hotbar embed -----
        self.yaw_gui_proj = nn.Linear(4, d_model)  # (yaw,pitch,gui[2])
        self.hotbar_embed = nn.Embedding(hotbar_vocab, d_model)

        # ----- Slot embeddings to differentiate the 4 tokens -----
        self.slot_embed = nn.Embedding(self.Sa, d_model)

    def forward(self, actions):
        """
        Returns: (B, T, Sa, D_model): Batch, Token clip length, Sa (action token from tokenizer), model dim.
        """
        mouse_cat = actions["mouse_cat"]   # (B,T)
        scroll    = actions["scroll"]      # (B,T)
        buttons   = actions["buttons"]     # (B,T,3)
        keys      = actions["keys"]        # (B,T,23)
        yaw_pitch = actions["yaw_pitch"]   # (B,T,2)
        gui       = actions["gui"]         # (B,T,2)
        hotbar    = actions["hotbar"]      # (B,T)

        B, T = mouse_cat.shape

        # ---------------- Token 0 ----------------
        tok0 = self.mouse_embed(mouse_cat)     # (B,T,D)

        # ---------------- Token 1 ----------------
        scroll_emb = self.scroll_embed(scroll)              # (B,T,D)
        buttons_emb = self.buttons_proj(buttons)            # (B,T,D)
        tok1 = scroll_emb + buttons_emb                     # (B,T,D)

        # ---------------- Token 2 ----------------
        tok2 = self.keys_proj(keys)                         # (B,T,D)

        # ---------------- Token 3 ----------------
        yaw_gui = torch.cat([yaw_pitch, gui], dim=-1)       # (B,T,4)
        yaw_gui_emb = self.yaw_gui_proj(yaw_gui)            # (B,T,D)
        hotbar_emb = self.hotbar_embed(hotbar)              # (B,T,D)
        tok3 = yaw_gui_emb + hotbar_emb                     # (B,T,D)

        # Stack into Sa tokens
        tokens = torch.stack([tok0, tok1, tok2, tok3], dim=2)  # (B,T,Sa,D)

        # Add slot embeddings (so token 0 ≠ token 1 ≠ token 2...)
        slot_ids = torch.arange(self.Sa, device=tokens.device)
        slot_emb = self.slot_embed(slot_ids)[None, None, :, :]   # (1,1,Sa,D)
        tokens = tokens + slot_emb

        return tokens


