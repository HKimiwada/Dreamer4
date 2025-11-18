# Data Loader to prepare data for input into WM.
# Input: latent tokens, numerically-converted actions (from wm_dataset)
# Output: (latent tokens, tokenized actions, register tokens, short cut token) for each timestep
import torch
from torch.nn as nn
import numpy as np
import random
from world_model.wm_preprocessing.wm_dataset import WorldModelDataset
from world_model.wm_preprocessing.action_tokenier import ActionTokenizer
 
class DataBuilderWM():
    """
    Generates sequence to feed into WM model.
    """
    def __init__(
        self,
        d_model: int,
        latent_dir: str = "data/latent_sequences",
        action_jsonl: str = "data/actions.jsonl",
        register_tokens: int = 8, # small learned vectors that help the transformer aggregate information. (Like a scratch pad of sorts)
        k_max: int = 64,
    )
        self.d_model = d_model
        self.latent_dir = latent_dir
        self.action_jsonl = action_jsonl
        self.register_tokens = register_tokens
        self.short_cut_token = short_cut_token

        self.register_embed = nn.Embedding(register_tokens, d_model)
        self.k_max = k_max 
        K = 2 ** self.k_max
        i = np.random.randint(0, k_max + 1)
        self.step_size = 2 ** i / K
        self.tau_values = [n * self.step_size for n in range(int(1/self.step_size)+ 1)] # Build list of potential tau grid
        self.tau_t = random.choice(self.tau_values)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self):
        dataset = WorldModelDataset(
            latent_dir=self.latent_dir,
            action_jsonl=self.action_jsonl,
            clip_length=8,
            device=self.device
        )
        tokenizer = ActionTokenizer(self.d_model).to(self.device)
        for i in range(len(dataset)):
            sample = dataset[i]
            lat = sample["latents"]
            act = sample["actions"]

            action_token = tokenizer(act)
            register_ids = [n for n in range(self.register_tokens)]
            register_token = self.register_embed()

            

        


