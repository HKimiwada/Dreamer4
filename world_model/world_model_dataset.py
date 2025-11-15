# Building LatentDataset class to feed world model.
import torch
from torch.utils.data import Dataset
from pathlib import Path

class LatentSequenceDataset(Dataset):
    """
    Convert data in root (data/latent_sequences) into dataset that can be fed into WM.
    context: context length (i.e. how many latent representations the WM has access to), 192 frames -> 9.6 seconds at 20 FPS
    """
    def __init__(self, root, context=192):
        self.files = sorted(list(Path(root).glob("*.pt")))
        self.context = context

    def __getitem__(self. idx):
        data = torch.load(self.files[idx])
        z = data["z"]
        T = z.shape[0]

        if T < self.context:
            pad = self.context - T
            z = torch.cat([z, z[-1:].repeat(pad,1,1)], dim=0)
