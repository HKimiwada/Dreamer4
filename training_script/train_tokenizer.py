"""
Overview of Training script for tokenizer:
    1. Load video patch data from your preprocessed dataset (TokenizerDataset).
    2. Feed it into the tokenizer (encoder–decoder).
    3. Compute the reconstruction loss (MSE + 0.2×LPIPS) only on masked patches.
    4. Backpropagate, optimize, and log training metrics.

Goal:
A trained tokenizer whose encoder + tanh bottleneck produce stable, 
compact latents for the world model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from tokenizer.tokenizer_dataset import TokenizerDataset
from tokenizer.model.encoder_decoder import CausalTokenizer  
from tokenizer.losses import lpips_loss                      
