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