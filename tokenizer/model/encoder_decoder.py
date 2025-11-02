"""
High-level overview:
    Given (patch_tokens, mask), how does the model encode visible patches, 
    compress them temporally, and then reconstruct the missing ones?
encoder_decoder.py:
    Defines how the tokenizer as a whole works (high-level pipeline). 
    Uses modules defined in transformer_blocks.py. 
Conceptual Overview:
    1. Input (masked patches)
    2. Stack of spatial + temporal blocks (encoder)
    3. tanh bottleneck
    4. Stack of spatial + temporal blocks (decoder)
    5. Linear projection 
    6. Output (reconstructed patches)
"""
import torch
import torch.nn as nn
from tokenizer.model.transformer_blocks import BlockCausalTransformer

import torch
import torch.nn as nn
from tokenizer.model.transformer_blocks import BlockCausalTransformer

class CausalTokenizer(nn.Module):
    """
    A masked-autoencoding encoder–decoder transformer that learns to reconstruct
    masked video patch tokens.

    Encoder learns to embed visible patches into compressed latent space (for processing by world model)
    Decoder provides self-supervised learning signal by reconstructing original patches (not used in actual world model)

    Architecture:
      - mask_token (learnable embedding for masked patches)
      - encoder: stack of BlockCausalTransformer layers (spatial + temporal)
      - latent bottleneck: linear + tanh projection
      - decoder: another stack of transformer blocks
      - output projection: linear mapping back to patch embedding dimension

    Args:
        input_dim: dimensionality of patch tokens (e.g., 768)
        embed_dim: transformer hidden dimension
        num_heads: number of attention heads per block
        num_layers: total number of layers per stack (encoder and decoder)
        latent_dim: dimension of bottleneck (latent projection)
        causal_masking_function: function to generate lower-triangular masks
    """

    def __init__(self, input_dim=768, embed_dim=768, num_heads=8, num_layers=12, latent_dim=256):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(embed_dim)) # Learnable mask embedding (used to replace masked patches)
        self.input_proj = nn.Linear(input_dim, embed_dim) # Linear projection to embedding space (patch_dim → embed_dim)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Encoder stack: alternating spatial/temporal transformer blocks
        encoder_blocks = []
        for i in range(num_layers):
            # every 4th block is temporal (causal=True)
            causal_time = (i % 4 == 3)
            encoder_blocks.append(
                BlockCausalTransformer(embed_dim, num_heads, causal_time)
            )
        self.encoder = nn.ModuleList(encoder_blocks)

        # Latent projection bottleneck (linear + tanh) 
        self.to_latent = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.Tanh()
        )
        
        # Decoder stack: same architecture as encoder 
        decoder_blocks = []
        for i in range(num_layers):
            causal_time = (i % 4 == 3)
            decoder_blocks.append(
                BlockCausalTransformer(embed_dim, num_heads, causal_time)
            )
        self.decoder = nn.ModuleList(decoder_blocks) 

        self.from_latent = nn.Linear(latent_dim, embed_dim) # Linear projection from latent back to embedding space.
        self.output_proj = nn.Linear(embed_dim, input_dim) # Linear projection from embedding space back to patch dimension
    
    def forward(self, patch_tokens, mask):
        """
        Forward pass through the Causal Tokenizer.
        Args:
            patch_tokens: Tensor of shape (B, T, N, D_in) - input patch tokens
            mask: Boolean Tensor of shape (B, T, N) - True for masked patches
                B: batch size
                T: number of frames per clip
                N: patches per frame (e.g., 24×40 = 960)
                D_in: patch embedding dimension (e.g., 768)
        Returns:
            reconstructed_tokens: Tensor of shape (B, T, N, D_in) - reconstructed patch tokens
        """
        B, T, N, D_in = patch_tokens.shape
        # Project to embed space
        x = self.input_proj(patch_tokens)               # (B, T, N, embed_dim)

        # Replace masked patches
        mask_exp = mask.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim)
        x = torch.where(mask_exp, self.mask_token.view(1, 1, 1, -1), x)

        # Flatten space–time
        x = x.view(B, T * N, self.embed_dim)

        # Encode
        for layer in self.encoder:
            x = layer(x)

        # Unflatten back to frames for bottleneck
        x = x.view(B, T, N, self.embed_dim)
        x = self.to_latent(x)
        x = self.from_latent(x)

        # Flatten again for decoder
        x = x.view(B, T * N, self.embed_dim)
        for layer in self.decoder:
            x = layer(x)

        # Reshape and project out
        x = x.view(B, T, N, self.embed_dim)
        reconstructed_tokens = self.output_proj(x)
        return reconstructed_tokens
