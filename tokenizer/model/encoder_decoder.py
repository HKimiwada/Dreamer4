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
import torch.utils.checkpoint as checkpoint
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
    """
    Memory-optimized masked autoencoding transformer (for DreamerV4 tokenizer training).

    Features:
      - Gradient checkpointing for lower memory usage.
      - Optional FlashAttention (if installed).
      - Same architecture and forward signature as baseline version.

    Args:
        input_dim: dimension of input patch tokens
        embed_dim: internal transformer embedding dimension
        num_heads: number of attention heads
        num_layers: layers per encoder/decoder stack
        latent_dim: latent bottleneck dimension
        use_checkpoint: enable gradient checkpointing
    """

    def __init__(
        self,
        input_dim: int = 768,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 12,
        latent_dim: int = 256,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(embed_dim))
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        # --- Encoder stack ---
        encoder_blocks = []
        for i in range(num_layers):
            causal_time = (i % 4 == 3)
            encoder_blocks.append(BlockCausalTransformer(embed_dim, num_heads, causal_time))
        self.encoder = nn.ModuleList(encoder_blocks)

        # --- Latent bottleneck ---
        self.to_latent = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.Tanh()
        )
        self.from_latent = nn.Linear(latent_dim, embed_dim)

        # --- Decoder stack ---
        decoder_blocks = []
        for i in range(num_layers):
            causal_time = (i % 4 == 3)
            decoder_blocks.append(BlockCausalTransformer(embed_dim, num_heads, causal_time))
        self.decoder = nn.ModuleList(decoder_blocks)

        self.output_proj = nn.Linear(embed_dim, input_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, 100000, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.5)
        self._max_len = 100000

        # --- Optional FlashAttention (PyTorch >=2.1) ---
        self.use_flash_attention = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if self.use_flash_attention:
            print("[INFO] FlashAttention supported — using efficient attention kernels.")
        else:
            print("[WARN] FlashAttention not available — using standard attention.")

    # ------------------------------------------------------------------
    def _run_stack(self, x, stack, T, N):
        """
        Helper: runs transformer stack with optional checkpointing.
        Now passes T and N for proper spatial/temporal attention.
        """
        for layer in stack:
            if self.use_checkpoint:
                # Checkpointing with T and N
                x = checkpoint.checkpoint(
                    layer, x, T, N, 
                    use_reentrant=False
                )
            else:
                x = layer(x, T, N)
        return x

    def forward(self, patch_tokens, mask):
        """
        Args:
            patch_tokens: (B, T, N, D_in)
            mask: (B, T, N) boolean tensor
        Returns:
            reconstructed_tokens: (B, T, N, D_in)
        """
        print("Using CausalTokenizer2")
        B, T, N, D_in = patch_tokens.shape
        x = self.input_proj(patch_tokens)  # (B, T, N, embed_dim)

        # Replace masked patches
        mask_exp = mask.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim)
        x = torch.where(mask_exp, self.mask_token.view(1, 1, 1, -1), x)

        # Flatten spatial+temporal dims
        x = x.view(B, T * N, self.embed_dim)

        # Positional Encoding
        seq_len = T * N
        x = x + self.pos_embed[:, :seq_len, :]

        # Encoder stack - NOW WITH T AND N
        x = self._run_stack(x, self.encoder, T, N)

        # Bottleneck
        x = x.view(B, T, N, self.embed_dim)
        x = self.to_latent(x)
        x = self.from_latent(x)

        # Decoder stack - NOW WITH T AND N
        x = x.view(B, T * N, self.embed_dim)
        x = self._run_stack(x, self.decoder, T, N)

        # Reconstruct
        x = x.view(B, T, N, self.embed_dim)
        reconstructed_tokens = self.output_proj(x)

        return reconstructed_tokens
