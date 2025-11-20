# Transformer blocks for World Model (reused from tokenizer)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import math
import copy

class RMSNorm(nn.Module):
    # Normalization layer to normalize input vectors and stabilize training. 
    def __init__(self, input_size, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(input_size))

    def forward(self, x):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x / torch.sqrt(ms + self.eps)
        output = x_normed * self.scale
        return output

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Project up to 2 * hidden_size for splitting
        self.up = nn.Linear(input_size, 2 * hidden_size)
        self.down = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(0.0)
    
    def forward(self, x):
        # Split the projection into two halves
        a, b = self.up(x).chunk(2, dim=-1)
        # SwiGLU activation: a * sigmoid(b)
        x = a * torch.sigmoid(b)
        # Project back down
        x = self.dropout(self.down(x))
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module with optional causal masking 
    (causal: by toggling on can mask future frames, by toggling off can disable masking when looking at one frame).
    """
    def __init__(self, input_size, num_heads, causal: bool):
        super().__init__()
        assert input_size % num_heads == 0 , "input_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        self.w_q = nn.Linear(input_size, input_size) # Query projection
        self.w_k = nn.Linear(input_size, input_size) # Key projection
        self.w_v = nn.Linear(input_size, input_size) # Value projection
        self.w_out = nn.Linear(input_size, input_size) # Output projection
        self.dropout = nn.Dropout(0.0)
        self.causal = causal # whether to apply temporal causal masking (not allow model to attend future frames)
  
    def _causal_masking_function(self, seq_len):
        """
        Small helper function used in temporal attention layers.
        Creates mask that ensures the model only attends to past (and current) time step frames.
        """
        # Create a causal mask for attention
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        return mask  # Shape: (seq_len, seq_len)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # reshape to (B, heads, L, d)
        Q = Q.view(batch_size, self.num_heads, seq_len, self.head_dim)
        K = K.view(batch_size, self.num_heads, seq_len, self.head_dim)
        V = V.view(batch_size, self.num_heads, seq_len, self.head_dim)

        # use SDPA (FlashAttention path if available) – no explicit scores tensor
        # When temporal causal is on, let SDPA do causal masking
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=self.causal  # True for temporal layers, False for spatial
        )
        # back to (B, L, input_size)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_out = self.dropout(self.w_out(attn_out))
        return attn_out

class BlockCausalTransformer(nn.Module):
    """
    Transformer block with separate spatial and temporal attention.
    - Spatial layers: full attention within each frame
    - Temporal layers: causal attention across time
    
    Positional embeddings are handled by the main model (CausalTokenizer),
    not within individual transformer blocks.
    """
    def __init__(self, input_size, num_heads, causal_time: bool):
        super().__init__()
        self.norm1 = RMSNorm(input_size)
        self.norm2 = RMSNorm(input_size)
        self.attn = MultiHeadAttention(input_size, num_heads, causal_time)
        self.ffn = FeedForward(input_size, hidden_size=int(4*input_size))
        self.causal_time = causal_time
        self.input_size = input_size
       
    def forward(self, x, T=None, N=None):
        """
        Args:
            x: (B, T*N, D) - flattened sequence
            T: number of timesteps
            N: number of tokens per timestep
        """
        B, seq_len, D = x.shape
        
        # If T and N not provided, treat as standard attention
        if T is None or N is None:
            norm_x = self.norm1(x)
            attn_output = self.attn(norm_x)
            x = x + attn_output  
            norm_x = self.norm2(x)
            ffn_output = self.ffn(norm_x)
            x = x + ffn_output
            return x
        
        # Factorized spatial/temporal attention
        if not self.causal_time:
            # SPATIAL LAYER: attend within each frame
            # Reshape to (B, T, N, D) then (B*T, N, D)
            x = x.view(B, T, N, D)
            
            # Flatten temporal dimension into batch for independent frame processing
            x_reshaped = x.reshape(B * T, N, D)
            
            # Apply attention within each frame
            norm_x = self.norm1(x_reshaped)
            attn_output = self.attn(norm_x)
            x_reshaped = x_reshaped + attn_output
            
            # Feedforward
            norm_x = self.norm2(x_reshaped)
            ffn_output = self.ffn(norm_x)
            x_reshaped = x_reshaped + ffn_output
            
            # Reshape back to (B, T*N, D)
            x = x_reshaped.view(B, T, N, D).reshape(B, T * N, D)
            
        else:
            # TEMPORAL LAYER: attend across time (causal)
            # Reshape to (B, T, N, D) then permute to (B, N, T, D)
            x = x.view(B, T, N, D).permute(0, 2, 1, 3)  # (B, N, T, D)
            
            # Flatten spatial dimension into batch for per-position temporal processing
            x_reshaped = x.reshape(B * N, T, D)
            
            # Apply causal attention across time
            norm_x = self.norm1(x_reshaped)
            attn_output = self.attn(norm_x)
            x_reshaped = x_reshaped + attn_output
            
            # Feedforward
            norm_x = self.norm2(x_reshaped)
            ffn_output = self.ffn(norm_x)
            x_reshaped = x_reshaped + ffn_output
            
            # Reshape back to (B, T*N, D)
            x = x_reshaped.view(B, N, T, D).permute(0, 2, 1, 3).reshape(B, T * N, D)
        
        return x
