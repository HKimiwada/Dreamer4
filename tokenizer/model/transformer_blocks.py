# python tokenizer/model/transformer_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import math
import copy

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x / torch.sqrt(ms + self.eps)
        output = x_normed * self.scale
        return output

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeedForward, self).__init__()
        # Project up to 2 * hidden_size for splitting
        self.up = nn.Linear(input_size, 2 * hidden_size)
        self.down = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Split the projection into two halves
        a, b = self.up(x).chunk(2, dim=-1)
        # SwiGLU activation: a * sigmoid(b)
        x = a * torch.sigmoid(b)
        # Project back down
        x = self.dropout(self.down(x))
        return x

def causal_masking_function(seq_len):
    """
    Small helper function used in temporal attention layers.
    Creates mask that ensures the model only attends to past (and current) time step frames.
    """
    # Create a causal mask for attention
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return mask  # Shape: (seq_len, seq_len)

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads, causal: bool, causal_masking_function):
        super(MultiHeadAttention, self).__init__()
        assert input_size % num_heads == 0 , "input_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        self.w_q = nn.Linear(input_size, input_size) # Query projection
        self.w_k = nn.Linear(input_size, input_size) # Key projection
        self.w_v = nn.Linear(input_size, input_size) # Value projection
        self.w_out = nn.Linear(input_size, input_size) # Output projection
        self.dropout = nn.Dropout(0.1)
        self.causal = causal # whether to apply temporal causal masking (not allow model to attend future frames)
        self.causal_masking_function = causal_masking_function

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # Project inputs to Q, K, V (batch_size, seq_len, input_size)
        Q = self.w_q(x) 
        K = self.w_k(x)
        V = self.w_v(x)
        # Reshape QKV into heads (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, self.num_heads, seq_len, self.head_dim)
        K = K.view(batch_size, self.num_heads, seq_len, self.head_dim)
        V = V.view(batch_size, self.num_heads, seq_len, self.head_dim)
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            mask = self.causal_masking_function(seq_len).to(x.device)  # (seq_len, seq_len)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        # Concatenate heads and project out
        attn_output = attn_output.view(batch_size, seq_len, -1)  # (batch_size, seq_len, input_size)
        attn_output = self.dropout(self.w_out(attn_output))
        return attn_output