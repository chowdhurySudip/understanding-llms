"""
Attention mechanisms for transformer models.

This module implements scaled dot-product attention and multi-head self-attention,
with optional rotary positional embeddings (RoPE).
"""

import torch
import torch.nn as nn

from .layers import Linear, softmax, RotaryPositionalEmbedding

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q: Query tensor of shape (..., seq_len, d_k)
        K: Key tensor of shape (..., seq_len, d_k)
        V: Value tensor of shape (..., seq_len, d_v)
        mask: Optional boolean mask tensor where True indicates positions to attend to.
              If None, applies causal (lower triangular) masking.
    
    Returns:
        Attention output tensor of shape (..., seq_len, d_v)
    """
    # Get the dimension of the key vectors for scaling
    d_k = Q.size()[-1]
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    attention_scores = (Q @ K.transpose(-1, -2)) / d_k**0.5
    
    # Apply masking
    if mask is not None:
        # Convert boolean mask: True -> 0 (attend), False -> -inf (mask out)
        mask = torch.where(mask, 0., -torch.inf)
    else:
        # Default to causal masking (upper triangular positions set to -inf)
        mask = torch.triu(torch.ones_like(attention_scores) * -torch.inf, diagonal=1)
    
    # Apply softmax to get attention weights
    masked_attention_weights = softmax(attention_scores + mask, -1)
    
    # Compute weighted sum of values
    weighted_sum = masked_attention_weights @ V
    return weighted_sum

class MultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention module.
    
    Implements the multi-head attention mechanism from "Attention Is All You Need"
    with optional rotary positional embeddings (RoPE).
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        **kwargs: Optional keyword arguments:
            - add_rope (bool): Whether to apply rotary positional embeddings. Default: False
            - theta (int): Base frequency for RoPE (required if add_rope=True)
            - max_seq_len (int): Maximum sequence length for RoPE (required if add_rope=True)
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        **kwargs
    ) -> None:
        super(MultiheadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        device = kwargs.get("device", torch.device("cpu"))
        
        # Initialize linear projection layers for queries, keys, values, and output
        # Each projects from d_model to d_model dimensions
        self.W_q = Linear(d_model, d_model, device=device)
        self.W_k = Linear(d_model, d_model, device=device)
        self.W_v = Linear(d_model, d_model, device=device)
        self.W_o = Linear(d_model, d_model, device=device)
        
        # Optional rotary positional embeddings
        self.add_rope = kwargs.get("add_rope", False)
        if self.add_rope:
            # Get theta and max_seq_len, which are required when add_rope is True
            theta = kwargs.get("theta")
            max_seq_len = kwargs.get("max_seq_len")
            assert theta is not None, "theta is required when add_rope=True"
            assert max_seq_len is not None, "max_seq_len is required when add_rope=True"
            self.theta: int = theta
            self.max_seq_len: int = max_seq_len
            self.rope = RotaryPositionalEmbedding(self.theta, self.head_dim, self.max_seq_len)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Optional tensor of token positions for RoPE, shape (batch_size, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Get input dimensions
        batch, seq_len, _ = x.shape

        # Project to Q, K, V and reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)
        Q = self.W_q(x)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary positional embeddings if enabled
        if self.add_rope:
            # RoPE expects (num_heads, batch, seq_len, head_dim), so transpose first
            Q = self.rope(Q.transpose(0, 1), token_positions)
            # Transpose back to (batch, num_heads, seq_len, head_dim)
            Q = Q.transpose(0, 1)
            K = self.rope(K.transpose(0, 1), token_positions)
            K = K.transpose(0, 1)
        
        V = self.W_v(x)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply scaled dot-product attention
        Z = scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        batch, num_heads, seq_len, head_dim_v = Z.shape
        Z = Z.transpose(-3, -2).reshape(batch, seq_len, num_heads * head_dim_v)
        
        # Apply output projection
        out = self.W_o(Z)
        return out

