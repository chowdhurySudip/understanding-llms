import torch
import torch.nn as nn

from .layers import Linear, SwiGLU, softmax, RMSNorm, Embedding
from .attention import MultiheadSelfAttention

class TransformerBlock(nn.Module):
    """
    Transformer block with pre-normalization architecture.
    
    Implements a single transformer layer following the pre-norm architecture:
    1. Layer normalization followed by multi-head self-attention with residual connection
    2. Layer normalization followed by position-wise feedforward network with residual connection
    
    This uses RMSNorm for layer normalization and SwiGLU for the feedforward network,
    following modern transformer architectures like LLaMA.
    
    Args:
        d_model: Dimension of the model (embedding dimension)
        num_heads: Number of attention heads
        d_ff: Hidden dimension of the feedforward network
        **kwargs: Additional keyword arguments passed to MultiheadSelfAttention
            (e.g., add_rope, theta, max_seq_len for rotary positional embeddings)
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, **kwargs) -> None:
        super(TransformerBlock, self).__init__()

        device = kwargs.get("device", torch.device("cpu"))
        
        # Multi-head self-attention module
        self.mha = MultiheadSelfAttention(d_model, num_heads, **kwargs)
        
        # Position-wise feedforward network using SwiGLU activation
        self.positionwise_ffn = SwiGLU(d_model, d_ff, device=device)
        
        # Layer normalization modules (RMSNorm)
        self.norm1 = RMSNorm(d_model, device=device)  # Pre-normalization for attention
        self.norm2 = RMSNorm(d_model, device=device)  # Pre-normalization for feedforward

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Applies the following operations in sequence:
        1. x = x + MultiHeadAttention(RMSNorm(x))
        2. x = x + FeedForward(RMSNorm(x))
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Optional tensor of token positions for RoPE, 
                           shape (batch_size, seq_len). If None, uses sequential positions [0, 1, 2, ...]
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Generate default sequential token positions if not provided
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        
        # First sublayer: Pre-norm + Multi-head self-attention + Residual connection
        x = x + self.mha(self.norm1(x), token_positions)
        
        # Second sublayer: Pre-norm + Position-wise feedforward + Residual connection
        x = x + self.positionwise_ffn(self.norm2(x))
        
        return x

class TransformerLM(nn.Module):
    """
    Transformer Language Model for autoregressive text generation.
    
    Implements a decoder-only transformer architecture following modern designs like GPT.
    The model uses:
    - Token embeddings (no positional embeddings - positions are encoded via RoPE)
    - Multiple transformer blocks with pre-normalization
    - RoPE (Rotary Position Embeddings) for positional information
    - RMSNorm for layer normalization
    - SwiGLU activation in feedforward networks
    - Final layer normalization before output projection
    
    Args:
        vocab_size: Size of the vocabulary (number of unique tokens)
        context_length: Maximum sequence length the model can process
        d_model: Dimension of the model (embedding dimension)
        num_layers: Number of transformer blocks to stack
        num_heads: Number of attention heads in each transformer block
        d_ff: Hidden dimension of the feedforward network in each block
        rope_theta: Base frequency (θ) parameter for Rotary Position Embeddings
    """
    
    def __init__(
            self, vocab_size: int,
            context_length: int, d_model: int,
            num_layers: int, num_heads: int, 
            d_ff: int | None, rope_theta: float,
            **kwargs
    ):
        super(TransformerLM, self).__init__()

        device = kwargs.get("device", torch.device("cpu"))
        
        # Token embedding layer: maps token indices to dense vectors
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        
        # Stack of transformer blocks with RoPE-enabled attention
        # Using ModuleList ensures PyTorch properly registers these as submodules
        self.attention_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                add_rope=True,
                theta=rope_theta,
                max_seq_len=context_length,
                **kwargs
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization before output projection
        self.norm = RMSNorm(d_model=d_model, device=device)
        
        # Language model head: projects hidden states to vocabulary logits
        self.ffn = Linear(d_model, vocab_size, device=device)

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer language model.
        
        Processes input token indices through the following pipeline:
        1. Token embedding lookup
        2. Sequential processing through all transformer blocks
        3. Final layer normalization
        4. Projection to vocabulary logits
        
        Args:
            token_indices: Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
            representing unnormalized log probabilities for next token prediction
        """
        # Embed input tokens: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.token_embeddings(token_indices)
        
        # Process through each transformer block sequentially
        # Each block applies self-attention and feedforward transformations
        for attn_block in self.attention_blocks:
            x = attn_block(x)
        
        # Apply final layer normalization
        x = self.norm(x)
        
        # Project to vocabulary size to get logits for next token prediction
        # Output shape: (batch_size, seq_len, vocab_size)
        out = self.ffn(x)
        
        return out