"""Basic layers and primitives for the Transformer model.

This module consolidates various layer definitions including:
- Linear: Dense layer with truncated normal initialization
- Activations: SiLU, SwiGLU, Softmax
- Normalization: RMSNorm
- Embeddings: Standard Embedding, RotaryPositionalEmbedding
"""

import torch
import torch.nn as nn
import numpy as np

# -----------------------------------------------------------------------------
# Linear Layer
# -----------------------------------------------------------------------------

class Linear(nn.Module):
    """Linear transformation layer (fully-connected layer without bias).
    
    Implements y = x @ W^T where W is the weight matrix.
    
    Uses truncated normal initialization with Xavier/Glorot-style scaling:
    std = sqrt(2 / (in_features + out_features))
    Values truncated to [-3*std, 3*std].
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        device: Device to place the parameters on (CPU/CUDA)
        dtype: Data type of the parameters
    """
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        """Initialize linear layer with truncated normal weights.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            device: Device for parameters (optional)
            dtype: Data type for parameters (optional)
        """
        super(Linear, self).__init__()
        # Compute standard deviation using Xavier/Glorot initialization
        # This helps maintain gradient scale across layers
        std = np.sqrt(2/(in_features+out_features))
        
        # Initialize weight matrix with truncated normal distribution
        # Shape: (out_features, in_features) for efficient matrix multiplication
        self.W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    (out_features, in_features), dtype=dtype, device=device
                ), std=std, a=-3*std, b=3*std
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Compute x @ W^T (transpose W for correct dimensions)
        return x @ self.W.T

# -----------------------------------------------------------------------------
# Activation Functions
# -----------------------------------------------------------------------------

class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU) activation function.
    
    Also known as Swish, this activation function is defined as:
    SiLU(x) = x * sigmoid(x)
    
    This provides smooth, non-monotonic behavior that often works well
    in deep neural networks.
    """
    
    def __init__(self) -> None:
        """Initialize SiLU activation."""
        super(SiLU, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SiLU activation.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Output tensor with same shape as input, with SiLU applied element-wise
        """
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """SwiGLU activation-based feed-forward network.
    
    Implements the Gated Linear Unit (GLU) variant with SiLU activation,
    as described in "GLU Variants Improve Transformer" (Shazeer, 2020).
    
    The computation is: SwiGLU(x) = (SiLU(W1 @ x) ⊙ W3 @ x) @ W2
    where ⊙ denotes element-wise multiplication.
    
    Args:
        d_model: Input and output dimension of the model
        d_ff: Hidden dimension of the feed-forward layer. If None, automatically
              computed as round_up(8/3 * d_model) to nearest multiple of 64.
    """
    
    def __init__(self, d_model: int, d_ff: int | None, device: torch.device) -> None:
        """Initialize SwiGLU feed-forward network.
        
        Args:
            d_model: Input and output dimension
            d_ff: Hidden dimension of the feed-forward layer. If None, computed
                  automatically using compute_d_ff method.
        """
        super(SwiGLU, self).__init__()

        # If d_ff not provided, compute it automatically
        if d_ff is None:
            d_ff = self._compute_d_ff(d_model)
        
        # First projection: d_model -> d_ff
        self.ffn1 = Linear(d_model, d_ff, device=device)
        # Output projection: d_ff -> d_model
        self.ffn2 = Linear(d_ff, d_model, device=device)
        # Gate projection: d_model -> d_ff
        self.ffn3 = Linear(d_model, d_ff, device=device)
        # SiLU activation
        self.silu = SiLU()
    
    def _compute_d_ff(self, d_model: int) -> int:
        """Compute hidden dimension d_ff based on d_model.
        
        Uses the formula d_ff = round_up(8/3 * d_model) to the nearest multiple of 64.
        This ensures efficient computation on modern hardware that benefits from
        aligned memory access patterns.

        Args:
            d_model: Input/output dimension of the model
            
        Returns:
            Computed d_ff value rounded up to nearest multiple of 64
        """
        raw = int(round(8 / 3 * d_model))
        # Round up to nearest multiple of 64
        d_ff = (raw + 63) // 64 * 64
        return d_ff

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
        """
        # Apply gated linear unit with SiLU: SiLU(W1 @ x) * (W3 @ x)
        # Then project back to d_model with W2
        return self.ffn2(self.silu(self.ffn1(x)) * self.ffn3(x))

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute numerically stable softmax.
    
    Implements softmax(x_i) = exp(x_i) / sum_j(exp(x_j)) with numerical stability
    by subtracting the maximum value before exponentiation to prevent overflow.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to apply softmax
        
    Returns:
        Output tensor with same shape as input, with softmax applied along specified dimension.
        Values along dim sum to 1.
    """
    # Subtract max for numerical stability (prevents overflow in exp)
    x_exp = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    # Normalize by sum to get probabilities
    x_soft = x_exp/x_exp.sum(dim=dim, keepdim=True)
    return x_soft

# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Implements RMSNorm as described in "Root Mean Square Layer Normalization"
    (Zhang & Sennrich, 2019). RMSNorm normalizes activations using only the
    root mean square statistic, without centering (no mean subtraction).
    
    The normalization is computed as:
    RMSNorm(x) = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    Args:
        d_model: Dimension of the input features to normalize
        eps: Small constant for numerical stability (prevents division by zero)
        device: Device to place the parameters on (CPU/CUDA)
        dtype: Data type of the parameters
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        """Initialize RMSNorm layer.
        
        Args:
            d_model: Feature dimension to normalize over
            eps: Epsilon for numerical stability (default: 1e-5)
            device: Device for parameters (optional)
            dtype: Data type for parameters (optional)
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        # Learnable scale parameter (initialized to ones)
        self.weight = nn.Parameter(torch.ones((d_model,), dtype=dtype, device=device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to input tensor.
        
        Computation is done in float32 for numerical stability,
        then converted back to input dtype.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Store original dtype for conversion back later
        in_dtype = x.dtype
        # Convert to float32 for stable computation
        x = x.to(torch.float32)
        # Compute RMS: sqrt(mean(x^2) + eps)
        norm = torch.sqrt(x.square().mean(-1, keepdim=True) + self.eps)
        # Normalize and scale by learnable weight
        output = x * self.weight / norm
        # Convert back to original dtype
        return output.to(in_dtype)

# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

class Embedding(nn.Module):
    """Token embedding layer with truncated normal initialization.
    
    Maps discrete token IDs to continuous vector representations.
    Uses truncated normal initialization in the range [-3, 3] standard deviations.
    
    Args:
        num_embeddings: Size of the vocabulary (number of unique tokens)
        embedding_dim: Dimension of the embedding vectors
        device: Device to place the parameters on (CPU/CUDA)
        dtype: Data type of the parameters
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """Initialize embedding layer.
        
        Args:
            num_embeddings: Vocabulary size
            embedding_dim: Dimension of embedding vectors
            device: Device for parameters (optional)
            dtype: Data type for parameters (optional)
        """
        super(Embedding, self).__init__()
        # Initialize embedding matrix with truncated normal distribution
        # Truncated to [-3, 3] standard deviations to prevent extreme values
        self.emb = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    (num_embeddings, embedding_dim), dtype=dtype, device=device
                ), a=-3, b=3
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for given token IDs.
        
        Args:
            token_ids: Tensor of token indices, shape (...)
            
        Returns:
            Embedding tensor of shape (..., embedding_dim)
        """
        return self.emb[token_ids]

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) layer.
    
    Implements RoPE as described in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021). RoPE encodes positional information by rotating query and key vectors
    in 2D subspaces, providing relative positional information.
    
    The rotation angles are computed as: θ_i = base^(-2i/d) where base is theta parameter.
    
    Args:
        theta: Base frequency for the rotation (typically 10000)
        d_k: Dimension of the key/query vectors (must be even)
        max_seq_len: Maximum sequence length to precompute rotations for
        device: Device to place the buffers on (CPU/CUDA)
    """
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """Initialize RoPE layer and precompute rotation angles.
        
        Args:
            theta: Base frequency parameter
            d_k: Dimension of embeddings (must be even)
            max_seq_len: Maximum sequence length
            device: Device for buffers (optional)
        """
        super(RotaryPositionalEmbedding, self).__init__()
        # Compute inverse frequencies for each dimension pair: theta^(-2i/d_k)
        pair_ids = torch.arange(0, d_k//2).unsqueeze(0).to(device)
        inv_freq = theta ** (-2*pair_ids/d_k)
        
        # Compute angles for all positions: pos * inv_freq
        positions = torch.arange(0, max_seq_len).unsqueeze(1).to(device)
        angle = positions * inv_freq
        
        # Precompute and cache cos and sin values
        cos_cached = angle.cos()
        sin_cached = angle.sin()
        # Register as buffers (not parameters) - won't be updated during training
        self.register_buffer("rope_cos", cos_cached, persistent=False)
        self.register_buffer("rope_sin", sin_cached, persistent=False)

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embeddings to input tensor.
        
        Rotates pairs of dimensions in x based on their positions.
        For each pair (x_2i, x_2i+1), applies 2D rotation:
        [x'_2i  ]   [cos(θ_i*pos)  -sin(θ_i*pos)] [x_2i  ]
        [x'_2i+1] = [sin(θ_i*pos)   cos(θ_i*pos)] [x_2i+1]
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices for each token, shape (..., seq_len)
            
        Returns:
            Rotated tensor with same shape as input (..., seq_len, d_k)
        """
        # Split into even and odd indices (pairs of dimensions to rotate)
        x_even = x[..., 0::2]  # Elements at indices 0, 2, 4, ...
        x_odd = x[..., 1::2]   # Elements at indices 1, 3, 5, ...
        
        # Get cached cos and sin values for the given positions
        cos = self.rope_cos[token_positions]
        sin = self.rope_sin[token_positions]
        
        # Match data type of input
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        
        # Apply 2D rotation to each dimension pair
        x_even_rot = x_even*cos - x_odd*sin
        x_odd_rot = x_even*sin + x_odd*cos
        
        # Interleave rotated even and odd elements back together
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_even_rot
        x_rot[..., 1::2] = x_odd_rot
        return x_rot
