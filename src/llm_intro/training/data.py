"""Data loading utilities for language model training.

Provides functions for creating random batches from tokenized datasets.
"""

import torch
import numpy as np
import numpy.typing as npt

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of sequences from a tokenized dataset.
    
    Creates training pairs (X, Y) where Y contains the next-token targets for X.
    Each sequence is randomly sampled from the dataset using memory-mapped I/O
    for efficiency with large datasets.
    
    Args:
        dataset: 1D numpy array of token indices (memory-mapped for large files)
        batch_size: Number of sequences to sample in the batch
        context_length: Length of each sequence (number of tokens)
        device: Device to place tensors on ("cpu", "cuda", etc.)
        
    Returns:
        tuple containing:
            - X: Input sequences of shape (batch_size, context_length)
            - Y: Target sequences of shape (batch_size, context_length)
                 where Y[i, j] = X[i, j+1] (next token prediction)
    
    Example:
        >>> dataset = np.memmap('tokens.npy', dtype=np.int64, mode='r')
        >>> X, Y = get_batch(dataset, batch_size=32, context_length=128, device='cuda')
        >>> assert X.shape == (32, 128)
        >>> assert torch.all(Y[:, :-1] == X[:, 1:])  # Y is shifted by 1
    """
    # Calculate valid starting positions (must leave room for context_length)
    n = dataset.shape[-1]
    max_id = n - context_length
    
    # Sample random starting indices for each sequence in the batch
    batch_indices = torch.randint(0, max_id, size=(batch_size,))
    
    # Extract input sequences: X[i] = dataset[start:start+context_length]
    X = torch.from_numpy(
        np.stack([dataset[ix : ix+context_length] for ix in batch_indices])
    ).to(dtype=torch.long, device=device)
    
    # Extract target sequences (shifted by 1): Y[i] = dataset[start+1:start+1+context_length]
    Y = torch.from_numpy(
        np.stack([dataset[ix+1 : ix+1+context_length] for ix in batch_indices])
    ).to(dtype=torch.long, device=device)
    
    return (X, Y)