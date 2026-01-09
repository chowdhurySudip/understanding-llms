"""Model checkpointing utilities for training persistence.

Provides functions to save and restore training state, enabling:
- Recovery from interruptions
- Model evaluation at different training stages
- Distributed training synchronization
"""

import os
import typing
import torch

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    """Save model and optimizer state to a checkpoint file.
    
    Creates a checkpoint containing all information needed to resume training:
    - Model weights and buffers (model.state_dict())
    - Optimizer state including momentum buffers (optimizer.state_dict())
    - Current iteration number for tracking progress
    
    Args:
        model: The neural network model to save
        optimizer: The optimizer with its current state
        iteration: Current training iteration/step number
        out: Output path (file path or file-like object) for the checkpoint
        
    Example:
        >>> model = TransformerLM(...)
        >>> optimizer = AdamW(...)
        >>> save_checkpoint(model, optimizer, iteration=1000, out='checkpoint_1000.pt')
    """
    # Bundle all training state into a single dictionary
    checkpoint = {
        "model_state": model.state_dict(),      # Model parameters
        "optimizer_state": optimizer.state_dict(),  # Optimizer state (momentums, etc.)
        "iteration": iteration                   # Training progress
    }
    
    # Save to disk using PyTorch's serialization
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer
) -> int:
    """Load model and optimizer state from a checkpoint file.
    
    Restores the complete training state from a previously saved checkpoint,
    allowing training to resume from the exact point where it was saved.
    
    Args:
        src: Source path (file path or file-like object) of the checkpoint
        model: Model instance to load weights into (must match saved architecture)
        optimizer: Optimizer instance to load state into (must match saved type)
        
    Returns:
        The iteration number at which the checkpoint was saved
        
    Raises:
        RuntimeError: If checkpoint architecture doesn't match current model
        
    Example:
        >>> model = TransformerLM(...)
        >>> optimizer = AdamW(...)
        >>> iteration = load_checkpoint('checkpoint_1000.pt', model, optimizer)
        >>> print(f"Resuming from iteration {iteration}")
    """
    # Load checkpoint from disk
    checkpoint = torch.load(src)
    
    # Restore model parameters (weights and buffers)
    model.load_state_dict(checkpoint["model_state"])
    
    # Restore optimizer state (momentum buffers, etc.)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    # Return the iteration number for resuming training
    return checkpoint["iteration"]