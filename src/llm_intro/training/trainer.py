"""Training loop example for transformer language model.

Demonstrates a complete training setup including:
- Data loading from memory-mapped files
- Model initialization
- Optimizer setup with AdamW
- Learning rate scheduling with warmup and cosine annealing
- Gradient clipping for stability
- Standard training loop

Note: This is a template/example. Replace file paths and hyperparameters
      with actual values for your training run.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import torch
import numpy as np
import yaml
import seaborn as sns
import matplotlib.pyplot as plt

from llm_intro.training.data import get_batch
from llm_intro.model.transformer import TransformerLM
from llm_intro.utils.loss import cross_entropy
from llm_intro.training.optimizer import AdamW, gradient_clipping, learning_rate_schedule
from llm_intro.utils.checkpointing import save_checkpoint, load_checkpoint


# =============================================================================
# Configuration Loading
# =============================================================================
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# =============================================================================
# Data Setup
# =============================================================================
def setup_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load training and validation data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_data, val_data) as memory-mapped numpy arrays
    """
    train_data = np.load(config['data']['train_path'], mmap_mode="r")
    val_data = np.load(config['data']['val_path'], mmap_mode="r")
    print(f"Loaded training data: {config['data']['train_path']}")
    print(f"Loaded validation data: {config['data']['val_path']}")
    return train_data, val_data


# =============================================================================
# Model Setup
# =============================================================================
def setup_model(config: Dict[str, Any]) -> TransformerLM:
    """Initialize transformer language model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized TransformerLM model
    """
    model_config = config['model']
    training_config = config['training']
    
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        context_length=training_config['context_length'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        rope_theta=model_config['rope_theta'],
        device=training_config['device']
    )
    
    model = model.to(training_config['device'])
    print(f"Initialized model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


# =============================================================================
# Optimizer Setup
# =============================================================================
def setup_optimizer(model: TransformerLM, config: Dict[str, Any]) -> AdamW:
    """Initialize AdamW optimizer.
    
    Args:
        model: Transformer model
        config: Configuration dictionary
        
    Returns:
        Initialized AdamW optimizer
    """
    opt_config = config['optimizer']
    
    optimizer = AdamW(
        model.parameters(),
        lr=opt_config['max_lr'],
        betas=tuple(opt_config['betas']),
        eps=opt_config['eps'],
        weight_decay=opt_config['weight_decay']
    )
    
    print(f"Initialized AdamW optimizer (lr={opt_config['max_lr']}, weight_decay={opt_config['weight_decay']})")
    return optimizer


# =============================================================================
# Training Step
# =============================================================================
def train_step(
    model: TransformerLM,
    optimizer: AdamW,
    train_data: np.ndarray,
    config: Dict[str, Any],
    current_lr: float
) -> float:
    """Execute a single training step.
    
    Args:
        model: Transformer model
        optimizer: Optimizer
        train_data: Training data array
        config: Configuration dictionary
        current_lr: Current learning rate
        
    Returns:
        Training loss value
    """
    training_config = config['training']
    opt_config = config['optimizer']
    
    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Get batch
    X, y = get_batch(
        train_data,
        training_config['batch_size'],
        training_config['context_length'],
        training_config['device']
    )
    
    # Forward pass
    logits = model(X)
    
    # Compute loss
    loss = cross_entropy(logits, y)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    gradient_clipping(model.parameters(), max_l2_norm=opt_config['max_grad_norm'])
    
    # Update parameters
    optimizer.step()
    
    return loss.item()


# =============================================================================
# Validation Step
# =============================================================================
def validate(
    model: TransformerLM,
    val_data: np.ndarray,
    config: Dict[str, Any]
) -> float:
    """Evaluate model on validation set.
    
    Args:
        model: Transformer model
        val_data: Validation data array
        config: Configuration dictionary
        
    Returns:
        Validation loss value
    """
    training_config = config['training']
    
    model.eval()
    with torch.no_grad():
        val_X, val_y = get_batch(
            val_data,
            training_config['val_batch_size'],
            training_config['context_length'],
            training_config['device']
        )
        val_logits = model(val_X)
        val_loss = cross_entropy(val_logits, val_y)
    
    model.train()
    return val_loss.item()


# =============================================================================
# Plotting
# =============================================================================
def plot_training_metrics(
    training_losses: List[float],
    val_losses: List[float],
    lrs: List[float],
    config: Dict[str, Any]
) -> None:
    """Plot training metrics.
    
    Args:
        training_losses: List of training loss values
        val_losses: List of validation loss values
        lrs: List of learning rate values
        config: Configuration dictionary
    """
    if not config['plotting']['enabled']:
        return
    
    plot_config = config['plotting']
    sns.set_theme(style=plot_config['style'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=tuple(plot_config['figsize']))
    
    # Training loss
    ax1.plot(training_losses, label="Training Loss")
    ax1.set_title("Training Loss Over Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    # Validation loss
    ax2.plot(val_losses, label="Validation Loss", color="green")
    ax2.set_title("Validation Loss Over Iterations")
    ax2.set_xlabel(f"Checkpoint (every {config['training']['checkpoint_interval']} iterations)")
    ax2.set_ylabel("Loss")
    ax2.legend()
    
    # Learning rate
    ax3.plot(lrs, label="Learning Rate", color="orange")
    ax3.set_title("Learning Rate Schedule")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Learning Rate")
    ax3.legend()
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main Training Loop
# =============================================================================
def train(config: Dict[str, Any]) -> None:
    """Main training loop.
    
    Args:
        config: Configuration dictionary containing all hyperparameters
    """
    # Setup
    train_data, val_data = setup_data(config)
    model = setup_model(config)
    optimizer = setup_optimizer(model, config)
    model = torch.compile(model, backend="aot_eager")
    
    # Extract configuration values
    training_config = config['training']
    opt_config = config['optimizer']
    schedule_config = config['schedule']
    
    max_iteration = training_config['max_iteration']
    warmup_iters = max_iteration * schedule_config['warmup_ratio']
    cosine_cycle_iters = max_iteration * schedule_config['cosine_ratio']
    
    # Tracking metrics
    training_losses = []
    val_losses = []
    lrs = []
    
    print(f"\nStarting training for {max_iteration} iterations...")
    print(f"Warmup: {warmup_iters:.0f} iters, Cosine annealing: {cosine_cycle_iters:.0f} iters\n")
    
    # Training loop
    itr = 0
    while itr < max_iteration:
        # Compute learning rate
        lr = learning_rate_schedule(
            itr,
            opt_config['max_lr'],
            opt_config['min_lr'],
            warmup_iters,
            cosine_cycle_iters
        )
        
        # Training step
        loss = train_step(model, optimizer, train_data, config, lr)
        
        # Track metrics
        training_losses.append(loss)
        lrs.append(lr)
        
        # Logging
        if (itr + 1) % training_config['val_interval'] == 0:
            val_loss = validate(model, val_data, config)
            val_losses.append(val_loss)
            print(f"Iteration {itr + 1}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}, LR = {lr:.6f}")
        
        # Checkpointing
        if (itr + 1) % training_config['checkpoint_interval'] == 0:
            checkpoint_dir = Path(training_config['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            output_path = checkpoint_dir / f"checkpoint_itr_{itr + 1}.pth"
            save_checkpoint(model, optimizer, itr + 1, str(output_path))
            print(f"  Checkpoint saved to {output_path}")
        
        itr += 1
    
    print(f"\nTraining complete! Final loss: {training_losses[-1]:.4f}")
    
    # Plot results
    plot_training_metrics(training_losses, val_losses, lrs, config)


# =============================================================================
# CLI Entry Point
# =============================================================================
def main():
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/config_llm_intro.yaml",
        help="Path to YAML configuration file (default: src/configs/config_llm_intro.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}\n")
    
    # Start training
    train(config)


if __name__ == "__main__":
    main()
