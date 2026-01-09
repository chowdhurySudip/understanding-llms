"""Loss functions for neural network training.

Implements numerically stable cross-entropy loss for classification tasks.
"""

import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute numerically stable cross-entropy loss.
    
    Implements the cross-entropy loss function with numerical stability by
    using the log-sum-exp trick (subtracting max before exponentiation).
    
    The loss is computed as:
    L = mean(log(sum(exp(logits))) - logits[target])
      = mean(max + log(sum(exp(logits - max))) - logits[target])
    
    Args:
        logits: Unnormalized log probabilities of shape (..., num_classes)
        targets: Target class indices of shape (...) or (..., 1)
                Values should be in range [0, num_classes-1]
    
    Returns:
        Scalar tensor containing the mean cross-entropy loss across all samples
    
    Example:
        >>> logits = torch.randn(32, 10)  # batch_size=32, num_classes=10
        >>> targets = torch.randint(0, 10, (32,))
        >>> loss = cross_entropy(logits, targets)
    """
    # Extract batch dimensions (everything except the last dimension)
    *batch, _ = logits.shape
    
    # Ensure targets have shape (..., 1) for gather operation
    targets = targets.view(*batch, 1)
    
    # Compute max logits for numerical stability (prevents overflow in exp)
    max_logits = logits.max(dim=-1, keepdim=True).values
    
    # Compute cross-entropy with log-sum-exp trick:
    # log(sum(exp(logits))) = max + log(sum(exp(logits - max)))
    loss = (
        max_logits  # First term
        + (logits - max_logits).exp().sum(-1, keepdim=True).log()  # log-sum-exp
        - logits.gather(-1, targets)  # Subtract the target class logit
    ).mean()  # Average over all samples
    
    return loss
    