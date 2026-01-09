"""
Optimizer module implementing AdamW and related training utilities.

This module provides:
- AdamW: An implementation of the Adam optimizer with weight decay decoupling
- learning_rate_schedule: A learning rate scheduler with warmup and cosine annealing
- gradient_clipping: A utility function for gradient norm clipping
"""

import math
import torch


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with decoupled weight decay.
    
    Implements the AdamW algorithm as described in "Decoupled Weight Decay Regularization".
    This variant applies weight decay separately from the gradient updates, improving
    generalization compared to standard Adam with L2 regularization.
    """
    def __init__(self, params, lr, betas, eps, weight_decay):
        """Initialize the AdamW optimizer.
        
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate.
            betas (tuple): Coefficients used for computing running averages of gradient
                          and its square. Typically (0.9, 0.999).
            eps (float): Term added to denominator for numerical stability. Typically 1e-8.
            weight_decay (float): Weight decay coefficient. Applied as decoupled regularization.
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        """Perform a single optimization step.
        
        Updates parameters using the AdamW algorithm:
        1. Update biased first moment estimate (momentum)
        2. Update biased second moment estimate (velocity)
        3. Apply bias correction to learning rate
        4. Update parameters using adaptive learning rate
        5. Apply decoupled weight decay
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data

                # Initialize state on first step
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)  # First moment (mean)
                    state["v"] = torch.zeros_like(p.data)  # Second moment (variance)
                
                beta1, beta2 = group["betas"]
                m, v = state["m"], state["v"]
                
                # Update biased first moment estimate (momentum)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second moment estimate
                v.mul_(beta2).add_(grad**2, alpha=1 - beta2)

                lr = group["lr"]
                state["t"] += 1
                
                # Compute bias-corrected learning rate
                adjusted_lr = lr * (math.sqrt(1 - beta2 ** state["t"])) / (1 - beta1 ** state["t"])
                
                # Update parameters with adaptive learning rate
                p.data.sub_(adjusted_lr * m / (torch.sqrt(v) + group["eps"]))
                
                # Apply decoupled weight decay
                p.data.mul_(1 - lr * group["weight_decay"])

def learning_rate_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    """Compute learning rate using warmup + cosine annealing schedule.
    
    The schedule consists of two phases:
    1. Warmup: Linear increase from 0 to max_learning_rate
    2. Cosine annealing: Cosine decay from max to min_learning_rate
    
    Args:
        it (int): Current iteration number.
        max_learning_rate (float): Maximum learning rate (at end of warmup).
        min_learning_rate (float): Minimum learning rate (at end of cosine cycle).
        warmup_iters (int): Number of iterations for the warmup phase.
        cosine_cycle_iters (int): Number of iterations for the cosine annealing phase.
    
    Returns:
        float: The learning rate for the current iteration.
    """
    # Warmup phase: linearly increase learning rate
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    # Cosine annealing phase: smoothly decay learning rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(((it - warmup_iters) * math.pi) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    # After cycle: maintain minimum learning rate
    else:
        return min_learning_rate

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    """Clip gradients by global L2 norm.
    
    Rescales all gradients if the total L2 norm exceeds max_l2_norm.
    This prevents exploding gradients during training.
    
    Args:
        parameters: Iterable of parameters with gradients to clip.
        max_l2_norm (float): Maximum allowed L2 norm of gradients.
        eps (float): Small epsilon for numerical stability. Default: 1e-6.
    
    Returns:
        None
    """
    # Compute L2 norm of each parameter's gradient
    l2s = [param.grad.square().sum().sqrt() for param in parameters if param.grad is not None]
    
    # Compute total L2 norm across all parameters
    total_norm = torch.sqrt(sum(l2**2 for l2 in l2s))
    
    # Scale gradients if total norm exceeds threshold
    [
        params.grad.mul_((max_l2_norm/(total_norm + eps)))
        for params in parameters
        if params.grad is not None and total_norm >= max_l2_norm
    ]
    return None