import torch
import numpy as np
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super(Linear, self).__init__()
        std = np.sqrt(2/(in_features+out_features))
        self.W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    (out_features, in_features), dtype=dtype, device=device
                ), std=std, a=-3*std, b=3*std
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T