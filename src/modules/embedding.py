import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.emb = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    (num_embeddings, embedding_dim), dtype=dtype, device=device
                ), a=-3, b=3
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids]