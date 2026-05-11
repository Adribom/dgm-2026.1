from typing import Optional

import torch
from torch import nn, Tensor


class FrequencyProjector(nn.Module):
    """MLP that maps frequency extractor features to LLM embedding tokens.

    Architecture mirrors the CLIP projector in LLaVA: two linear layers
    with GELU activation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 4096,
        num_tokens: int = 1,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden_dim, output_dim * num_tokens)

    def forward(self, x: Tensor) -> Tensor:
        """Map frequency features to LLM token embeddings.

        Args:
            x: [B, input_dim]

        Returns:
            [B, num_tokens, output_dim]
        """
        x = self.act(self.linear_1(x))
        x = self.linear_2(x)
        return x.view(x.shape[0], self.num_tokens, self.output_dim)
