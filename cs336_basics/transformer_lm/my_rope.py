import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Compute inverse frequencies
        # freq: [d_k//2]
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))

        # positions: [max_seq_len]
        positions = torch.arange(max_seq_len).float().unsqueeze(1)  # [seq_len, 1]

        # angles: [seq_len, d_k//2]
        angles = positions * inv_freq.unsqueeze(0)

        # Store cos and sin buffers of shape [seq_len, d_k]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        cos = torch.stack([cos, cos], dim=-1).reshape(max_seq_len, d_k)
        sin = torch.stack([sin, sin], dim=-1).reshape(max_seq_len, d_k)

        self.register_buffer("cos_cached", cos.to(device), persistent=False)
        self.register_buffer("sin_cached", sin.to(device), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., seq_len, d_k] input tensor
            token_positions: [..., seq_len] indices of positions in sequence
        Returns:
            Rotated x using RoPE
        """
        cos = self.cos_cached[token_positions]  # [..., seq_len, d_k]
        sin = self.sin_cached[token_positions]  # [..., seq_len, d_k]

        # Split into pairs: (x_even, x_odd)
        x1, x2 = x[..., ::2], x[..., 1::2]  # both shape [..., seq_len, d_k//2]

        # Apply 2D rotation: (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
        x_rotated = torch.stack([x1 * cos[..., ::2] - x2 * sin[..., ::2],
                                 x1 * sin[..., ::2] + x2 * cos[..., ::2]],
                                dim=-1)

        return x_rotated.flatten(-2)  # merge last dim back to d_k