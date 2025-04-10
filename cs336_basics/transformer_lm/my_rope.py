import torch

class RotaryPositionalEmbedding():
    def __init__(self, 
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device: torch.device|None=None):
        pass

    def forward(self, 
                x:torch.Tensor,
                token_positions: torch.Tensor) -> torch.Tensor:
        pass 