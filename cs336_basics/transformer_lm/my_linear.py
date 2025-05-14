from einops import rearrange, einsum, reduce
from jaxtyping import Float, Int
import numpy as np
import torch
from torch import Tensor



class Linear(torch.nn.Module):
    def __init__(self,
                  d_in: int,
                  d_out: int,
                  weights: Float[Tensor, " d_out d_in"],
                  device: torch.device|None=None,
                  dtype: torch.dtype|None=None):
        
        super().__init__()
        assert weights.shape == (d_out, d_in)
        
        self.device=device
        self.dtype=dtype
        # weights=torch.tensor(weights, dtype=dtype,device=device)
        weights=weights.to(dtype=dtype, device=device)
        self.weights_t=torch.nn.Parameter(rearrange(weights, "... d_out d_in-> ... d_in d_out"))

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        # x = rearrange(x, "batch seq d_in -> batch d_in seq")
        # breakpoint()
        # Y=einsum(x, self.weights_t, "batch sequence d_in, d_out d_in -> batch sequence d_out" )
        Y=einsum(x, self.weights_t, "... d_in,  ... d_in d_out -> ... d_out" )
        # breakpoint()
        return Y
    

