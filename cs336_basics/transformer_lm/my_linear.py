from einops import rearrange, einsum, reduce
from jaxtyping import Float, Int
import torch
from torch import Tensor



class Linear(torch.nn.Module):
    def __init__(self,
                  d_in: int,
                  d_out: int,
                  weights: Float[Tensor, " d_out d_in"],
                  in_features:Float[Tensor, " ... d_in"],
                  device: torch.device|None=None,
                  dtype: torch.dtype|None=None):
        super().__init__()

        self.device=device
        self.dtype=dtype
        # weights=torch.tensor(weights, dtype=dtype,device=device)
        weights=weights.to(dtype=dtype, device=device)
        self.weights_t=torch.nn.Parameter(rearrange(weights, 
                                                    "d_out d_in-> d_in d_out"))
    
        # variance=2/(d_in+d_out)
        # cutoff=3*np.sqrt(variance)
        
        # self.weights_t=torch.nn.Parameter(torch.nn.init.trunc_normal_(self.weights_t, 
        #                                                        mean=0,
        #                                                        std=np.sqrt(variance),
        #                                                        a=-cutoff,
        #                                                        b=cutoff))

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        # x = rearrange(x, "batch seq d_in -> batch d_in seq")
        # breakpoint()
        # Y=einsum(x, self.weights_t, "batch sequence d_in, d_out d_in -> batch sequence d_out" )
        Y=einsum(x, self.weights_t, "... d_in, d_in d_out -> ... d_out" )

        result = x@self.weights_t
        # breakpoint()
        return Y
    

