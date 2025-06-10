from einops import rearrange, einsum
from jaxtyping import Float
import torch 
from torch import Tensor
import torch.nn as nn


def silu(in_features: Float[Tensor, "..." ])-> Float[Tensor,"..."]:
        
    return in_features * torch.sigmoid(in_features)


def swiglu(d_model:int,
           d_ff:int, 
           w1_weight: Float[Tensor, " d_ff d_model"],
           w2_weight: Float[Tensor, " d_model d_ff"],
           w3_weight: Float[Tensor, " d_ff d_model"],
           in_features: Float[Tensor, " ... d_model"],
        ) -> Float[Tensor, " ... d_model"]:
        """Run the SwiGLU feedforward network using the given weights."""


        # d_model_, d_ff_ = find_factors_with_multiple_of_64(d_model*d_ff)

        w1_weight = rearrange(w1_weight, "d_ff d_model -> d_ff d_model", d_model=d_model)
        w2_weight = rearrange(w2_weight, "d_model d_ff -> d_model d_ff", d_model=d_model)
        w3_weight = rearrange(w3_weight, "d_ff d_model -> d_ff d_model", d_model=d_model)

        # Project input using W1 and W3: shape (..., d_model) -> (..., d_ff)
        x1=einsum(in_features, w1_weight, "... d_model, d_ff d_model -> ... d_ff" )
        x3=einsum(in_features, w3_weight, "... d_model, d_ff d_model -> ... d_ff" )

        # SwiGLU activation: Swish(x1) * x2
        swiglu_output = silu(x1) * x3  # (..., d_ff)

        # Down-project back to d_model using W2
        output=einsum(swiglu_output, w2_weight, "... d_ff, d_model d_ff -> ... d_model" )

        return output


class SwiGLU(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 w1_weight: Float[Tensor, "d_ff d_model"],
                 w2_weight: Float[Tensor, "d_model d_ff"],
                 w3_weight: Float[Tensor, "d_ff d_model"]):
        """
        SwiGLU feedforward layer implemented as a class module.

        Args:
            d_model: Input/output dimensionality.
            d_ff: Intermediate feedforward dimensionality.
            w1_weight, w2_weight, w3_weight: Pre-initialized weight tensors.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Register weights as parameters (or buffers if you don't want them trainable)
        self.w1 = nn.Parameter(rearrange(w1_weight, "d_ff d_model -> d_ff d_model"))
        self.w2 = nn.Parameter(rearrange(w2_weight, "d_model d_ff -> d_model d_ff"))
        self.w3 = nn.Parameter(rearrange(w3_weight, "d_ff d_model -> d_ff d_model"))


    def forward(self, in_features: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        Run the SwiGLU feedforward computation.

        Args:
            in_features: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)
        """
        x1 = einsum(in_features, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        x3 = einsum(in_features, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        swiglu_output = silu(x1) * x3
        output = einsum(swiglu_output, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        return output
