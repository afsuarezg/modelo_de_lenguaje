from einops import rearrange, einsum
from jaxtyping import Float
import torch 
from torch import Tensor


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
        breakpoint()
        def find_factors_with_multiple_of_64(result):
            """
            Finds and returns a tuple (a, b) such that:
            - a is a multiple of 64
            - b = result / a
            - a * b == result
            Returns None if no such a is found.
            """
            # Start from 64 and check up to the result
            for a in range(64, result + 1, 64):
                if result % a == 0:
                    b = result // a
                    return a, b
            return None  # No such pair found
        
        d_model_, d_ff_ = find_factors_with_multiple_of_64(d_model*d_ff)

        w1_weight = rearrange(w1_weight, "d_ff d_model -> d_ff d_model", d_model=d_model_)
        w2_weight = rearrange(w2_weight, "d_model d_ff -> d_model d_ff", d_model=d_model_)
        w3_weight = rearrange(w3_weight, "d_ff d_model -> d_ff d_model", d_model=d_model_)

        # Project input using W1 and W3: shape (..., d_model) -> (..., d_ff)
        x1=einsum(in_features, w1_weight, "... d_model, d_ff d_model -> ... d_ff" )
        x3=einsum(in_features, w3_weight, "... d_model, d_ff d_model -> ... d_ff" )

        # SwiGLU activation: Swish(x1) * x2
        swiglu_output = silu(x1) * x3  # (..., d_ff)

        # Down-project back to d_model using W2
        output=einsum(swiglu_output, w2_weight, "... d_ff, d_model d_ff -> ... d_model" )

        return output
