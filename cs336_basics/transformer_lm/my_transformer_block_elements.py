#!/usr/bin/env python3
from einops import rearrange, einsum, reduce
from jaxtyping import Float, Int
import numpy as np
import sys
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, List



def softmax(in_features: torch.FloatTensor,
             dim: int=-1) -> torch.FloatTensor:
    
    max_values , _= in_features.max(dim=dim, keepdim=True)
    in_features = in_features.sub(max_values)
    in_features = in_features.exp()
    denominator = in_features.sum(dim=-1, keepdim=True)
   
    return in_features.divide(denominator)


def gelu(in_features: torch.FloatTensor) -> torch.FloatTensor:
    erf = torch.special.erf(in_features.divide(torch.sqrt(torch.tensor([2]))))
    return in_features.divide(2).multiply(1+erf)


def positionwise_feedforward(        
        d_model:int, 
        d_ff: int,
        weights: dict[str, torch.FloatTensor],
        in_features:torch.FloatTensor,
        current_layer=int)-> torch.FloatTensor:
    
    W1=weights[f'layers.{current_layer}.ffn.w1.weight'].transpose(0,1)
    W2=weights[f'layers.{current_layer}.ffn.w2.weight'].transpose(0,1)
      
    assert W1.shape==(d_model, d_ff)
    assert W2.shape==(d_ff, d_model)

    # First linear transformation: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
    hidden = gelu(torch.matmul(in_features, W1))

    # Second linear transformation: (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
    output = torch.matmul(hidden, W2)

    return output

class RMSLayerNorm(nn.Module):

    def __init__(self, 
                 d_model: int, 
                 eps:float,
                 weights:Float[Tensor, " d_model"], 
                 device: torch.device | None = None,
                 dtype: torch.dtype | None=None):
        
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        weights=weights.to(dtype=dtype, device=device)
        self.weights= nn.Parameter(weights)        


    def forward(self,
                x:torch.Tensor)-> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """    
        in_dtype=x.dtype
        x=x.to(torch.float32)

        x_ = x.pow(exponent=2)
        x_ = reduce(x_, "batch_size sequence_length d_model -> batch_size sequence_length", "sum")
        x_ = (x_+self.eps)/self.d_model
        x_ = x_.sqrt()
        x_ = rearrange(x_, "batch_size sequence_length -> batch_size sequence_length 1")
        x = torch.div(x, x_)
        weights = rearrange(self.weights, "d_model -> 1 1 d_model")
        x = x*weights

        return x.to(in_dtype)
        
    
