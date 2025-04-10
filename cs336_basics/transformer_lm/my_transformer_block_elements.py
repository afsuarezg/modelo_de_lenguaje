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
             dim: int) -> torch.FloatTensor:
    max_values , _= torch.max(in_features,dim=dim, keepdim=True)
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
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model: int
            The dimensionality of the RMSNorm input.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `weight`
                Weights of the RMSNorm affine transform.
                Shape is (d_model,).
        in_features: torch.FloatTensor
            Input features to run RMSNorm on. Tensor of (*, d_model), where *
            can be an arbitrary number of dimensions with arbitrary values.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
  
    def __init__(self,
                d_model: int,
                eps: float, 
                weights: Float[Tensor, " d_model"],
                in_features: torch.FloatTensor,
                weight_name:str,
                device: torch.device|None=None,
                dtype: torch.dtype| None=None) :
        
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = weights
        self.in_features = in_features
        self.weight_name = weight_name

    def forward(self)-> torch.FloatTensor:
        denominator = self.in_features.square()
        denominator =torch.mean(denominator, dim=-1, keepdim=True)
        denominator=denominator+self.eps    
        denominator=denominator.sqrt()                           
        return self.in_features.divide(denominator)*self.weights[self.weight_name]
    

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
        
    
