#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys
import numpy as np
from typing import Optional, List
import torch.nn.functional as F

from .my_transformer_block_elements import positionwise_feedforward, RMSLayerNorm
from .my_transformer_attention import causalMultiHeadSelfAttention


def transformer_block( d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        residual_pdrop: float,
        weights: dict[str, torch.FloatTensor],
        in_features: torch.FloatTensor, 
        current_layer: int) -> torch.FloatTensor:
    
    #multihead self attention layer
    x = RMSLayerNorm(d_model=d_model, 
                     eps=1e-5, 
                     weights=weights, 
                     in_features=in_features, 
                     weight_name=f'layers.{current_layer}.ln1.weight').forward()
    
    x= causalMultiHeadSelfAttention(d_model=d_model, 
                                    num_heads=num_heads,
                                    attn_pdrop=attn_pdrop, 
                                    weights=weights,
                                    in_features=x,
                                    current_layer=current_layer).multi_head_self_attention()
    
    x= F.dropout(x,p=attn_pdrop)

    x+=in_features
    x_attention_layer=x

    #positionwise feedforward    
    x = RMSLayerNorm(d_model=d_model,
                     eps=1e-5,
                     weights=weights,
                     in_features=x,
                     weight_name=f'layers.{current_layer}.ln2.weight').forward()
    
    x=positionwise_feedforward(d_model=d_model,
                               d_ff=d_ff,
                               weights=weights,
                               in_features=x, 
                               current_layer=current_layer)

    x=F.dropout(x,p=residual_pdrop)

    x+=x_attention_layer

    return x