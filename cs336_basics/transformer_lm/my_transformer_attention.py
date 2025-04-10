#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys
import numpy as np
from typing import Optional, List
import torch.nn.functional as F

torch.set_printoptions(threshold=float(500))
np.set_printoptions(threshold=500)
sys.path.insert(0, r'C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\2023-2024\Spring\CS336 -  LLMs scratch\Assignments\Assignment 1\repo-feb25\tests')

from common import FIXTURES_PATH

from cs336_basics.transformer_lm.my_transformer_block_elements import softmax, RMSLayerNorm


class causalMultiHeadSelfAttention(nn.Module): 
    def __init__(self,
                 d_model:int, 
                 num_heads: int, 
                 attn_pdrop: float, 
                 weights: dict[str, torch.FloatTensor],
                 in_features: torch.FloatTensor,
                 current_layer:int):
        
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.attn_pdrop=attn_pdrop
        self.weights=weights
        self.in_features=in_features
        self.seq_length=in_features.shape[-2]
        self.mask=torch.triu(torch.ones(self.seq_length, self.seq_length), diagonal=1).bool()
        self.current_layer=current_layer

        self.W_Q=weights[f'layers.{self.current_layer}.attn.q_proj.weight']
        self.W_K=weights[f'layers.{self.current_layer}.attn.k_proj.weight']
        self.W_V=weights[f'layers.{self.current_layer}.attn.v_proj.weight']
        self.W_O=weights[f'layers.{self.current_layer}.attn.output_proj.weight']

    def multi_head_self_attention(self):
        Q = torch.matmul(self.in_features, self.W_Q.transpose(-1,-2))
        K = torch.matmul(self.in_features, self.W_K.transpose(-1,-2))
        V = torch.matmul(self.in_features, self.W_V.transpose(-1,-2))

        Q=self.split_heads(Q, self.num_heads)
        K=self.split_heads(K, self.num_heads)
        V=self.split_heads(V, self.num_heads)

        attn_output = scaled_dot_product_attention(K, Q, V, self.mask, self.attn_pdrop)
 
        attn_output = self.combine_heads(attn_output, self.num_heads)

        output = torch.matmul(attn_output, self.W_O.t())
        
        return output

    def split_heads(self, x, num_heads):
        batch_size, seq_length, d_model = x.size()
        depth = d_model // num_heads
        x = x.view(batch_size, seq_length, num_heads, depth)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x, num_heads):
        batch_size, num_heads, seq_length, depth = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_length, num_heads * depth)
    

def scaled_dot_product_attention(
        K: torch.FloatTensor,
        Q: torch.FloatTensor,
        V: torch.FloatTensor,
        mask: Optional[torch.BoolTensor] = None,
        pdrop: Optional[float] = None,
    ) -> torch.FloatTensor:
    """Given key (K), query (Q), and value (V) tensors, return the output of your scaled dot product attention implementation.

    Args:
        K: torch.FloatTensor
            Tensor with attention keys. Shape is (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other batch dimensions (e.g., num_heads).
        Q: torch.FloatTensor
            Tensor with attention queries. Shape is (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other batch dimensions (e.g., num_heads).
        V: torch.FloatTensor
            Tensor with attention values. Shape is (batch_size, ..., seq_len, value_dimension), where
            "..." is optional and represents any number of other batch dimensions (e.g., num_heads).
        mask: Optional[torch.BoolTensor]
            An (optional) mask of shape (seq_len, seq_len).
            Attention scores for positions with a mask value of `True` should
            be masked out, i.e., not affect the softmaxed attention probabilities.
        pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.

    Returns:
        torch.FloatTensor of shape (batch_size, ..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """
    # Calculate the dot product between Q and K transpose
    scores=torch.matmul(Q, K.transpose(-2,-1))
    # Scale the scores by the square root of the key dimension
    d_k= K.size(-1)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Apply softmax to get the attention probabilities
    attention_probs = softmax(scores, dim=-1)

    # Apply dropout to attention probabilities (if pdrop is provided)
    if pdrop is not None:
        attention_probs = F.dropout(attention_probs, p=pdrop)

    # Multiply the attention probabilities with the value tensor to get the output
    output = torch.matmul(attention_probs, V)

    return output
