#!/usr/bin/env python3
from einops import einsum, rearrange
from jaxtyping import Float, Int
import math
import torch
import torch.nn as nn
from torch import Tensor
import sys
import numpy as np
from typing import Optional, List
import torch.nn.functional as F

torch.set_printoptions(threshold=float(500))
np.set_printoptions(threshold=500)
sys.path.insert(0, r'C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\2023-2024\Spring\CS336 -  LLMs scratch\Assignments\Assignment 1\repo-feb25\tests')

from common import FIXTURES_PATH

from cs336_basics.transformer_lm.my_transformer_block_elements import softmax, RMSLayerNorm
from cs336_basics.transformer_lm.my_rope import RotaryPositionalEmbedding


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
    

def scaled_dot_product_attention_deprecated(
        K: torch.FloatTensor,
        Q: torch.FloatTensor,
        V: torch.FloatTensor,
        mask: Optional[torch.BoolTensor] = None,
        pdrop: Optional[float] = None,
    ) -> torch.FloatTensor:
    """    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
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



def scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],    
        mask: Float[Tensor, " ... queries keys"] | None = None) -> Float[Tensor, " ... queries d_v"]:
    """    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # breakpoint()
    d_k=Q.shape[-1]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")/np.sqrt(d_k)
    # breakpoint()

    if mask == None:
    # mask = torch.tensor(np.tril(mask))
        mask = torch.ones_like(input=scores, dtype=torch.int).triu(diagonal=1)
        # mask = torch.triu(input=scores, diagonal=1)
    # breakpoint()
    # scores_masked = scores.masked_fill(mask==1, float('-inf'))
    scores_masked = scores.masked_fill(mask!=0, float('-inf'))
    # scores_masked = scores.masked_fill(mask==0, float('-inf')) # con este funciona para la función con una sola cabeza

    scores_softmaxed = softmax(in_features=scores_masked, dim=-1)
    # breakpoint()
    # result = einsum(scores_softmaxed, V, "... q k, ... k d -> ... q d")
    result = einsum(scores_softmaxed, V, "... queries keys, ... keys d_v -> ... queries d_v")

    return result 


class causalMultiHeadSelfAttention(nn.Module): 
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.   
    """

    def __init__(self,
                 d_model:int, 
                 num_heads: int, 
                 q_proj_weight: Float[Tensor, " d_model d_model"],
                 k_proj_weight: Float[Tensor, " d_model d_model"],
                 v_proj_weight: Float[Tensor, " d_model d_model"],
                 o_proj_weight: Float[Tensor, " d_model d_o"],
                 in_features: Float[Tensor, " ... sequence_length d_model"],
                 rope:bool=False,
                 max_seq_len:int=None,
                 token_positions:  Int[Tensor, " ... sequence_length"] | None = None,
                 theta: float=1000.0, 
                 ):
        super().__init__()
        # breakpoint()
        self.in_features=in_features
        self.q_proj_weight=rearrange(q_proj_weight, "(d_k heads) d_model -> heads d_model d_k", heads=num_heads) 
        self.k_proj_weight=rearrange(k_proj_weight, "(d_k heads) d_model -> heads d_model d_k", heads=num_heads) 
        self.v_proj_weight=rearrange(v_proj_weight, "(d_v heads) d_model -> heads d_model d_v", heads=num_heads) 
        # breakpoint()
        self.o_proj_weight=o_proj_weight
        self.d_model=d_model
        self.num_heads=num_heads
        
        self.d_k = d_model/num_heads
        
        self.rope=rope
        if self.rope:
            self.theta=theta
            self.max_seq_len=max_seq_len
            self.rope_class = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
            self.token_positions=token_positions



    def multi_head_self_attention(self)-> Float[Tensor, " ... sequence_length d_out"]:
        # breakpoint()
        xq=einsum(self.in_features, self.q_proj_weight, "... sequence_length d_model, heads d_model d_k -> ... heads sequence_length d_k" )
        xk=einsum(self.in_features, self.k_proj_weight, "... sequence_length d_model, heads d_model d_k -> ... heads sequence_length d_k" )
        xv=einsum(self.in_features, self.v_proj_weight, "... sequence_length d_model, heads d_model d_v -> ... heads sequence_length d_v" )
        # breakpoint()        
        # breakpoint()
        if self.rope: 
            xq=self.rope_class.forward(x=xq, token_positions=self.token_positions)
            xk=self.rope_class.forward(x=xk, token_positions=self.token_positions)

        multihead=scaled_dot_product_attention(Q=xq, K=xk, V=xv)
        breakpoint()
        multihead=rearrange(multihead, "... heads seq_len d_emb -> ... seq_len (heads d_emb)") 
        w0multihead=einsum(self.o_proj_weight, multihead, " d_k d_out, ... seq_len d_k -> ... seq_len d_out ")
        # breakpoint()
        return w0multihead


def multihead_self_attention_chatgpt(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... seq_len d_in"],
) -> Float[Tensor, " ... seq_len d_model"]:
    """
    Implements batched multi-head self-attention.
    """
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    *batch_dims, seq_len, d_in = in_features.shape
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]
    d_head = d_k // num_heads
    d_out = d_model

    # Project input to Q, K, V using matrix multiplication
    Q = torch.matmul(in_features, q_proj_weight.T)  # [..., seq_len, d_k]
    K = torch.matmul(in_features, k_proj_weight.T)  # [..., seq_len, d_k]
    V = torch.matmul(in_features, v_proj_weight.T)  # [..., seq_len, d_v]

    # Reshape Q, K, V for multi-head attention
    def split_heads(x, d_split):
        # Input shape: [..., seq_len, d_split]
        # Output shape: [..., num_heads, seq_len, d_head]
        return x.view(*batch_dims, seq_len, num_heads, d_split // num_heads).transpose(-3, -2)

    Q = split_heads(Q, d_k)
    K = split_heads(K, d_k)
    V = split_heads(V, d_v)

    # Scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (d_head ** 0.5)  # [..., num_heads, seq_len, seq_len]
    attn_weights = softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, V)  # [..., num_heads, seq_len, d_head]

    # Combine heads
    def combine_heads(x):
        # Input shape: [..., num_heads, seq_len, d_head]
        # Output shape: [..., seq_len, d_k or d_v]
        x = x.transpose(-3, -2)  # [..., seq_len, num_heads, d_head]
        return x.reshape(*batch_dims, seq_len, -1)  # combine heads

    combined = combine_heads(attn_output)  # [..., seq_len, d_v]

    # Final linear projection
    output = torch.matmul(combined, o_proj_weight.T)  # [..., seq_len, d_model]

    return output