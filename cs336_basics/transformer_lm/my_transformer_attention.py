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

# from common import FIXTURES_PATH
from cs336_basics.transformer_lm.my_transformer_block_elements import softmax, RMSLayerNorm
from cs336_basics.transformer_lm.my_rope import myRotaryPositionalEmbedding
# from cs336_basics.transformer_lm.rope_previous import RotaryPositionalEmbedding
from cs336_basics.transformer_lm.my_linear import Linear


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
    # queries, keys and values should be all equal to seq_len.
    # assert Q.shape[-2] == K.shape[-2], "La última dimensión de Q y K no es igual." 
    assert K.shape[-2] == V.shape[-2], "La última dimensión de K y V no son iguales." 
    d_k=Q.shape[-1]
    queries_size=Q.shape[-2]

    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    #scaled scores
    scores=scores/np.sqrt(d_k)
    # breakpoint()
    if mask == None:
        mask = 1 - torch.ones_like(input=scores, dtype=torch.int).triu(diagonal=1)
    # breakpoint()
    scores_masked = scores.masked_fill(mask==0, float('-inf'))

    scores_softmaxed = softmax(in_features=scores_masked)
    # breakpoint()

    assert K.shape[-2] == V.shape[-2], "Seq_len de K no es igual a seq_len de V"
    result = einsum(scores_softmaxed, V, "... queries seq_len, ... seq_len d_v -> ... queries d_v")

    assert result.shape[-2]==queries_size, "La penúltima dimensión del resultado de attention no es igual a la penúltima dimensión de Q."

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
            q_proj_weight: Float[Tensor, " d_k d_in"],
            k_proj_weight: Float[Tensor, " d_k d_in"],
            v_proj_weight: Float[Tensor, " d_v d_in"],
            o_proj_weight: Float[Tensor, " d_model d_v"],
            #attributes in case of using rope
            rope:bool=False,
            max_seq_len:int=None,
            token_positions:  Int[Tensor, " ... sequence_length"] | None = None,
            theta: float=1000.0):
        
        super().__init__()
        # breakpoint()
        #d_model=d_in=lenght of each word embedding
        assert q_proj_weight.shape==k_proj_weight.shape==v_proj_weight.shape
        assert q_proj_weight.shape[-1] == d_model, "La última dimensión de wq, wk, y wv no es igual a d_model ( donde d_model es el tamano del vector de embeddings para cada token)."
        assert q_proj_weight.shape[-2]%num_heads == 0

        self.q_proj_weight=nn.Parameter(q_proj_weight)
        self.k_proj_weight=nn.Parameter(k_proj_weight)
        self.v_proj_weight=nn.Parameter(v_proj_weight)
        self.o_proj_weight=nn.Parameter(o_proj_weight)
        self.d_model=d_model
        self.num_heads=num_heads
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.d_k = int(d_model/num_heads)
        self.rope=rope
        if self.rope:
            self.theta=theta
            self.max_seq_len=max_seq_len
            self.rope_class = myRotaryPositionalEmbedding(theta=theta, 
                                                          d_k=self.d_k, 
                                                          max_seq_len=max_seq_len)
            self.token_positions=token_positions.to(self.device)
            


    def forward(self, x:Float[Tensor, " ... sequence_length d_in"])-> Float[Tensor, " ... sequence_length d_out"]:
        
        xq=einsum(x, self.q_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k" )
        xk=einsum(x, self.k_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k" )
        xv=einsum(x, self.v_proj_weight, "... sequence_length d_in, d_v d_in -> ... sequence_length d_v" )
        # heads*d_k_h=d_k
        xq=rearrange(xq, "... sequence_length (heads d_k_h) -> ... heads sequence_length d_k_h", heads=self.num_heads)
        xk=rearrange(xk, "... sequence_length (heads d_k_h) -> ... heads sequence_length d_k_h", heads=self.num_heads)
        xv=rearrange(xv, "... sequence_length (heads d_v_h) -> ... heads sequence_length d_v_h", heads=self.num_heads)

        if self.rope: 
            
            xq=self.rope_class.forward(x=xq, token_positions=self.token_positions)
            xk=self.rope_class.forward(x=xk, token_positions=self.token_positions)

        multihead=scaled_dot_product_attention(Q=xq, K=xk, V=xv)
        multihead=rearrange(multihead, "... heads seq_len d_v_h -> ... seq_len (heads d_v_h)") 
        multiheadW0=einsum(self.o_proj_weight, multihead,  "d_model d_v,... seq_len d_v -> ... seq_len d_model ")

        return multiheadW0


