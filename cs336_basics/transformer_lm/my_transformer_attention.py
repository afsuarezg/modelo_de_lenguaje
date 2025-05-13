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

        self.q_proj_weight=q_proj_weight
        self.k_proj_weight=k_proj_weight
        self.v_proj_weight=v_proj_weight
        self.o_proj_weight=o_proj_weight
        self.d_model=d_model
        self.num_heads=num_heads

        self.d_k = int(d_model/num_heads)
        self.rope=rope
        if self.rope:
            self.theta=theta
            self.max_seq_len=max_seq_len
            self.rope_class = myRotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
            self.token_positions=token_positions

    def forward(self, x:Float[Tensor, " ... sequence_length d_in"])-> Float[Tensor, " ... sequence_length d_out"]:

        xq=einsum(x, self.q_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k" )
        xk=einsum(x, self.k_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k" )
        xv=einsum(x, self.v_proj_weight, "... sequence_length d_in, d_v d_in -> ... sequence_length d_v" )
        # heads*d_k_h=d_k
        xq=rearrange(xq, "... sequence_length (heads d_k_h) -> ... heads sequence_length d_k_h", heads=self.num_heads)
        xk=rearrange(xk, "... sequence_length (heads d_k_h) -> ... heads sequence_length d_k_h", heads=self.num_heads)
        xv=rearrange(xv, "... sequence_length (heads d_v_h) -> ... heads sequence_length d_v_h", heads=self.num_heads)

        if self.rope: 
            xq=self.rope_class.forward(x=xq, token_positions=self.token_positions)#
            xk=self.rope_class.forward(x=xk, token_positions=self.token_positions)

        multihead=scaled_dot_product_attention(Q=xq, K=xk, V=xv)
        multihead=rearrange(multihead, "... heads seq_len d_v_h -> ... seq_len (heads d_v_h)") 
        multiheadW0=einsum(self.o_proj_weight, multihead,  "d_model d_v,... seq_len d_v -> ... seq_len d_model ")

        return multiheadW0


class causalMultiHeadSelfAttention__(nn.Module): 
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
            in_features: Float[Tensor, " ... sequence_length d_in"],
            rope:bool=False,
            max_seq_len:int=None,
            token_positions:  Int[Tensor, " ... sequence_length"] | None = None,
            theta: float=1000.0, 
            ):
        super().__init__()
        # breakpoint()
        #d_model=d_in=lenght of each word embedding
        self.in_features=in_features
        self.q_proj_weight=rearrange(q_proj_weight, "(heads d_k_h) d_in -> heads d_k_h d_in", heads=num_heads) 
        self.k_proj_weight=rearrange(k_proj_weight, "(heads d_k_h) d_in -> heads d_k_h d_in", heads=num_heads) 
        self.v_proj_weight=rearrange(v_proj_weight, "(heads d_v_h) d_in -> heads d_v_h d_in", heads=num_heads) 
        # breakpoint()
        self.o_proj_weight=o_proj_weight
        self.d_model=d_model
        self.num_heads=num_heads
        
        self.d_k = d_model/num_heads
        breakpoint()
        if rope:
            self.rope=rope
            self.theta=theta
            self.max_seq_len=max_seq_len
            self.rope_class = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
            self.token_positions=token_positions

    def multi_head_self_attention(self)-> Float[Tensor, " ... sequence_length d_out"]:
        # breakpoint()
        xq=einsum(self.in_features, self.q_proj_weight, "batch sequence_length d_in, heads d_in d_k_h -> batch heads sequence_length d_k_h" )
        xk=einsum(self.in_features, self.k_proj_weight, "batch sequence_length d_in, heads d_in d_k_h -> batch heads sequence_length d_k_h" )
        xv=einsum(self.in_features, self.v_proj_weight, "batch sequence_length d_in, heads d_in d_v_h -> batch heads sequence_length d_v_h" )
        # breakpoint()        
        if self.rope: 
            xq=self.rope_class.forward(x=xq, token_positions=self.token_positions)
            xk=self.rope_class.forward(x=xk, token_positions=self.token_positions)

        multihead=scaled_dot_product_attention(Q=xq, K=xk, V=xv)
        breakpoint()
        multihead=rearrange(multihead, "batch heads seq_len d_v_h -> batch seq_len (heads d_v_h)") 
        multiheadW0=einsum(multihead, self.o_proj_weight,  "batch seq_len d_v, d_v d_out -> batch seq_len d_out ")
        # breakpoint()
        return multiheadW0
    


class CausalMultiHeadSelfAttention_noneinops(nn.Module):
    """Multi-Head Self-Attention

    This function implements section 3.2.2 of the Transformer paper. In particular,
    given an input tensor of shape `(batch_size, sequence_length, d_model)`, we project
    it to create queries, keys, and values, and then perform causl multi-headed attention with
    those queries, keys, and values.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        attn_pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            q_proj_weight: Float[Tensor, " d_k d_in"],
            k_proj_weight: Float[Tensor, " d_k d_in"],
            v_proj_weight: Float[Tensor, " d_v d_in"],
            o_proj_weight: Float[Tensor, " d_model d_v"],
            in_features: Float[Tensor, " ... sequence_length d_in"],
            rope:bool=False,
            max_seq_len:int=None,
            token_positions:  Int[Tensor, " ... sequence_length"] | None = None,
            theta: float=1000.0):

        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.in_features=in_features
        self.attn_pdrop=None
        self.d_k = int(d_model / num_heads)
        self.d_v = self.d_k

        self.q_proj_weight=q_proj_weight
        self.k_proj_weight=k_proj_weight
        self.v_proj_weight=v_proj_weight

        # W_{O} in the Transformer paper.
        self.output_proj_weight=o_proj_weight
        
        if rope:
            self.rope=rope
            self.theta=theta
            self.max_seq_len=max_seq_len
            self.rope_class = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
            self.token_positions=token_positions

    def scaled_dot_product_attention(self,
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


    def forward(self, x: torch.FloatTensor):
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                The input to perform multi-headed self-attention on.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, d_model)`.
        """
        batch_size, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        # Project the keys, queries, and values for grouped-query attention.
        # Q is of shape (batch_size, seq_len, num_heads * key_dim)
        Q = torch.matmul(x, self.q_proj_weight.transpose(-1,-2))
        # Q = self.q_proj(x)
        # K is of shape (batch_size, seq_len, num_key_value_heads * key_dim)
        K = torch.matmul(x, self.k_proj_weight.transpose(-1,-2))
        # K = self.k_proj(x)
        # V is of shape (batch_size, seq_len, num_key_value_heads * key_dim)
        V = torch.matmul(x, self.v_proj_weight.transpose(-1,-2))
        # V = self.v_proj(x)

        # Reshape Q, K, V to (batch_size, num_heads, seq_len, d_k).
        # First, we reshape from (batch_size, seq_len, d_model) to (batch_size, seq_len, num_heads, d_k)
        # Then, we transpose to go from (batch_size, seq_len, num_heads, d_k) to (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, sequence_length, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.num_heads, self.d_k).transpose(1, 2)
        # TODO(nfliu): check if register_buffer is faster?
        causal_mask = torch.triu(torch.ones(sequence_length, sequence_length, device=x.device), diagonal=1).bool()
        # Shape: (batch_size, num_heads, sequence_length, d_k)
        attn_output = self.scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)
        # Now, we need to "concat" the outputs of the different heads by reshaping to
        # (batch_size, sequence_length, num_heads * d_v).
        # First, we need to undo the earlier transpose to go from (batch_size, num_heads, sequence_length, d_v)
        # to (batch_size, sequence_length, num_heads, d_v)
        # Then, we combine the last two dimensions via reshaping to (batch_size, sequence_length, num_heads * d_v)
        attn_output = (attn_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_v * self.num_heads))
        # Apply the output projection
        # output = self.output_proj(attn_output)
        output = torch.matmul(attn_output, self.output_proj_weight)
        return output


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

class causalMultiHeadSelfAttention___(nn.Module): 
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

    
def scaled_dot_product_attention_(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.Tensor] = None,
    pdrop: Optional[float] = None,
):
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        K: torch.FloatTensor
            Tensor with attention keys. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        Q: torch.FloatTensor
            Tensor with attention queries. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        V: torch.FloatTensor
            Tensor with attention values. Shape is
            (batch_size, ..., seq_len, value_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
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
    d_k = K.size(-1)
    # Shape: (batch_size, sequence_length, sequence_length)
    attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    # Apply the mask, if we have one.
    # breakpoint()
    if mask is not None:
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
    attention_weights = softmax(attention_scores, dim=-1)
    # NOTE: This dropout isn't really mentioned in the paper (besides the start of section 6.3,
    # "We performed only a small number of experiments to select the dropout, both
    # attention and residual"), but it appears in T2T.
    # https://stats.stackexchange.com/questions/509798/attention-dropout-where-was-it-proposed-used-first
    if pdrop is not None:
        attention_weights = F.dropout(attention_weights, p=pdrop)
    # Shape: (batch_size, sequence_length, d_v)
    return attention_weights @ V

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