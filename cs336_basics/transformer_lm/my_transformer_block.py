#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys
import numpy as np
from typing import Optional, List
import torch.nn.functional as F

from cs336_basics.transformer_lm.my_embedding import Embedding
from cs336_basics.transformer_lm.my_feedforward_swiglu import swiglu
from cs336_basics.transformer_lm.my_linear import Linear
from cs336_basics.transformer_lm.my_transformer_attention import causalMultiHeadSelfAttention
from cs336_basics.transformer_lm.my_transformer_block_elements import positionwise_feedforward, RMSLayerNorm, softmax




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


class my_transformer_block(nn.Module):
    def __init__(self, 
                d_model: int,
                num_heads: int,
                d_ff: int,
                max_seq_len: int,
                theta: float,
                weights: dict[str, torch.FloatTensor],
                in_features: torch.FloatTensor, 
                iteration: int|None=None, 
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32): 
   
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.in_features = in_features
        self.token_positions = torch.arange(0, max_seq_len, dtype=torch.int32)
        self.device = device
        self.dtype = dtype
 
        if iteration is None:
            self.weights=weights
        else: #if type(iteration) is int:
            self.weights={}
            self.weights['ln1.weight']=weights[f'layers.{iteration}.ln1.weight']
            self.weights['ln2.weight']=weights[f'layers.{iteration}.ln2.weight']
            self.weights['attn.q_proj.weight']=weights[f'layers.{iteration}.attn.q_proj.weight']
            self.weights['attn.k_proj.weight']=weights[f'layers.{iteration}.attn.k_proj.weight']
            self.weights['attn.v_proj.weight']=weights[f'layers.{iteration}.attn.v_proj.weight']
            self.weights['attn.output_proj.weight']=weights[f'layers.{iteration}.attn.output_proj.weight']
            self.weights['ffn.w1.weight']=weights[f'layers.{iteration}.ffn.w1.weight']
            self.weights['ffn.w2.weight']=weights[f'layers.{iteration}.ffn.w2.weight']
            self.weights['ffn.w3.weight']=weights[f'layers.{iteration}.ffn.w3.weight']


    def multihead_self_attention_sublayer(self, 
                                          in_features: torch.FloatTensor) -> torch.FloatTensor:

        x = RMSLayerNorm(d_model=self.d_model,
                         eps=1e-5,
                         weights=self.weights['ln1.weight'],
                         device=self.device,
                         dtype=self.dtype).forward(x=in_features)

        x = causalMultiHeadSelfAttention(d_model=self.d_model,
                                    num_heads=self.num_heads,
                                    q_proj_weight=self.weights['attn.q_proj.weight'],
                                    k_proj_weight=self.weights['attn.k_proj.weight'],
                                    v_proj_weight=self.weights['attn.v_proj.weight'],
                                    o_proj_weight=self.weights['attn.output_proj.weight'], 
                                    rope=True, 
                                    max_seq_len=self.max_seq_len,
                                    token_positions=self.token_positions,
                                    theta=self.theta).forward(x=x)

        x+=in_features
        return x


    def positionwise_feedforward_sublayer(self, 
                                          in_features: torch.FloatTensor) -> torch.FloatTensor:
        x = RMSLayerNorm(d_model=self.d_model,
                         eps=1e-5,
                         weights=self.weights['ln2.weight'],
                         device=self.device,    
                         dtype=self.dtype).forward(x=in_features)
                         
        x = swiglu(d_model=self.d_model,
                  d_ff=self.d_ff,
                  w1_weight=self.weights['ffn.w1.weight'],
                  w2_weight=self.weights['ffn.w2.weight'],
                  w3_weight=self.weights['ffn.w3.weight'],
                  in_features=x)
        
        x+=in_features
        return x

     
    def forward(self, 
                in_features: torch.FloatTensor) -> torch.FloatTensor:
        x= self.multihead_self_attention_sublayer(in_features=in_features)

        x= self.positionwise_feedforward_sublayer(in_features=x)
        return x


class my_transformer_lm(nn.Module):
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE Theta parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """

    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 weights: dict[str, torch.FloatTensor],
                 vocab_size: int,
                 context_length: int,
                 num_layers: int, 
                 device: torch.device=torch.device('cpu'),
                 dtype: torch.dtype=torch.float32):
                         
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta    
        self.weights = weights
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.max_seq_len = context_length
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        self.embedding_layer=Embedding(vocab_size=vocab_size,
                                       d_model=d_model,
                                       weights=self.weights['token_embeddings.weight'],
                                       device=device,
                                       dtype=dtype)
        

    def forward(self, 
                in_indices: torch.IntTensor) -> torch.FloatTensor:
        
        x=self.embedding_layer(in_indices)

        for i in range(self.num_layers):
            x=my_transformer_block(d_model=self.d_model,
                                   num_heads=self.num_heads,
                                   d_ff=self.d_ff,
                                   max_seq_len=self.max_seq_len,
                                   theta=self.rope_theta,
                                   weights=self.weights,
                                   in_features=x, 
                                   iteration=i).forward(in_features=x)
            
        x=RMSLayerNorm(d_model=self.d_model,
                       eps=1e-5,
                       weights=self.weights['ln_final.weight'],
                       device=self.device,
                       dtype=self.dtype).forward(x=x)
        
        x=Linear(d_in=self.d_model,
                 d_out=self.vocab_size,
                 weights=self.weights['lm_head.weight'],
                 device=self.device,
                 dtype=self.dtype).forward(x=x)
        
        return x
        
        



                       
        
        


        

