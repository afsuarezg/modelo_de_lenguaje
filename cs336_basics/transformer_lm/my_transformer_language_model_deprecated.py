#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys
import numpy as np
from typing import Optional, List
import torch.nn.functional as F

from cs336_basics.transformer_lm.my_transformer_block import my_transformer_block
from cs336_basics.transformer_lm.my_transformer_block_elements import RMSLayerNorm, softmax


class TransformerLM(nn.Module):
    def __init__(self, 
                vocab_size:int, 
                context_length:int,
                d_model:int, 
                num_layers:int,
                num_heads:int, 
                d_ff:int,
                attn_pdrop:float, 
                residual_pdrop:float,
                weights: dict[str, torch.FloatTensor], 
                in_indices: torch.LongTensor):
        
        super().__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.d_model=d_model
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.attn_pdrop=attn_pdrop
        self.residual_pdrop=residual_pdrop
        self.weights=weights
        self.in_indices=in_indices #Tensor with input indices to run the language model on. 
                                   #Shape is (batch_size, sequence_length), where 'sequence_length'is at most 'context_length'. 
        self.token_embeddings=weights['token_embeddings.weight']
        self.position_embeddings=weights['position_embeddings.weight']
        # self.token_embeddings=nn.Embedding(vocab_size, d_model)
        # self.absolute_pos_embeddings=nn.Embedding(context_length, d_model)
        self.batch_size=in_indices.shape[0]
        self.seq_length=in_indices.shape[1]
        breakpoint()




    def forward(self):
        token_embeddings=F.embedding(self.in_indices, self.token_embeddings)
        position_ids=torch.arange(self.seq_length).unsqueeze(0)
        position_embeddings=F.embedding(position_ids,self.position_embeddings)
        x = token_embeddings + position_embeddings

        #add & dropout
        x=F.dropout(input=x, p=self.residual_pdrop)

        #transformer blocks 
        for i in range(self.num_layers):
            x = my_transformer_block(d_model=self.d_model,
                                  num_heads=self.num_heads,
                                  attn_pdrop=self.attn_pdrop,
                                  residual_pdrop=self.residual_pdrop,
                                  d_ff=self.d_ff,
                                  weights=self.weights,
                                  in_features=x,
                                  current_layer=i)

        #normalization
        x=RMSLayerNorm(d_model=self.d_model,
                       eps=1e-5,
                       weights=self.weights,
                       in_features=x,
                       weight_name='ln_final.weight').forward()
        
        #linear
        x=torch.matmul(x, self.weights['lm_head.weight'].transpose(0,1))

        #softmax
        # x=softmax(in_features=x, dim=-1)

        return  x

