from einops import rearrange
from jaxtyping import Float, Int
import torch 
from torch import Tensor
import torch.nn as nn


class Embedding(nn.Module):
    """
    Implement the Embedding class that inherits from torch.nn.Module and performs an
    embedding lookup. Your implementation should follow the interface of PyTorch’s built-in nn.Embedding module.
    """

    def __init__(self, 
            vocab_size: int,
            d_model: int,
            weights: Float[Tensor, "vocab_size d_model"],
            device: torch.device|None=None,
            dtype: torch.dtype|None=None): 
        
        super().__init__()
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.device=device
        self.dtype=dtype
        # x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
        if weights.shape[1] != d_model:
            weights = rearrange(weights, f"{vocab_size} {d_model} -> {vocab_size} {d_model}")
        
        self.embeddings=nn.Parameter(weights).to(device)
        breakpoint()
        
                
    def forward(self, token_ids: torch.LongTensor)-> Float[Tensor, "... d_model"]:
        """
        The forward method should select the embedding vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a torch.LongTensor of token IDs with shape (batch_size, sequence_length).
        """
        return self.embeddings[token_ids]
    