from einops import einsum
import numpy as np
import torch
import torch.nn as nn
from typing import List

	
class myRotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device:torch.device|None=None):
        super().__init__()

        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.device=device
        self.ks=2*torch.arange(0, d_k/2).float()
        self.one_over_theta=1.0/(theta**(self.ks/d_k)).to(device)
        self.num_positions=torch.arange(max_seq_len)
        self.angles=torch.outer(self.num_positions, self.one_over_theta).float()
        self.blocks=[torch.tensor(self.make_block_diag(n=d_k, values=self.angles[i])) for i in range(len(self.angles))]
        
        self.blocks_diagonal_matrix=torch.stack(self.blocks, dim=0)
        self.blocks_diagonal_matrix=self.blocks_diagonal_matrix.numpy()
        range_d_k=range(d_k)
        for blocks in self.blocks_diagonal_matrix:  
            blocks[range_d_k, range_d_k] = np.cos(blocks[range_d_k, range_d_k])

        a=[e for e in range(0,d_k,2)]
        b=[e for e in range(1,d_k,2)]
        for blocks in self.blocks_diagonal_matrix: 
            blocks[a,b] = -np.sin(blocks[a,b])

        e=[e for e in range(1,d_k,2)]
        f=[e for e in range(0,d_k,2)]
        for blocks in self.blocks_diagonal_matrix: 
            blocks[e,f] = np.sin(blocks[e,f])
        self.blocks_diagonal_matrix=torch.from_numpy(self.blocks_diagonal_matrix).to(torch.float32)
        

    def make_block_diag(self, n:int, values:List):
        assert n % 2 == 0, "n must be even"
        assert len(values) == n // 2, "values must be of length n/2"

        blocks = [np.full((2, 2), val) for val in values]
        return np.block([[blocks[i] if i == j else np.zeros((2, 2)) for j in range(n // 2)] for i in range(n // 2)])
    

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor):
        """
        x: shape (batch_size, seq_len, d_k)
        token_positions: shape (batch_size, seq_len)
        """

        min_seq_len = min(x.shape[-2], token_positions.shape[-1])
        token_positions = torch.arange(min_seq_len, device=self.device)
        # token_positions = list(range(min_seq_len))

        assert x.shape[-2]==len(token_positions), "x and token_positions must have the same sequence length"
        assert x.shape[-1]==self.d_k, "x must have the same number of columns as d_k"
        # breakpoint()

        # Get the block diagonal matrix for the current token positions
        rotated_vectors=einsum(self.blocks_diagonal_matrix[token_positions], x, " ... seq_len d_k d_a, ... seq_len d_a-> ... seq_len d_k")

        # Save rotated vectors tensor to file
        # Get the root directory of the repo
        torch.save(rotated_vectors, r"c:\Users\Andres.DESKTOP-D77KM25\Documents\assignment1-basics\printsdebug\rotated_vectors.pt")
        
        return rotated_vectors


    def save_block_diagonal_matrix(self, filepath: str):
        """
        Save the block diagonal matrix to a file using torch.save
        
        Args:
            filepath: Path where to save the matrix
        """
        torch.save(self.blocks_diagonal_matrix, filepath)


if __name__ == "__main__":
    torch.set_printoptions(linewidth=5000)
    rope = myRotaryPositionalEmbedding(theta=10000, d_k=10, max_seq_len=12)
    print(rope.block_diagonal_matrix)
