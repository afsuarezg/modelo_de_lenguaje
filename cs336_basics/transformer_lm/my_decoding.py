import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from jaxtyping import Float, Int
from torch import Tensor

def decode(
    model: torch.nn.Module,
    prompt: Int[Tensor, "batch_size seq_len"],
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token: int = 50256,  # <|endoftext|> token ID
    device: Optional[torch.device] = None
) -> Tuple[Int[Tensor, "batch_size generated_seq_len"], List[float]]:
    """
    Generate text completions for a given prompt using the language model.
    
    Args:
        model: The transformer language model
        prompt: Input prompt tensor of shape (batch_size, seq_len)
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (higher = more random)
        top_p: Nucleus sampling parameter (1.0 = no filtering)
        end_token: Token ID that signals the end of generation
        device: Device to run the model on
        
    Returns:
        Tuple containing:
        - Generated sequence tensor of shape (batch_size, generated_seq_len)
        - List of log probabilities for each generated token
    """
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = prompt.shape[0]
    generated = prompt.clone()
    log_probs = []
    
    # Generate tokens one at a time
    for _ in range(max_tokens):
        # Get model predictions
        with torch.no_grad():
            logits = model(generated)
            next_token_logits = logits[:, -1, :]  # Get predictions for next token
            
            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create a mask for tokens to keep
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Get the log probability of the sampled token
            token_log_probs = torch.log(probs)
            selected_log_probs = token_log_probs.gather(1, next_token)
            log_probs.extend(selected_log_probs.squeeze(-1).tolist())
            
            # Append the new token to the generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if we've generated the end token
            if (next_token == end_token).all():
                break
    
    return generated, log_probs
