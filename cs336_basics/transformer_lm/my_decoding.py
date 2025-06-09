import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from jaxtyping import Float, Int
from torch import Tensor

def decode(
    model: torch.nn.Module,
    prompt: Union[Int[Tensor, "batch_size seq_len"], List[int]],
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token: int = 50256,  # <|endoftext|> token ID
    device: Optional[torch.device] = None,
    batch_size: int = 1,
    early_stopping: bool = True
) -> Tuple[Int[Tensor, "batch_size generated_seq_len"], List[float]]:
    """
    Generate text completions for a given prompt using the language model.
    
    Args:
        model: The transformer language model
        prompt: Input prompt tensor of shape (batch_size, seq_len) or list of token IDs
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (higher = more random, lower = more deterministic)
        top_p: Nucleus sampling parameter (1.0 = no filtering, lower = more focused sampling)
        end_token: Token ID that signals the end of generation
        device: Device to run the model on
        batch_size: Number of sequences to generate in parallel
        early_stopping: Whether to stop generation when all sequences hit end_token
        
    Returns:
        Tuple containing:
        - Generated sequence tensor of shape (batch_size, generated_seq_len)
        - List of log probabilities for each generated token
        
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If model fails to generate tokens
    """
    # Input validation
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be in (0, 1]")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    
    # Convert prompt to tensor if it's a list
    if isinstance(prompt, list):
        prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
    
    # Ensure prompt is 2D
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
    
    # Set device
    if device is None:
        device = next(model.parameters()).device
    prompt = prompt.to(device)
    
    batch_size = prompt.shape[0]
    generated = prompt.clone()
    log_probs = []
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    try:
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
                
                # Update finished sequences
                if early_stopping:
                    finished = finished | (next_token.squeeze(-1) == end_token)
                    if finished.all():
                        break
    
    except Exception as e:
        raise RuntimeError(f"Error during generation: {str(e)}")
    
    return generated, log_probs
