#!/usr/bin/env python3
import numpy as np
import torch
import numpy.typing as npt
import math
import os
import typing



def data_loading(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset: np.array
            1D numpy array of integer token IDs in the dataset.
        batch_size: int
            Desired batch size to sample.
        context_length: int
            Desired context length of each sampled example.
        device: str
            PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """    

    # Ensure dataset is long enough
    assert len(dataset) > context_length, "Dataset must be longer than the context length."
    
    # Generate start indices for each batch
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    
    # Prepare input sequences and corresponding targets
    input_sequences = []
    targets = []
    
    for idx in start_indices:
        input_sequence = dataset[idx:idx + context_length]
        target_sequence = dataset[idx + 1:idx + context_length + 1]
        input_sequences.append(input_sequence)
        targets.append(target_sequence)
    
    # Convert lists to numpy arrays
    input_sequences = np.array(input_sequences)
    targets = np.array(targets)
    
    # Convert numpy arrays to PyTorch tensors
    input_sequences_tensor = torch.tensor(input_sequences, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    
    return input_sequences_tensor, targets_tensor


def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model: torch.nn.Module
            Serialize the state of this model.
        optimizer: torch.optim.Optimizer,
            Serialize the state of this optimizer.
        iteration: int
            Serialize this value, which represents the number of training iterations
            we've completed.
        out: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(obj=checkpoint, f=out)


def load_checkpoint(src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]], 
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration


def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")
    


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    ):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it: int
            Iteration number to get learning rate for.
        max_learning_rate: float
            alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate: float
            alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters: int
            T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters: int
            T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        lr = (it/warmup_iters)*max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5*(1 + math.cos((it-warmup_iters)/(cosine_cycle_iters - warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
    else:
        lr = min_learning_rate
    return lr 