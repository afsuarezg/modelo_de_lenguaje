#!/usr/bin/env python3
from einops import rearrange, einsum, reduce
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import os
import argparse
from datetime import datetime
import time    # Import tiktoken for tokenization
import tiktoken

from cs336_basics.transformer_lm.my_transformer_block import my_transformer_lm
from cs336_basics.transformer_lm.my_training_utils import data_loading, save_checkpoint, load_checkpoint, get_device, learning_rate_schedule
from cs336_basics.transformer_lm.my_loss_optimizer import cross_entropy, AdamW, gradient_clipping 
from cs336_basics.transformer_lm.my_transformer_block_elements import softmax

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process training hyperparameters for an LLM.")
 
    # Training name and data paths
    parser.add_argument("--name", type=str, default="default_training", help="Name of the training run.")
    parser.add_argument("--training_data_path", type=str, default=r"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\2023-2024\Spring\CS336 -  LLMs scratch\Assignments\Assignment 1\repo-july24\data\TinyStoriesV2-GPT4-train.txt", help="Path to the training data file.")
    parser.add_argument("--validation_data_path", type=str, default=r"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\2023-2024\Spring\CS336 -  LLMs scratch\Assignments\Assignment 1\repo-july24\data\TinyStoriesV2-GPT4-valid.txt", help="Path to the validation data file.")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    
    # Transformer architecture parameters
    parser.add_argument("--vocab_size", type=int, default=50257, help="Size of the vocabulary/number of tokens.")
    parser.add_argument("--context_length", type=int, default=64, help="The maximum number of tokens to process at once.")
    parser.add_argument("--d_model", type=int, default=768, help="Dimensionality of the feedforward input and output.")
    parser.add_argument("--num_transformer_layers", type=int, default=12, help="Number of transformer layers in the model.")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads to use in multi-head attention.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="The maximum sequence length for the model.")
    parser.add_argument("--d_ff", type=int, default=3072, help="Dimensionality of the feedforward network's inner layer.")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="Base value for RoPE positional embeddings.")
    parser.add_argument("--eps_rmsnorm", type=float, default=1e-5, help="A value added to the denominator for numerical stability.")
    parser.add_argument("--rope", type=bool, default=True, help="Whether to use RoPE positional embeddings.")
    
    # Loss function parameters
    parser.add_argument("--loss_function", type=str, choices=["cross_entropy", "mse", "kl_divergence"], default="cross_entropy", help="Loss function to use.")

    # Optimizer parameters
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd", "rmsprop"], default="adamw", help="Optimizer for training.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 coefficient for AdamW optimizer.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 coefficient for AdamW optimizer.") 
    parser.add_argument("--eps_adamw", type=float, default=1e-8, help="Epsilon value for numerical stability in AdamW optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate.")

    # Learning rate scheduler parameters
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warm-up steps for learning rate scheduling.")
    parser.add_argument("--max_learning_rate", type=float, default=1e-4, help="The maximum learning rate for cosine learning rate schedule.")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="The number of iterations to linearly warm-up the learning rate.")
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000, help="The number of cosine annealing iterations")
    parser.add_argument("--min_learning_rate", type=float, default=1e-5, help="The minimum learning rate for cosine learning rate schedule.")

    # Gradient clipping parameters
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="Maximum gradient norm for clipping.")
    parser.add_argument("--max_l2_norm", type=float, default=1.0, help="A positive value containing the maximum l2-norm.")
    
    # Training loop parameters
    parser.add_argument("--num_train_steps", type=int, default=1000, help="Total number of training steps.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of training examples per batch.")    
    parser.add_argument("--num_epochs", type=int, default=10, help="Total number of training epochs.")
    parser.add_argument("--validation_frequency", type=int, default=1000, help="Number of steps between validation runs.")
    parser.add_argument("--in_indices", type=torch.LongTensor, default=None, help="Input token indices.")
    parser.add_argument("--wandb_upload", type=bool, default=False, help="Whether to upload metrics to wandb.")

    # Checkpoint parameters
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="Save model checkpoint every N steps.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")

    # Precision parameters
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Floating-point precision type.")

    # Additional settings
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps before performing a gradient update.")
    
    args = parser.parse_args()
    return args


def llm_train_loop(training_name:str,#='default1',
                training_data_path:str, #=r"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\2023-2024\Spring\CS336 - LLMs scratch\Assignments\Assignment 1\repo-july24\data\TinyStoriesV2-GPT4-train.txt" ,
                validation_data_path:str,#=r"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\2023-2024\Spring\CS336 - LLMs scratch\Assignments\Assignment 1\repo-july24\data\TinyStoriesV2-GPT4-valid.txt",
                vocab_size:int,#=50257,  # GPT-2 vocab size
                context_length:int,#=64,  # Standard GPT-2 context length
                d_model:int,#=768,  # Hidden dimension
                num_layers:int,#=12,  # Number of transformer layers
                d_ff:int,#=3072,  # Feed-forward dimension (4x d_model)
                num_heads:int,#=12,  # Number of attention heads
                eps_rmsnorm:float,#=1e-5,  # RMSNorm epsilon
                warmup_steps:int,#=1000,
                max_learning_rate:float,#=1e-4,  # Peak learning rate
                min_learning_rate:float,#=1e-5,  # Minimum learning rate
                warmup_iters:int,#=1000,  # Learning rate warmup iterations
                cosine_cycle_iters:int,#=10000,  # Cosine cycle length
                batch_size:int,#=32,  # Training batch size
                optimizer:str,#="adamw",  # Optimizer choice
                loss_function:str,#="cross_entropy",  # Loss function
                num_epochs:int,#=10,  # Number of training epochs
                max_l2_norm:float,#=1.0,  # Gradient clipping norm
                in_indices:torch.LongTensor,#=None,  # Input token indices
                num_train_steps:int,#=100,  # Total training steps
                weight_decay:float,#=0.01,  # Weight decay coefficient
                device:str,#="cuda",
                rope:bool,#=True,
                rope_theta:float,#=10000.0,
                beta1:float,#=0.9,
                beta2:float,#=0.999,
                eps_adamw:float,#=1e-8,
                gradient_clipping:float,#=1.0,
                checkpoint_interval:int,#=5000, 
                checkpoint_dir:str,#="checkpoints",
                dtype:str,#="fp32",
                gradient_accumulation_steps:int,#=1,
                validation_frequency:int,#=1000,
                wandb_upload:bool):#=False):
    

    device= (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    # Track total training time
    total_start_time = time.time()
    step_times = []  # List to store time per step
    
    #initialize weights from scrath
    weights={
        'token_embeddings.weight': torch.randn(vocab_size, d_model),
        'ln_final.weight': torch.randn(d_model),
        'ln_final.bias': torch.randn(d_model),
        'lm_head.weight': torch.randn(vocab_size, d_model),
        'lm_head.bias': torch.randn(vocab_size)
    }

    # Add weights for each transformer layer
    for i in range(num_layers):
        # Attention weights
        weights[f'layers.{i}.attn.q_proj.weight'] = torch.randn(num_heads * (d_model // num_heads), d_model)
        weights[f'layers.{i}.attn.k_proj.weight'] = torch.randn(num_heads * (d_model // num_heads), d_model) 
        weights[f'layers.{i}.attn.v_proj.weight'] = torch.randn(num_heads * (d_model // num_heads), d_model)
        weights[f'layers.{i}.attn.output_proj.weight'] = torch.randn(d_model, d_model)

        # Layer norms
        weights[f'layers.{i}.ln1.weight'] = torch.randn(d_model)
        weights[f'layers.{i}.ln2.weight'] = torch.randn(d_model)

        # Feedforward weights
        weights[f'layers.{i}.ffn.w1.weight'] = torch.randn(d_model, d_ff)
        weights[f'layers.{i}.ffn.w2.weight'] = torch.randn(d_ff, d_model)
        weights[f'layers.{i}.ffn.w3.weight'] = torch.randn(d_model, d_ff)

    #wandb config
    if wandb_upload:
        wandb_config = {
        "name": training_name,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "rope": rope,
        "rope_theta": rope_theta,
        "num_heads": num_heads,
        "eps_rmsnorm": eps_rmsnorm,
        "max_learning_rate": max_learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "cosine_cycle_iters": cosine_cycle_iters,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "loss_function": loss_function,
        "num_epochs": num_epochs,
        "max_l2_norm": max_l2_norm,
        "in_indices": in_indices,
        "num_train_steps": num_train_steps,
        "weight_decay": weight_decay,
        "validation_frequency": validation_frequency}

        current_date = datetime.now().strftime("%Y-%m-%d")
        wandb.init(project="LLM_train", name=current_date, config=wandb_config)   


    model=my_transformer_lm(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope=rope,
        rope_theta=rope_theta,
        weights=weights,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        device=device,
        dtype=dtype)

    optimizer=AdamW(model.parameters(), 
                    it=0,
                    max_learning_rate=max_learning_rate,
                    min_learning_rate=min_learning_rate,
                    warmup_iters=warmup_iters,
                    cosine_cycle_iters=cosine_cycle_iters,
                    betas=(beta1, beta2),
                    eps=eps_adamw, 
                    weight_decay=weight_decay)

    optimizer=torch.optim.AdamW(model.parameters(), 
                                lr=0.001, 
                                betas=(beta1, beta2), 
                                eps=eps_adamw, 
                                weight_decay=weight_decay)

    # Create tiktoken encoder using GPT-2 encoding
    enc = tiktoken.get_encoding("gpt2")
    # breakpoint()
    # Read and encode training data lazily
    training_chunks = []
    with open(training_data_path, 'r', encoding='utf-8') as f:
        # while chunk := f.read(1024 * 1024):  # Read 1MB at a time
        content=f.read()
    
    training_chunks.extend(enc.encode(content[:2000], allowed_special={'<|endoftext|>'}))
    training_chunks = np.array(training_chunks, dtype=np.int32)
    # training_chunks_tensor=torch.tensor(training_chunks, dtype=torch.int32)
    
    # Read and encode validation data lazily
    validation_chunks = []
    with open(validation_data_path, 'r', encoding='utf-8') as f:
        # while chunk := f.read(1024 * 1024):  # Read 1MB at a time
        content=f.read()
    validation_chunks.extend(enc.encode(content[:2000], allowed_special={'<|endoftext|>'}))
    validation_chunks = np.array(validation_chunks, dtype=np.int32)
    # validation_chunks_tensor=torch.tensor(validation_chunks, dtype=torch.int32)

    training_data=training_chunks
    validation_data=validation_chunks

    # training_data= np.memmap(filename=training_data_path, dtype=np.int32)
    # validation_data=np.memmap(filename=validation_data_path, dtype=np.int32)
   
    for t in range(num_train_steps):
        optimizer.zero_grad(set_to_none=True)
        step_start_time = time.time()  # Start timing this step

        input_tensor_training, targets_tensor_training = data_loading(dataset=training_data,
                                                            batch_size=batch_size,
                                                            context_length=context_length,
                                                            device=device)
        
   
        # Forward (compute loss)
        training_pred_tensor = model(input_tensor_training)

        #TODO: apply softmax here
        training_pred_tensor=softmax(training_pred_tensor)
        training_pred_tensor=rearrange(training_pred_tensor, ' "batch_size sequence_length vocab_size" -> (batch_size sequence_length) vocab_size')
        targets_tensor_training=rearrange(targets_tensor_training, ' "batch_size sequence_length" -> (batch_size sequence_length)')
        training_loss = cross_entropy(training_pred_tensor, targets_tensor_training)

        # Calculate elapsed time
        elapsed_time = time.time() - total_start_time
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        
        # Validation loss - now using validation_frequency
        if t % validation_frequency == 0:
            input_tensor_validation, target_tensor_validation = data_loading(dataset=validation_data,
                                                                        batch_size=batch_size,
                                                                        context_length=context_length,
                                                                        device=device)
            
            validation_pred_tensor = model(input_tensor_validation)
            validation_pred_tensor=softmax(validation_pred_tensor)
            validation_pred_tensor=rearrange(validation_pred_tensor, ' "batch_size sequence_length vocab_size" -> (batch_size sequence_length) vocab_size')
            target_tensor_validation=rearrange(target_tensor_validation, ' "batch_size sequence_length" -> (batch_size sequence_length)')
            validation_loss = cross_entropy(validation_pred_tensor, target_tensor_validation)
        
            # Log training metrics to wandb
            if wandb_upload:
                wandb.log({
                    "step": t,
                    "training_loss": training_loss.item(),
                    "validation_loss": validation_loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "wall_clock_time": elapsed_time,  # Total elapsed time
                    "avg_step_time": avg_step_time,  # Average time per step
                    "steps_per_second": 1.0 / avg_step_time if avg_step_time > 0 else 0,  # Training speed
                    "estimated_time_remaining": (num_train_steps - t) * avg_step_time if avg_step_time > 0 else 0  # ETA
                })

        # Backward (compute gradients)
        training_loss.backward()
        gradient_clipping(model.parameters(),
                        max_l2_norm=max_l2_norm)
        
        # Update parameters        
        # optimizer.learning_rate_schedule()     
        optimizer.step()

        # Record step time
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        if len(step_times) > 100:  # Keep only last 100 steps for moving average
            step_times.pop(0)

        # Checkpointing 
        if t % checkpoint_interval == 0:
            checkpoint_path = os.path.join("results", f"checkpoint_iter_{t}.bin")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            out = open(checkpoint_path, "wb")
            # out = open(r"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\2023-2024\Spring\CS336 -  LLMs scratch\Assignments\Assignment 1\results\file.bin", "wb")
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            iteration=t,
                            out=out)
    
    
    # Log final training time
    total_training_time = time.time() - total_start_time
    if wandb_upload:
        wandb.log({
            "total_training_time": total_training_time,
            "final_steps_per_second": num_train_steps / total_training_time
        })



def main():
    args = parse_arguments()

    print("Training Hyperparameters:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    
    # Call llm_train_loop with parsed arguments
    llm_train_loop(
        training_name=args.name if hasattr(args, 'name') else "default_training",
        training_data_path=args.training_data_path if hasattr(args, 'training_data_path') else "training_data.bin",
        validation_data_path=args.validation_data_path if hasattr(args, 'validation_data_path') else "validation_data.bin",
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_transformer_layers,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        eps_rmsnorm=args.eps_rmsnorm,
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        loss_function=args.loss_function,
        num_epochs=args.num_epochs,
        max_l2_norm=args.max_l2_norm,
        num_train_steps=args.num_train_steps,
        # lr=args.lr if hasattr(args, 'lr') else 3e-4,
        weight_decay=args.weight_decay,
        device=args.device,
        rope_theta=args.rope_theta,
        beta1=args.beta1,
        beta2=args.beta2,
        eps_adamw=args.eps_adamw,
        warmup_steps=args.warmup_steps,
        gradient_clipping=args.gradient_clipping,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        dtype=args.dtype,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        validation_frequency=args.validation_frequency,
        wandb_upload=args.wandb_upload,
        in_indices=args.in_indices,
        rope=args.rope,
    )

if __name__ == "__main__":
    main()
    