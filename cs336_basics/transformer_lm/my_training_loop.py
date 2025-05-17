#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datetime import datetime

from my_transformer_language_model import TransformerLM
from my_training_utils import data_loading, save_checkpoint, load_checkpoint, get_device
from my_loss_optimizer import cross_entropy, AdamW, gradient_clipping 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process training hyperparameters for an LLM.")
 
    #hardware parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    
    #transformer architecture parameters
    parser.add_argument("--vocab_size", type=int, help="Size of the vocabulary/number of tokens.")
    parser.add_argument("--context_length", type=int, help="The maximum number of tokens to process at once.")
    parser.add_argument("--d_model", type=int, help="Dimensionality of the feedforward input and output.")
    parser.add_argument("--num_transformer_layers", type=int, help="Number of transformer layers in the model.")
    parser.add_argument("--num_heads", type=int, help="Number of heads to use in multi-head attention.")
    parser.add_argument("--max_seq_len", type=int, help="The maximum sequence length for the model.")
    parser.add_argument("--d_ff", type=int, help="Dimensionality of the feedforward network's inner layer.")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="Base value for RoPE positional embeddings.")
    parser.add_argument("--eps_rmsnorm", type=float, default=1e-5, help="A value added to the denominator for numerical stability.")
    
    #loss function parameters
    parser.add_argument("--loss_function", type=str, choices=["cross_entropy", "mse", "kl_divergence"], default="cross_entropy", help="Loss function to use.")

    #optimizer parameters
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd", "rmsprop"], default="adamw", help="Optimizer for training.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 coefficient for AdamW optimizer.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 coefficient for AdamW optimizer.") 
    parser.add_argument("--eps_adamw", type=float, default=1e-8, help="Epsilon value for numerical stability in AdamW optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization.")

    #learning rate scheduler parameters
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warm-up steps for learning rate scheduling.")
    parser.add_argument("--max_learning_rate", type=float, help="the maximum learning rate for cosine learning rate schedule.")
    parser.add_argument("--warmup_iters", type=int, help="The number of iterations to linearly warm-up the learning rate.")
    parser.add_argument("--cosine_cycle_iters", type=int, help="The number of cosine annealing iterations")
    parser.add_argument("--min_learning_rate", type=float, help="The minimum learning rate for cosine learning rate schedule.")

    #gradient clipping parameters
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="Maximum gradient norm for clipping.")
    parser.add_argument("--max_l2_norm", type=float, help="A positive value containing the maximum l2-norm.")

    
    #training loop parameters
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32, help="Number of training examples per batch.")    
    parser.add_argument("--num_epochs", type=int, default=10, help="Total number of training epochs.")

    #checkpoint parameters
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="Save model checkpoint every N steps.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")

    #precision parameters
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Floating-point precision type.")

    # Additional settings
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps before performing a gradient update.")
    breakpoint()
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    
    print("Training Hyperparameters:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")


def llm_train_loop(              
              name:str,
              vocab_size:int,
              context_length:int,  
              d_model:int,
              num_layers:int,
              d_ff:int, 
              attn_pdrop:float,
              num_heads:int, 
              residual_pdrop:float,
              eps_rmsnorm:float,
              max_learning_rate:float,
              min_learning_rate:float,
              warmup_iters:int,
              cosine_cycle_iters:int, 
              batch_size:int,
              optimizer:str,
              loss_function:str,
              num_epochs:int,
              max_l2_norm:float,  
              in_indices:torch.LongTensor, 
              num_train_steps:int,
              lr:float,
              weight_decay:float,
              device:str="cuda",
              rope_theta:float=10000.0,
              beta1:float=0.9,
              beta2:float=0.999,
              eps_adamw:float=1e-8,
              warmup_steps:int=1000,
              gradient_clipping:float=1.0,
              checkpoint_interval:int=5000,
              checkpoint_dir:str="checkpoints",
              dtype:str="fp32",
              gradient_accumulation_steps:int=1):
    
    device= (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    model=TransformerLM(vocab_size=vocab_size, 
                        context_length=context_length, 
                        d_model=d_model,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        d_ff=d_ff,
                        attn_pdrop=attn_pdrop,
                        residual_pdrop=residual_pdrop,
                        in_indices=in_indices,
                        weights={})

    config = {
        "name": name,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "attn_pdrop": attn_pdrop,
        "num_heads": num_heads,
        "residual_pdrop": residual_pdrop,
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
        "lr": lr,
        "weight_decay": weight_decay}
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    wandb.init(project="LLM_train", name=current_date, config=config)

    optimizer=AdamW(model.parameters(), 
                    lr=lr, 
                    betas=(0.9,0.999),
                    eps=1e-8, 
                    weight_decay=weight_decay)
    
    optimizer.zero_grads(set_to_none=True)
    
    training_data= np.memmap(filename='PENDING TRAINING FILE TOKENIZED', dtype=np.int32)
    validation_data=np.memmap(filename=f'PENDING VALIDATION FILE TOKENIZED', dtype=np.int32)
    
    for t in range(num_train_steps):
        input_tensor_training, targets_tensor_training = data_loading(dataset=training_data,
                                                             batch_size=batch_size,
                                                             context_length=context_length,
                                                             device=device)
        
        input_tensor_validation, target_tensor_validation = data_loading(dataset=validation_data,
                                                                         batch_size=batch_size,
                                                                         context_length=context_length,
                                                                         device=device)
        
        # Forward (compute loss)
        training_pred_tensor = model(input_tensor_training)
        training_loss = cross_entropy(training_pred_tensor, targets_tensor_training)
        
        # Validation loss 
        validation_pred_tensor=model(input_tensor_validation)
        validation_loss = cross_entropy(validation_pred_tensor, target_tensor_validation)
        
        # Log training metrics to wandb
        wandb.log({
            "step": t,
            "training_loss": training_loss.item(),
            "validation_loss": validation_loss.item(),
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Backward (compute gradients)
        training_loss.backward()
        gradient_clipping(model.parameters(),
                          max_l2_norm=max_l2_norm)
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grads(set_to_none=True)

        # Learning rate update
        optimizer.learning_rate_schedule()        

        # Checkpointing 
        if t%100==0:
            out = open(r"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\2023-2024\Spring\CS336 -  LLMs scratch\Assignments\Assignment 1\results\file.bin", "wb")
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            iteration=t,
                            out=out)
    



if __name__=="__main__":
    import argparse
    from datetime import datetime
    main()
    