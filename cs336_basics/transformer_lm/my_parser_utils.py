import argparse
import torch


# Custom function to convert string to torch.device
def parse_device(device_str):
    if device_str == 'cpu':
        return torch.device('cpu')
    elif device_str.startswith('cuda'):
        return torch.device(device_str)
    elif device_str.startswith('mps'):  # For Apple M1/M2 GPUs
        return torch.device(device_str)
    else:
        try:
            return torch.device(device_str)
        except:
            raise argparse.ArgumentTypeError(f"Invalid device: {device_str}")
        
    
def parse_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    else:
        raise argparse.ArgumentTypeError(f"Invalid dtype: {dtype_str}")

    