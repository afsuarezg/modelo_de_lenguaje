#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys
import numpy as np
from typing import Optional, List
import torch.nn.functional as F
from torch.optim import Optimizer
from collections.abc import Callable, Iterable
import math

sys.path.insert(0, r'C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\2023-2024\Spring\CS336 -  LLMs scratch\Assignments\Assignment 1\repo-feb25\tests')


def cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs: torch.FloatTensor
            FloatTensor of shape (batch_size, num_classes). inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets: torch.LongTensor
            LongTensor of shape (batch_size, ) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Tensor of shape () with the average cross-entropy loss across examples.
    """
    #substract the largest element for numerical stability.
    max_values, _ = torch.max(inputs, dim=-1, keepdim=True)
    inputs = inputs - max_values

    # Cancel out log and exp whenever possible.
    targets = targets.unsqueeze(1)
    numerators = torch.gather(inputs, 1, targets)
    denominators = inputs.exp()
    denominators = denominators.sum(dim=-1, keepdim=False)
    denominators = denominators.log()
    denominators = denominators.unsqueeze(1)

    a = numerators - denominators
    a = -a.sum().divide(inputs.shape[0])

    return a


class AdamW(torch.optim.Optimizer):

    def __init__(self, 
                 params, 
                 it:int=0,
                 max_learning_rate:float=1e-4,
                 min_learning_rate:float=1e-5,
                 lr:float=1e-4,
                 warmup_iters:int=1000,
                 cosine_cycle_iters:int=10000,
                 betas=(0.9, 0.999), 
                 eps=1e-8, 
                 weight_decay=1e-2,
                 device: torch.device|None=None,
                 dtype: torch.dtype|None=None):
        
        self.params=params
        self.it=it
        self.max_learning_rate=max_learning_rate
        self.min_learning_rate=min_learning_rate
        self.warmup_iters=warmup_iters
        self.cosine_cycle_iters=cosine_cycle_iters
        self.betas=betas#
        self.eps=eps#
        self.weight_decay=weight_decay#
        self.lr=lr # type annotation allows None initially but enforces float when set
        self.device=device
        self.dtype=dtype
        
        defaults = dict(lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=weight_decay, device=self.device, dtype=self.dtype)
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable]=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr= group['lr']
            beta_1, beta_2 = group['betas']
            eps= group['eps']
            weight_decay=group['weight_decay']
            for p in group['params']:#al interior de los parámetros en el grupo
                # Compute the gradient of the loss at the current time step. 
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')              

                #Link to the self state 
                state = self.state[p]

                t = state.get('t', 1)
                #obtain the state from the dictionary, set it to zero for the first iteration
                # State initialization para la primera iteración
                prev_m_t= state.get("m", torch.zeros_like(grad))
                prev_v_t = state.get("v", torch.zeros_like(grad)) 

                #update the first moment estimate
                m_t = beta_1*prev_m_t + (1-beta_1)*grad
                #update the second moment estimate 
                v_t = beta_2*prev_v_t + (1-beta_2)*grad.square()
                #compute adjusted learning rate for iteration t

                lr_t = lr * (math.sqrt(1 - (beta_2**t))) / (1 - (beta_1**t))
                #update parameters
                p.data = p.data - lr_t*(m_t / (v_t.sqrt() + eps))
                #apply weight decay 
                p.data = p.data - lr*weight_decay*p.data

                state['t'] = t+1
                state['m'] = m_t
                state['v'] = v_t

        return loss


    def learning_rate_schedule(self):
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
        if self.it < self.warmup_iters:
            self.lr = (self.it/self.warmup_iters)*self.max_learning_rate
        elif self.warmup_iters <= self.it <= self.cosine_cycle_iters:
            self.lr = self.min_learning_rate + 0.5*(1 + math.cos((self.it-self.warmup_iters)/(self.cosine_cycle_iters - self.warmup_iters)*math.pi))*(self.max_learning_rate-self.min_learning_rate)
        else:
            self.lr = self.min_learning_rate
        


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: collection of trainable parameters.
        max_l2_norm: a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.

    Returns:
        None
    """

    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


def learning_rate_schedule(it: int,
                           max_learning_rate: float,
                           min_learning_rate: float, 
                           warmup_iters: int, 
                           cosine_cycle_iters: int):
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