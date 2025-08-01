__all__ = ["softmax", "AdamW", "cosine_learning_rate_schedule", "gradient_clipping"]

import torch
import math
import numpy as np
from torch import Tensor, optim, nn
from jaxtyping import Float, Int
from einops import einsum
from collections.abc import Callable, Iterable
from typing import Optional, Tuple

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
        Given a tensor of inputs, return the output of softmaxing the given `dim`
        of the input.

        Args:
            in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
            dim (int): Dimension of the `in_features` to apply softmax to.

        Returns:
            Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
            softmax normalizing the specified `dim`.
        """
    max_in = torch.max(in_features, dim=dim, keepdim=True).values
    in_features = in_features - max_in
    output = torch.exp(in_features) / torch.sum(torch.exp(in_features), dim=dim, keepdim=True)
    return output

class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.99), weight_decay=1, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0 or betas[0] >= 1 :
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if betas[1] < 0 or betas[1] >= 1 :
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data

                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data, device=grad.device)
                    state["v"] = torch.zeros_like(p.data, device=grad.device)
                t = state["t"]
                m = state["m"]
                v = state["v"]

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                a_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= a_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss

def cosine_learning_rate_schedule(t:int, lr_max: float, lr_min:float, t_w:int, t_c:int) ->float:
    if t < t_w:
        return t * lr_max / t_w
    elif t > t_c:
        return lr_min
    else:
        return lr_min + 0.5 * (1 + np.cos(np.pi * (t - t_w) / (t_c - t_w))) * (lr_max - lr_min)

def gradient_clipping(parameters:Iterable[nn.Parameter] , max_norm:float):
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    grads_norm = torch.norm(torch.cat(grads), p=2)
    if grads_norm > max_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(max_norm / (grads_norm + eps))

def get_batch_data(data, batch_size:int, context_length:int, device:str) -> Tuple[torch.Tensor, torch.Tensor]:
    pass