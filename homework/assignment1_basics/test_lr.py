# from collections.abc import Callable, Iterable
# from typing import Optional
# import torch
# import math
# import numpy as np
# import seaborn as sns
#
# class SGD(torch.optim.Optimizer):
#     def __init__(self, params, lr=1e-3):
#         if lr < 0:
#             raise ValueError(f"Invalid learning rate: {lr}")
#         defaults = {"lr": lr}
#         super().__init__(params, defaults)
#     def step(self, closure: Optional[Callable] = None):
#         loss = None if closure is None else closure()
#         for group in self.param_groups:
#             lr = group["lr"] # Get the learning rate.
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 state = self.state[p]
#                 t = state.get("t", 0)
#                 grad = p.grad.data
#                 p.data -= lr / math.sqrt(t + 1) * grad
#                 state["t"] = t + 1
#         return loss
#
# torch.manual_seed(123456)
#
# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# weights2 = torch.nn.Parameter(5 * torch.randn((10, 10)))
# weights3 = torch.nn.Parameter(5 * torch.randn((10, 10)))
#
# data = np.empty((3, 10))
# opt = SGD([weights], lr=1e1)
# for t in range(10):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     num[0][t] = loss.cpu().item()
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.
#
# data = num.data.numpy()
# sns.lineplot(data=data, x="t", y="data")
#
#
#
# opt = SGD([weights2], lr=1e2)
# for t in range(10):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     num[1][t] = loss.cpu().item()
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.
#
# opt = SGD([weights3], lr=1e3)
# for t in range(10):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     num[2][t] = loss.cpu().item()
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.
#
# data = num.data.numpy()
# print(data)
# import numpy
# a = numpy.load(r"C:\Users\A016512\Desktop\Learn-Stanford-CS336\homework\assignment1_basics\tests\_snapshots\test_adamw.npz")
# print(a["array"].shape)
import torch
from torch import optim
import math
from collections.abc import Callable, Iterable
from typing import Optional
class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.99), weight_decay=0.01, eps=1e-8):
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
                a_t = lr* math.sqrt(1 - beta2 ** 2) / (1 - beta1 ** t)
                p.data -= a_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss






torch.manual_seed(42)
model = torch.nn.Linear(3, 2, bias=False)
opt = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
)
# Use 1000 optimization steps for testing
for _ in range(1000):
    opt.zero_grad()
    x = torch.rand(model.in_features)
    y_hat = model(x)
    y = torch.tensor([x[0] + x[1], -x[2]])
    loss = ((y - y_hat) ** 2).sum()
    loss.backward()
    opt.step()