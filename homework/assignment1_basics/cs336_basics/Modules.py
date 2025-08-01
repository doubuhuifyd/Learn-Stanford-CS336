__all__ = ["Linear", "Embedding", "RMSNorm", "SwigLU", "RotaryPositionalEmbedding",
           "ScaledDotProductAttention", "CausalMultiHeadAttention", "TransformerBlock"]

import torch
from torch import nn, Tensor
from einops import rearrange,einsum
from jaxtyping import Float, Int

from  cs336_basics.utils import softmax

class Linear(nn.Module):
    """Simple linear layer"""
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.nn.init.trunc_normal_(torch.Tensor(out_features, in_features),
                                                               mean=0,
                                                               std=2/(in_features + out_features),
                                                               a=-6/(in_features + out_features),
                                                               b=6/(in_features + out_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : column vector """
        ans = einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
        return ans

class Embedding(nn.Module):
    """Simple embedding layer"""
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.nn.init.trunc_normal_(torch.Tensor(num_embeddings, embedding_dim),
                                                               mean=0,
                                                               std=1,
                                                               a=-3,
                                                               b=3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids : column vector """
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight  = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : column vector"""
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        RMS = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) / self.d_model + self.eps)
        output = x / RMS * self.weight.to(dtype=torch.float32)
        return output.to(in_dtype)

class SwigLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        super().__init__()
        self.weight1 = nn.Parameter(torch.ones(d_model, d_ff))
        self.weight2 = nn.Parameter(torch.ones(d_ff, d_model))
        self.weight3 = nn.Parameter(torch.ones(d_model, d_ff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(self.weight1.data, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3_x = einsum(self.weight3.data, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu_w1_x = w1_x * torch.sigmoid(w1_x)

        return einsum(self.weight2, silu_w1_x * w3_x, "d_model d_ff, ... d_ff -> ... d_model")

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k:int, max_seq_len:int, device=None) -> None:
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        k = torch.arange(0, d_k, 2, device=device)
        i = torch.arange(max_seq_len, device=device)
        angles = 1. / theta ** (k / d_k)
        angles = torch.outer(i, angles)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        self.register_buffer('sin', sin, persistent=False)
        self.register_buffer('cos', cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        output = torch.empty_like(x)
        output[..., 0::2] = x_even * self.cos[token_positions] - x_odd * self.sin[token_positions]
        output[..., 1::2] = x_odd * self.cos[token_positions] + x_even * self.sin[token_positions]
        return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                Q: Float[Tensor, " ... queries d_k"],
                K: Float[Tensor, " ... keys d_k"],
                V: Float[Tensor, " ... values d_v"],
                mask: Float[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        """
        Given key (K), query (Q), and value (V) tensors, return
        the output of your scaled dot product attention implementation.

        Args:
            Q (Float[Tensor, " ... queries d_k"]): Query tensor
            K (Float[Tensor, " ... keys d_k"]): Key tensor
            V (Float[Tensor, " ... values d_v"]): Values tensor
            mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
        Returns:
            Float[Tensor, " ... queries d_v"]: Output of SDPA
        """
        d_k = Q.shape[-1]
        qt_k = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        weight = qt_k / torch.sqrt(torch.tensor(d_k, device=Q.device))
        if mask is not None:
            weight = torch.where(mask, weight, -torch.inf)
        output = einsum(softmax(weight, -1), V, "... queries keys, ... keys d_v -> ... queries d_v")
        return output

class CausalMultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 max_seq_len:int,
                 theta:float = 10000.0,
                 use_rope:bool = True,
                 device=None,
                 dtype=None,
                 ):
        super().__init__()
        self.attention = ScaledDotProductAttention()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.d_k = d_model // num_heads
        # factory_kwargs = {'device': device, 'dtype': dtype}
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = [Linear(d_model, d_model)
                                                              for _ in range(4)]
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(
            self,
            x: Float[Tensor, "... seq_len d_model"],
            token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        q, k, v = [rearrange(proj(x), "b s (h d) -> b h s d", h=self.num_heads)
                   for proj in [self.q_proj, self.k_proj, self.v_proj]]
        if self.use_rope:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)

        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        output = self.attention(q, k, v, mask=mask)
        output = rearrange(output, "b h s d -> b s (h d)")
        output = self.o_proj(output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 d_ff:int,
                 max_seq_len:int,
                 theta:float = 10000.0,
                 use_rope:bool = True,
                 device=None,
                 dtype=None,
                 ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.causal_multi_head_attention = CausalMultiHeadAttention(d_model, num_heads, max_seq_len, theta, use_rope=use_rope)
        self.feed_forward = SwigLU(d_model, d_ff)

    def forward(self,
                x:torch.Tensor,
                token_positions: torch.Tensor | None = None
                ) -> torch.Tensor:
        x = x + self.causal_multi_head_attention(self.norm1(x), token_positions=token_positions)
        y = x + self.feed_forward(self.norm2(x))
        return y