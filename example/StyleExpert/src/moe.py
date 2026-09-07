import inspect
import math
from typing import Callable, List, Optional, Tuple, Union
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from diffusers.models.attention_processor import Attention


class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer= None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer):
        self.lora_layer = lora_layer

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.lora_layer is None:
            out = super().forward(hidden_states)
            return out
        else:
            out = super().forward(hidden_states) + (scale * self.lora_layer(hidden_states))
            return out

class param_CondLoRAMoELayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        cond_dim: int,
        num_experts: int = 4,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        top_k: int = 1,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        use_shared_expert: bool = True,  # New argument for shared expert
        shared_expert_rank: int = None,
    ):
        super().__init__()
        self.rank = rank
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_lora_scale = 16 // rank
        self.device = device
        self.dtype = dtype
        self.use_shared_expert = use_shared_expert  # Store whether to use shared expert
        # num_experts -= int(use_shared_expert)

        # Directly split expert into A and B
        self.loraA = nn.Parameter(
            torch.zeros(num_experts, rank, in_features, device=device, dtype=dtype)
        )
        self.loraB = nn.Parameter(
            torch.zeros(num_experts, out_features, rank, device=device, dtype=dtype)
        )

        # Shared expert parameters (if enabled)
        if self.use_shared_expert:
            rank = shared_expert_rank if shared_expert_rank else rank
            self.shared_A = nn.Parameter(
                torch.zeros(1, rank, in_features, device=device, dtype=dtype)
            )
            self.shared_B = nn.Parameter(
                torch.zeros(1, out_features, rank, device=device, dtype=dtype)
            )

        # Gating
        self.cond_gate = nn.Linear(cond_dim, num_experts, device=device, dtype=dtype, bias=False)

        self.uninit_expert_idx = 0

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.loraA.normal_(mean=0.0, std=1.0 / float(self.rank))
            self.loraB.zero_()

            # Initialize shared expert weights (if using shared expert)
            if self.use_shared_expert:
                self.shared_A.normal_(mean=0.0, std=1.0 / float(self.rank))
                self.shared_B.zero_()

    def set_latents(self, cond_hidden_states: torch.Tensor = None):
        self.cond_hidden_states = cond_hidden_states

    def clear_latents(self):
        self.cond_hidden_states = None

    def set_pretrained_expert_weights(self, svd_lora_weights, keep_rank=None):
        # Default keep_rank to self.rank if not provided
        if keep_rank is None:
            keep_rank = self.rank
        if self.uninit_expert_idx == self.num_experts:
            print("attn processor 已经初始化满了")
            return

        A_new, B_new = svd_lora_weights
        A_new, B_new = A_new[:keep_rank, :], B_new[:, :keep_rank]  # Use keep_rank for the slicing

        # Handle the case when keep_rank > self.rank
        num_splits = keep_rank // self.rank
        for i in range(num_splits):
            start_idx = i * self.rank
            end_idx = (i + 1) * self.rank
            self.loraA.data[self.uninit_expert_idx + i] = A_new[start_idx:end_idx, :].to(device=self.device, dtype=self.dtype)
            self.loraB.data[self.uninit_expert_idx + i] = B_new[:, start_idx:end_idx].to(device=self.device, dtype=self.dtype)
        self.uninit_expert_idx += num_splits

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # gating
        gate_logits = self.cond_gate(self.cond_hidden_states)  # [B, num_experts]
        # ====== Top-k before softmax ======
        if self.top_k == self.num_experts:
            topk_logits = gate_logits
            topk_idx = torch.arange(self.num_experts, device=self.device).expand(gate_logits.size(0), -1)
        else:
            topk_logits, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)  # [B, k]
        # softmax only on selected logits
        topk_scores = F.softmax(topk_logits, dim=-1)

        self.top_k_idx = topk_idx
        topk_scores *= self.norm_lora_scale

        # Ensure input shape is [B, T, D]
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)  # [B, 1, D]
            squeeze_back = True
        else:
            squeeze_back = False
        B, T, D = hidden_states.shape

        # Select top-k expert parameters
        A_selected = self.loraA[topk_idx]  # [B, k, r, D]
        B_selected = self.loraB[topk_idx]  # [B, k, out_features, r]

        # Include shared expert (if enabled)
        if self.use_shared_expert:
            A_shared = self.shared_A.expand(B, -1, -1, -1)  # [B, 1, r, D]
            B_shared = self.shared_B.expand(B, -1, -1, -1)  # [B, 1, out_features, r]
            A_selected = torch.cat([A_shared, A_selected], dim=1)  # [B, k+1, r, D]
            B_selected = torch.cat([B_shared, B_selected], dim=1)  # [B, k+1, out_features, r]
            topk_scores = F.pad(topk_scores, (0, 1), "constant", 0)  # Pad scores for shared expert

        # Replicate the input for top-k selection
        flat_in = hidden_states.unsqueeze(1).expand(-1, self.top_k + int(self.use_shared_expert), -1, -1)  # [B, k+1, T, D]

        # Calculate (x @ A^T) @ B^T
        inter = torch.einsum("bktd,bkrd->bktr", flat_in, A_selected)  # [B, k+1, T, r]
        expert_out = torch.einsum("bktr,bkor->bkto", inter, B_selected)  # [B, k+1, T, out_features]

        # Weighted sum
        outputs = torch.einsum("bkto,bk->bto", expert_out, topk_scores)

        if squeeze_back:
            outputs = outputs.squeeze(1)

        return outputs

# ---- 测试用例 ----
if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, D = 3, 5, 8   # batch=3, time=5, in_features=8
    out_features = 6
    cond_dim = 4
    num_experts = 4
    rank = 2
    top_k = 2

    layer = param_CondLoRAMoELayer(
        in_features=D,
        out_features=out_features,
        cond_dim=cond_dim,
        num_experts=num_experts,
        rank=rank,
        top_k=top_k,
        use_shared_expert=True,
    )

    hidden_states = torch.randn(B, T, D)          # [3, 5, 8]
    cond_hidden_states = torch.randn(B, cond_dim) # [3, 4]
    layer.set_latents(cond_hidden_states=cond_hidden_states)

    out = layer(hidden_states)
    print("Output shape:", out.shape)  # 期望 [3, 5, 6]
    print(out)
