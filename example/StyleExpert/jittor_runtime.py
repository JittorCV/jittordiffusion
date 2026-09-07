"""Jittor/JTorch runtime setup shared by the inference entry points."""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Dict


def configure_jittor() -> tuple:
    """Select the Jittor device and verify that ``torch`` is the JTorch shim."""
    force_cpu = os.getenv("STYLEEXPERT_FORCE_CPU", "0").lower() in {"1", "true", "yes"}
    if force_cpu:
        # Jittor checks this environment variable during import.  It prevents a
        # fresh Windows CPU smoke test from auto-downloading a CUDA toolkit.
        os.environ.setdefault("nvcc_path", "")
    try:
        import jittor as jt
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "StyleExpert-jittor requires Jittor and JTorch. "
            "Install the packages listed in requirements.txt first."
        ) from exc

    # JTorch intentionally exposes a package named ``torch``.  Running with real
    # PyTorch would look valid at import time while silently defeating the port.
    if torch.Tensor is not jt.Var:
        raise RuntimeError(
            "The imported 'torch' module is real PyTorch, not the JTorch shim. "
            "Use a clean environment and install Jittor/JTorch before the other "
            "dependencies; see README.md."
        )

    if force_cpu:
        # Do not even query jt.has_cuda here: on a fresh Windows install that
        # query may trigger Jittor's optional CUDA toolkit download.
        jt.flags.use_cuda = 0
    else:
        jt.flags.use_cuda = int(bool(getattr(jt, "has_cuda", False)))
    torch.cuda.is_available = lambda: bool(jt.flags.use_cuda)
    torch.cuda.device_count = lambda: int(bool(jt.flags.use_cuda))

    if not hasattr(torch, "is_grad_enabled"):
        # StyleExpert-jittor currently exposes inference only.
        torch.is_grad_enabled = lambda: False

    # Jittor's elementwise ``equal`` is different from torch.equal, which must
    # return one Python bool for the whole tensor.
    def tensors_equal(left, right):
        if tuple(left.shape) != tuple(right.shape):
            return False
        return bool((left == right).all().item())

    torch.equal = tensors_equal

    # Fill two small gaps in JTorch 0.2's Generator API that Diffusers queries.
    generator_cls = torch.Generator
    if not hasattr(generator_cls, "device"):
        generator_cls.device = property(
            lambda _self: torch.device("cuda" if jt.flags.use_cuda else "cpu")
        )
    original_manual_seed = generator_cls.manual_seed
    if not getattr(original_manual_seed, "_styleexpert_returns_self", False):
        def manual_seed(self, seed):
            original_manual_seed(self, seed)
            return self

        manual_seed._styleexpert_returns_self = True
        generator_cls.manual_seed = manual_seed

    # JTorch 0.2 does not expose PyTorch 2.x SDPA on every Jittor build, while
    # FLUX uses it heavily.  Register a numerically equivalent eager fallback.
    functional = torch.nn.functional
    if not hasattr(functional, "scaled_dot_product_attention"):
        def scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            **_kwargs,
        ):
            scale_factor = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
            scores = (query @ key.transpose(-2, -1)) * scale_factor
            if is_causal:
                q_pos = torch.arange(query.shape[-2], device=query.device)[:, None]
                k_pos = torch.arange(key.shape[-2], device=key.device)[None, :]
                scores = scores + (k_pos > q_pos).to(scores.dtype) * -1e4
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores = scores + (~attn_mask).to(scores.dtype) * -1e4
                else:
                    scores = scores + attn_mask.to(scores.dtype)
            probs = functional.softmax(scores, dim=-1)
            if dropout_p:
                probs = functional.dropout(probs, p=dropout_p, training=True)
            return probs @ value

        functional.scaled_dot_product_attention = scaled_dot_product_attention

    return jt, torch


jt, torch = configure_jittor()


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    try:
        from jittor.misc import set_global_seed

        set_global_seed(seed)
    except (ImportError, AttributeError):
        jt.set_global_seed(seed)


def load_safetensors(path: str | os.PathLike) -> Dict[str, object]:
    """Load safetensors after the JTorch shim has registered as ``torch``."""
    from safetensors import safe_open

    tensors = {}
    with safe_open(str(Path(path)), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors
