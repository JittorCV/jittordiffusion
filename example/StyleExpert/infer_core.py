import os
import sys
import random
import numpy as np
from types import SimpleNamespace
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Configure the backend before any module can import ``torch``.
from jittor_runtime import jt, torch, seed_everything

from myutils.config_tool import load_config, dict_to_namespace
from myutils.extra_objects import ExtraModules, ExtraItems

from PIL import Image
import pandas as pd
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from transformers import SiglipVisionModel, SiglipImageProcessor
from diffusers import AutoencoderKL

from src.pipeline import MoEKontextPipeline
from src.ori_transformer_flux import FluxTransformer2DModel
from src.siglip_layers import SigLIPMultiFeatProjModel
from src.moe import LoRACompatibleLinear, param_CondLoRAMoELayer
from src.lora_helper import load_checkpoint

def _load_cfg(config_path: str):
    cfg_path = config_path
    if not os.path.isabs(cfg_path):
        cfg_path = str((BASE_DIR / cfg_path).resolve())
    try:
        return dict_to_namespace(load_config(cfg_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load config {cfg_path}: {exc}") from exc


def _get_cfg_attr(cfg, name: str, default=None):
    if cfg is None:
        return default
    value = getattr(cfg, name, default)
    return default if value is None else value


def _resolve_pretrained_path(arch: str) -> str:
    if arch == "flux_kontext_dev":
        return str(BASE_DIR / "models" / "FLUX.1-Kontext-dev")
    if arch == "flux_dev":
        return str(BASE_DIR / "models" / "FLUX.1-dev")
    raise ValueError(f"Unsupported arch: {arch}")


def _import_model_class(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    if model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    raise ValueError(f"{model_class} is not supported.")


def _read_base_keys(csv_path: str):
    if not os.path.isabs(csv_path):
        csv_path = str((BASE_DIR / csv_path).resolve())
    df = pd.read_csv(csv_path)
    return df["base_key"].dropna().tolist()


def _create_and_replace_layers(transformer, module_names, moe_cfg):
    moe_layers = []
    checkpoint = load_checkpoint(moe_cfg.moe_layers_pretrained_path) if moe_cfg.moe_layers_pretrained_path else None

    def group_lora_layers(state):
        grouped = {}
        for k, v in state.items():
            if ".lora_layer." in k:
                prefix, suffix = k.split(".lora_layer.", 1)
                grouped.setdefault(prefix, {})[suffix] = v
        return grouped

    checkpoint = group_lora_layers(checkpoint) if checkpoint else None

    for name in module_names:
        parent_module = transformer
        name = ".".join(name.split(".")[1:])

        def get_next(current_module, n: str):
            if n.isdigit():
                return current_module[int(n)]
            return getattr(current_module, n)

        def set_next(current_module, n: str, value):
            if n.isdigit():
                current_module[int(n)] = value
            else:
                setattr(current_module, n, value)

        names = name.split(".")
        for n in names[:-1]:
            parent_module = get_next(parent_module, n)
        last_module = get_next(parent_module, names[-1])

        kwargs = {
            "cond_dim": moe_cfg.cond_dim,
            "num_experts": moe_cfg.num_experts,
            "rank": moe_cfg.moe_rank,
            "network_alpha": moe_cfg.moe_rank,
            "top_k": moe_cfg.top_k,
        }

        def get_compatible(layer):
            new_layer = LoRACompatibleLinear(
                in_features=layer.in_features,
                out_features=layer.out_features,
                bias=layer.bias is not None,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone().detach()
            new_layer.weight.data = layer.weight.data.clone().detach()
            return new_layer

        set_next(parent_module, names[-1], get_compatible(last_module))
        last_module = get_next(parent_module, names[-1])

        moe_layer = param_CondLoRAMoELayer(
            in_features=last_module.in_features,
            out_features=last_module.out_features,
            device=last_module.weight.device,
            dtype=last_module.weight.dtype,
            **kwargs,
        )

        if checkpoint and name in checkpoint:
            layer_dict = checkpoint[name]
            for k, v in layer_dict.items():
                sub_module = getattr(moe_layer, k.split(".")[0])
                if "." in k:
                    param_name = k.split(".")[1]
                    getattr(sub_module, param_name).data.copy_(v)
                else:
                    sub_module.data.copy_(v)

        moe_layers.append(moe_layer)
        last_module.set_lora_layer(moe_layer)
    return moe_layers


def _build_pipeline(config_path: str):
    cfg = _load_cfg(config_path)

    pretrained_path = _resolve_pretrained_path("flux_kontext_dev")
    revision = None
    variant = None

    cond_size = 1024
    height = 1024
    width = 1024
    num_steps = 28
    guidance = 3.5
    max_seq = 128
    prompt_default = _get_cfg_attr(cfg, "validation_prompt", "")

    # Jittor/JTorch has broader FLUX operator coverage in fp16 than bf16.
    device = torch.device("cuda" if jt.flags.use_cuda else "cpu")
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer_one = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer", revision=revision)
    tokenizer_two = T5TokenizerFast.from_pretrained(pretrained_path, subfolder="tokenizer_2", revision=revision)


    text_encoder_cls_one = _import_model_class(pretrained_path, revision)
    text_encoder_cls_two = _import_model_class(pretrained_path, revision, subfolder="text_encoder_2")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        pretrained_path,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        pretrained_path,
        subfolder="text_encoder_2",
        revision=revision,
        variant=variant,
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_path,
        subfolder="vae",
        revision=revision,
        variant=variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_path,
        subfolder="transformer",
        revision=revision,
        variant=variant,
    )

    extra_modules = ExtraModules()
    extra_items = ExtraItems()

    # SigLIP
    siglip_path = str(BASE_DIR / "models" / "siglip-so400m-patch14-384")
    siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
    siglip_model = SiglipVisionModel.from_pretrained(siglip_path).to(device)
    siglip_model.eval()
    extra_items.add_items(siglip_processor=siglip_processor, siglip_model=siglip_model)

    # MoE config
    moe_cfg = _get_cfg_attr(cfg, "moe_config", None)
    if moe_cfg is None:
        moe_cfg = SimpleNamespace()
    moe_cfg.cond_dim = getattr(moe_cfg, "cond_dim", 3072)
    moe_cfg.num_experts = getattr(moe_cfg, "num_experts", 16)
    moe_cfg.moe_rank = getattr(moe_cfg, "moe_rank", 8)
    moe_cfg.top_k = getattr(moe_cfg, "top_k", 2)
    moe_cfg.moe_layers_pretrained_path = getattr(moe_cfg, "moe_layers_pretrained_path", None)
    moe_cfg.train_modules_csv = getattr(moe_cfg, "train_modules_csv", None)
    moe_cfg.sty_encoder_pretrained_path = getattr(moe_cfg, "sty_encoder_pretrained_path", None)
    if moe_cfg.moe_layers_pretrained_path and not os.path.isabs(moe_cfg.moe_layers_pretrained_path):
        moe_cfg.moe_layers_pretrained_path = str((BASE_DIR / moe_cfg.moe_layers_pretrained_path).resolve())
    if moe_cfg.train_modules_csv and not os.path.isabs(moe_cfg.train_modules_csv):
        moe_cfg.train_modules_csv = str((BASE_DIR / moe_cfg.train_modules_csv).resolve())
    if moe_cfg.sty_encoder_pretrained_path and not os.path.isabs(moe_cfg.sty_encoder_pretrained_path):
        moe_cfg.sty_encoder_pretrained_path = str((BASE_DIR / moe_cfg.sty_encoder_pretrained_path).resolve())

    encoder_kwargs = {
        "layer_indices": [-2, -11, -20],
        "siglip_token_nums": 729,
        "style_token_nums": 8,
        "siglip_token_dims": 1152,
        "hidden_size": 128,
        "context_layer_norm": True,
    }
    sty_encoder = SigLIPMultiFeatProjModel(**encoder_kwargs)
    extra_modules.add_modules(sty_encoder=sty_encoder)
    if moe_cfg.sty_encoder_pretrained_path:
        extra_modules.sty_encoder.load_proj_model(moe_cfg.sty_encoder_pretrained_path)

    # Style token concat (optional).
    style_token_cfg = getattr(moe_cfg, "style_token_config", None)
    if style_token_cfg:
        extra_items.add_items(style_token_concat=True)

    extra_items.add_items(style_offset=_get_cfg_attr(cfg, "style_offset", True))
    # ``_native_flash`` is a PyTorch/Diffusers backend selector.  Jittor picks
    # its own implementation; jittor_runtime also provides an SDPA fallback.

    # Dtype/device.
    vae.to(device, dtype=weight_dtype)
    transformer.to(device, dtype=weight_dtype)
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)
    extra_modules.to(device, dtype=weight_dtype)

    # MoE LoRA layers.
    if moe_cfg.moe_layers_pretrained_path:
        if not moe_cfg.train_modules_csv:
            raise ValueError("moe_config.train_modules_csv is required to load MoE layers.")
        module_names = _read_base_keys(moe_cfg.train_modules_csv)
        _create_and_replace_layers(transformer, module_names, moe_cfg)

    pipeline = MoEKontextPipeline.from_pretrained(
        pretrained_path,
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        transformer=transformer,
        revision=revision,
        variant=variant,
        torch_dtype=weight_dtype,
        extra_modules=extra_modules,
        extra_items=extra_items,
    ).to(device)

    pipeline.set_progress_bar_config(disable=True)

    defaults = SimpleNamespace(
        prompt=prompt_default,
        height=height,
        width=width,
        cond_size=cond_size,
        num_steps=num_steps,
        guidance=guidance,
        max_seq=max_seq,
    )
    return pipeline, device, defaults


_PIPELINE_CACHE = {}


def _get_pipeline(config_path: str):
    config_key = str((BASE_DIR / config_path).resolve()) if not os.path.isabs(config_path) else config_path
    if config_key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[config_key] = _build_pipeline(config_key)
    return _PIPELINE_CACHE[config_key]


def run_inference_with_bundle(
    pipeline,
    device,
    defaults,
    content_image: Image.Image,
    style_image: Image.Image,
    generator=None,
    prompt: str | None = None,
) -> Image.Image:
    prompt = prompt if prompt is not None else (defaults.prompt or "")

    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            height=defaults.height,
            width=defaults.width,
            num_inference_steps=defaults.num_steps,
            guidance_scale=defaults.guidance,
            max_sequence_length=defaults.max_seq,
            spatial_images=[content_image],
            subject_images=[style_image],
            cond_size=defaults.cond_size,
            generator=generator,
        )

    return result.images[0]


def get_pipeline_bundle(config_path: str):
    return _get_pipeline(config_path)


def inference(content_path: str, style_path: str, config_path: str, seed: int = 42, prompt: str | None = None) -> Image.Image:
    # Align global RNG state with training/inference script behavior (`set_seed(args.seed, deterministic=True)`).
    # This matters because some modules (e.g. text encoder dropout in train mode) may consume global RNG.
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        seed_everything(seed)

    pipeline, device, defaults = _get_pipeline(config_path)
    content_image = Image.open(content_path).convert("RGB")
    style_image = Image.open(style_path).convert("RGB")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    return run_inference_with_bundle(
        pipeline,
        device,
        defaults,
        content_image,
        style_image,
        generator=generator,
        prompt=prompt,
    )
