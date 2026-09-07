import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import FluxKontextPipeline

from diffusers.image_processor import (VaeImageProcessor)
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from torchvision.transforms.functional import pad
from .transformer_flux import FluxTransformer2DModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def prepare_latent_image_ids_2(height, width, device, dtype):
    latent_image_ids = torch.zeros(height//2, width//2, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height//2, device=device)[:, None]  # y坐标
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width//2, device=device)[None, :]   # x坐标
    return latent_image_ids

def prepare_latent_subject_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2, device=device)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2, device=device)[None, :]
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    return latent_image_ids.to(device=device, dtype=dtype)

def resize_position_encoding(batch_size, original_height, original_width, target_height, target_width, device, dtype):
    latent_image_ids = prepare_latent_image_ids_2(original_height, original_width, device, dtype)
    scale_h = original_height / target_height
    scale_w = original_width / target_width
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    #spatial进行PE插值
    latent_image_ids_resized = torch.zeros(target_height//2, target_width//2, 3, device=device, dtype=dtype)
    for i in range(target_height//2):
        for j in range(target_width//2):
            latent_image_ids_resized[i, j, 1] = i*scale_h
            latent_image_ids_resized[i, j, 2] = j*scale_w
    cond_latent_image_id_height, cond_latent_image_id_width, cond_latent_image_id_channels = latent_image_ids_resized.shape
    cond_latent_image_ids = latent_image_ids_resized.reshape(
            cond_latent_image_id_height * cond_latent_image_id_width, cond_latent_image_id_channels
        )
    # latent_image_ids_ = torch.concat([latent_image_ids, cond_latent_image_ids], dim=0)
    return latent_image_ids, cond_latent_image_ids #, latent_image_ids_
    
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
        encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_t5_prompt_embeds
def _get_t5_prompt_embeds_input_ids(
    self,
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    ret_input_ids = False,
):
    device = device or self._execution_device
    dtype = dtype or self.text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if isinstance(self, TextualInversionLoaderMixin):
        prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

    text_inputs = self.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

    dtype = self.text_encoder_2.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    if ret_input_ids:
        return prompt_embeds, text_input_ids
    return prompt_embeds
    
# Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.encode_prompt
def encode_prompt_input_ids(
    self,
    prompt: Union[str, List[str]],
    prompt_2: Optional[Union[str, List[str]]] = None,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    max_sequence_length: int = 512,
    lora_scale: Optional[float] = None,
    ret_input_ids=False,
):
    device = device or self._execution_device

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
        self._lora_scale = lora_scale

        # dynamically adjust the LoRA scale
        if self.text_encoder is not None and USE_PEFT_BACKEND:
            scale_lora_layers(self.text_encoder, lora_scale)
        if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
            scale_lora_layers(self.text_encoder_2, lora_scale)

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # We only use the pooled prompt output from the CLIPTextModel
        pooled_prompt_embeds = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        prompt_embeds, input_ids = _get_t5_prompt_embeds_input_ids(
            self,
            prompt=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            ret_input_ids=True
        )

    if self.text_encoder is not None:
        if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

    if self.text_encoder_2 is not None:
        if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder_2, lora_scale)

    dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    if ret_input_ids:
        input_ids = input_ids.to(device=device, dtype=dtype)
        return prompt_embeds, pooled_prompt_embeds, text_ids, input_ids

    return prompt_embeds, pooled_prompt_embeds, text_ids 



# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def set_moe_layers_latents(
        subject_images,
        sty_encoder, 
        siglip_processor,
        siglip_model,
        moe_layers = None,
    ):
    with torch.no_grad():
        inputs = siglip_processor(images=subject_images, return_tensors="pt").to(siglip_model.device)
        siglip_feats = siglip_model(**inputs, output_hidden_states=True)
    # style_feats = siglip_model(**inputs).pooler_output

    latents = sty_encoder(siglip_feats).flatten(1)
    
    cond_hidden_states = latents
    for layer in moe_layers:
        layer.set_latents(cond_hidden_states=cond_hidden_states)

def insert_style_tokens(
    prompt_embeds, 
    sty_token_id, con_token_id, sty_ori_token_id, 
    sty_tokens,
    text_input_ids, text_ids
    ):
    def insert_tokens(prompt_embed: torch.Tensor, sty_token: torch.Tensor, index: int) -> torch.Tensor:
        if sty_token.dim() == 1:  # (hidden_dim,)
            sty_token = sty_token.unsqueeze(0)  # (1, hidden_dim)
        if sty_token.dim() == 2:  # (1, hidden_dim)
            sty_token = sty_token.unsqueeze(0)  # (1, 1, hidden_dim)

        before = prompt_embed[:, :index, :]
        after = prompt_embed[:, index:, :]
        new_prompt_embed = torch.cat([before, sty_token, after], dim=1)
        return new_prompt_embed

    new_prompt_embeds = []
    for i in range(len(prompt_embeds)):
        input_ids = text_input_ids[i]
        sty_token_index = -1
        for index, token_id in enumerate(input_ids.tolist()):
            if token_id == sty_token_id:
                sty_token_index = index
                break
        prompt_embed = prompt_embeds[i]
        prompt_embed = prompt_embed.unsqueeze(0)
        prompt_embed = insert_tokens(prompt_embed, sty_tokens, sty_token_index)
        # sty_token_mask = [True if sty_token_index <= i < sty_token_index+1 else False for i in range(prompt_embeds.shape[1])]
        # sty_token_mask = torch.tensor(sty_token_mask, dtype=torch.bool).unsqueeze(0).to(accelerator.device)
        # updated_embed = photo_encoder(cond_A_pixel_value, prompt_embed, sty_token_mask)
        new_prompt_embeds.append(prompt_embed)
    prompt_embeds = torch.cat(new_prompt_embeds, dim=0)
    style_len = sty_tokens.shape[1]
    text_ids = torch.cat([text_ids, torch.zeros(style_len, 3, device=text_ids.device)])
    return prompt_embeds, text_ids

from .moe import param_CondLoRAMoELayer
class myKontextPipeline(FluxKontextPipeline):
    def prepare_latents(

            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            subject_image,
            condition_image,
            latents=None,
            cond_number=1,
            sub_number=1,
    ):
        height_cond = 2 * (self.cond_size // (self.vae_scale_factor*2))
        width_cond = 2 * (self.cond_size // (self.vae_scale_factor*2))
        height = 2 * (int(height) // (self.vae_scale_factor*2))
        width = 2 * (int(width) // (self.vae_scale_factor*2))

        shape = (batch_size, num_channels_latents, height, width)  # 1 16 106 80
        noise_latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  
        noise_latent_image_ids = self._prepare_latent_image_ids(
            noise_latents.shape[0],
            noise_latents.shape[2] // 2,
            noise_latents.shape[3] // 2,
            device,
            dtype,
        )
        noise_latents = self._pack_latents(noise_latents, batch_size, num_channels_latents, height, width)
        
        latents_to_concat = []  # 不包含 latents
        latents_ids_to_concat = [noise_latent_image_ids]
        
                # spatial
        if condition_image is not None:
            cond_number = 1

            B, N, C, H, W = condition_image.shape  # 1, 3, 3, 512, 512
            condition_image = condition_image.view(B * N, C, H, W).to(dtype=dtype)

            condition_image = condition_image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=condition_image, generator=generator)
            cond_latent_image_ids = self._prepare_latent_image_ids(
                image_latents.shape[0],
                image_latents.shape[2] // 2,
                image_latents.shape[3] // 2,
                device,
                dtype,
            )
            cond_latents = self._pack_latents(image_latents, B*N, num_channels_latents, height_cond*cond_number, width_cond)

            # cond_latents = self.con_encoder(cond_latents)  # 新增
            cond_latents = cond_latents.view(B, N, *cond_latents.shape[1:])
            cond_latents = cond_latents.mean(dim=1)
            # print("In pipeline, through con_encoder")

            cond_latent_image_ids = torch.concat([cond_latent_image_ids for _ in range(cond_number)], dim=-2)
            cond_latent_image_ids[..., 0] = 1
            latents_ids_to_concat.append(cond_latent_image_ids)
            latents_to_concat.append(cond_latents)

        # subject
        if subject_image is not None and getattr(self, "style_token_concat", True):
            sub_number = 1

            B, N, C, H, W = subject_image.shape  # 1, 3, 3, 512, 512
            subject_image = subject_image.view(B * N, C, H, W).to(dtype=dtype)

            subject_image = subject_image.to(device=device, dtype=dtype)
            subject_image_latents = self._encode_vae_image(image=subject_image, generator=generator)
            if getattr(self, "inference_args", None):
                style_multi = self.inference_args.style_multi if self.inference_args.style_multi else 1
                subject_image_latents = subject_image_latents * style_multi
            
            latent_subject_ids = self._prepare_latent_image_ids(
                subject_image_latents.shape[0],
                subject_image_latents.shape[2] // 2,
                subject_image_latents.shape[3] // 2,
                device,
                dtype,
            )
            image_latent_height, image_latent_width = subject_image_latents.shape[2:]
            subject_latents = self._pack_latents(subject_image_latents, B*N, num_channels_latents, image_latent_height*sub_number, image_latent_width)

            # subject_latents = self.sty_encoder(subject_latents)  # 新增
            subject_latents = subject_latents.view(B, N, *subject_latents.shape[1:])
            subject_latents = subject_latents.mean(dim=1)
            # print("In pipeline, through sty_encoder")

            # latent_subject_ids = prepare_latent_subject_ids(height_cond, width_cond, device, dtype)
            if hasattr(self, "style_offset") and self.style_offset:
                latent_subject_ids[:, 1] += 64

            latent_subject_ids[..., 0] = 2
            subject_latent_image_ids = torch.concat([latent_subject_ids for _ in range(sub_number)], dim=-2)
            latents_to_concat.append(subject_latents)
            latents_ids_to_concat.append(subject_latent_image_ids)
            
        cond_latents = torch.concat(latents_to_concat, dim=1)
        latent_image_ids = torch.concat(latents_ids_to_concat, dim=0)
        return cond_latents, latent_image_ids, noise_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
        spatial_images=None,
        subject_images=None,
        cond_size=1024,
    ):
        self.cond_size = cond_size
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_height, original_width = height, width
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        if height != original_height or width != original_width:
            logger.warning(
                f"Generation `height` and `width` have been adjusted to {height} and {width} to fit the model requirements."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        cond_number = len(spatial_images) if spatial_images else 0
        sub_number = len(subject_images) if subject_images else 0
        
        def process_image(image):
            img = image[0] if isinstance(image, list) else image
            image_height, image_width = self.image_processor.get_default_height_width(img)
            aspect_ratio = image_width / image_height
            if _auto_resize:
                # Kontext is trained on specific resolutions, using one of them is recommended
                _, image_width, image_height = min(
                    (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                )
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width)
            return image

        if sub_number > 0:
            subject_image_ls = []
            for subject_image in subject_images:
                subject_image_ls.append(process_image(subject_image))
            subject_image = torch.stack(subject_image_ls, dim=1)
        else:
            subject_image = None
        
        if cond_number > 0:
            condition_image_ls = []
            for img in spatial_images:
                # condition_image = self.image_processor.preprocess(img, height=self.cond_size, width=self.cond_size)
                # condition_image = condition_image.to(dtype=torch.float32)
                condition_image_ls.append(process_image(img))
            condition_image = torch.stack(condition_image_ls, dim=1)
        else:
            condition_image = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        # latents, image_latents, latent_ids, image_ids = self.prepare_latents(
            # image,
            # batch_size * num_images_per_prompt,
            # num_channels_latents,
            # height,
            # width,
            # prompt_embeds.dtype,
            # device,
            # generator,
            # latents,
        # )
        cond_latents, latent_ids, latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            subject_image,
            condition_image,
            latents,
            cond_number,
            sub_number
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )


        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                latent_model_input = torch.cat([latents, cond_latents], dim=1)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents.to(dtype=self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

class MoEKontextPipeline(myKontextPipeline):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"
    _optional_components = [
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder = None,
        feature_extractor = None,
        # more
        extra_modules = None,
        extra_items = None
        # siglip_processor=None,
        # siglip_model=None,
        # sty_encoder=None,
        # sty_token_encoder=None,
        # con_token_id=None,
        # sty_token_id=None,
        # sty_ori_token_id=None,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            image_encoder = image_encoder,
            feature_extractor = feature_extractor,
        )
        self.sty_encoder = extra_modules.sty_encoder
        self.sty_token_encoder = extra_modules.get_module("sty_token_encoder")

        self.siglip_processor = extra_items.siglip_processor
        self.siglip_model = extra_items.siglip_model
        self.con_token_id = extra_items.con_token_id
        self.sty_token_id = extra_items.sty_token_id
        self.sty_ori_token_id = extra_items.sty_ori_token_id

        self.style_token_concat = extra_items.style_token_concat or False
        self.style_offset = extra_items.style_offset

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
        spatial_images=None,
        subject_images=None,
        cond_size=1024,
        get_topk_indices=False,
    ):
        self.cond_size = cond_size
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_height, original_width = height, width
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        if height != original_height or width != original_width:
            logger.warning(
                f"Generation `height` and `width` have been adjusted to {height} and {width} to fit the model requirements."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        cond_number = len(spatial_images) if spatial_images else 0
        sub_number = len(subject_images) if subject_images else 0
        
        def process_image(image):
            img = image[0] if isinstance(image, list) else image
            image_height, image_width = self.image_processor.get_default_height_width(img)
            aspect_ratio = image_width / image_height
            if _auto_resize:
                # Kontext is trained on specific resolutions, using one of them is recommended
                _, image_width, image_height = min(
                    (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                )
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width)
            return image

        if sub_number > 0:
            subject_image_ls = []
            for subject_image in subject_images:
                subject_image_ls.append(process_image(subject_image))
            subject_image = torch.stack(subject_image_ls, dim=1)
        else:
            subject_image = None
        
        if cond_number > 0:
            condition_image_ls = []
            for img in spatial_images:
                # condition_image = self.image_processor.preprocess(img, height=self.cond_size, width=self.cond_size)
                # condition_image = condition_image.to(dtype=torch.float32)
                condition_image_ls.append(process_image(img))
            condition_image = torch.stack(condition_image_ls, dim=1)
        else:
            condition_image = None

        moe_layers = [
            module for name, module in self.transformer.named_modules()
                if isinstance(module, param_CondLoRAMoELayer)
        ]
        
        if sub_number > 0 and len(moe_layers) > 0: # 暂时先1个
            set_moe_layers_latents(
                subject_images[0],
                self.sty_encoder,
                self.siglip_processor,
                self.siglip_model,
                moe_layers,
            )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            input_ids,
        ) = encode_prompt_input_ids(
            self,
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
            ret_input_ids=True
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        if sub_number > 0 and self.sty_token_encoder: # 暂时先1个
            inputs = self.siglip_processor(images=subject_images[0], return_tensors="pt").to(self.siglip_model.device)

            with torch.no_grad():
                style_feats = self.siglip_model(**inputs, output_hidden_states=True)

            sty_tokens = self.sty_token_encoder(style_feats).to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)

            prompt_embeds, text_ids = insert_style_tokens(
                prompt_embeds, 
                self.sty_token_id, self.con_token_id, self.sty_ori_token_id, 
                sty_tokens,
                input_ids, text_ids
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        # latents, image_latents, latent_ids, image_ids = self.prepare_latents(
            # image,
            # batch_size * num_images_per_prompt,
            # num_channels_latents,
            # height,
            # width,
            # prompt_embeds.dtype,
            # device,
            # generator,
            # latents,
        # )
        cond_latents, latent_ids, latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            subject_image,
            condition_image,
            latents,
            cond_number,
            sub_number
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )


        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                latent_model_input = torch.cat([latents, cond_latents], dim=1)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                if get_topk_indices:
                    topk_indices = []
                    for layer in moe_layers:
                        topk_indices.append(layer.top_k_idx)
                    return topk_indices
                noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents.to(dtype=self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        
        for layer in moe_layers:
            layer.clear_latents()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
