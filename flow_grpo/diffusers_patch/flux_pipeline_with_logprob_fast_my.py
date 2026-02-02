from typing import Any, Dict, List, Optional, Union, Callable
import torch
import numpy as np
from .flux_pipeline_with_logprob_fast import pipeline_with_logprob, calculate_shift
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .sd3_sde_with_logprob import sde_step_with_logprob
import random

@torch.no_grad()
def pipeline_ode_sampling(
    pipeline,
    prompt_embeds,
    pooled_prompt_embeds,
    num_inference_steps=28,
    guidance_scale=3.5,
    height=1024,
    width=1024,
    generator=None,
    latents=None,
    output_type="pt",
    max_sequence_length=512,
):
    """
    Pure ODE sampling function (reusing existing pipeline logic but with noise_level=0)
    """
    # Simply call the existing pipeline with sde_window_size=0 (which means pure ODE)
    images, _, _, _, _, _ = pipeline_with_logprob(
        pipeline,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type=output_type,
        height=height,
        width=width,
        generator=generator,
        latents=latents,
        max_sequence_length=max_sequence_length,
        noise_level=0,  # Pure ODE
        sde_window_size=0,  # No SDE window
        sde_type='sde',
    )
    return images

# @torch.no_grad()
# def pipeline_mixed_sampling_with_intermediate_ode_original(
#     pipeline,
#     prompt_embeds,
#     pooled_prompt_embeds,
#     num_inference_steps=28,
#     guidance_scale=3.5,
#     height=1024,
#     width=1024,
#     generator=None,
#     latents=None,
#     output_type="pt",
#     max_sequence_length=512,
#     noise_level=0.7,
#     sde_window_size=3,
#     sde_window_range=(0, 5),
#     sde_type='sde',
# ):
#     """
#     Mixed SDE+ODE sampling with intermediate ODE completions
#     Each intermediate result provides the same structure as main results for GRPO training

#     Returns:
#         main_results: Main SDE+ODE sampling results (same structure as original)
#         intermediate_ode_results: List of {sde_steps_count, images, latents, log_probs, timesteps, image_ids, text_ids}
#     """
#     device = pipeline._execution_device
#     batch_size = prompt_embeds.shape[0]

#     # Setup similar to existing pipeline
#     num_channels_latents = pipeline.transformer.config.in_channels // 4
#     original_latents = latents
#     if latents is None:
#         latents, latent_image_ids = pipeline.prepare_latents(
#             batch_size,
#             num_channels_latents,
#             height,
#             width,
#             prompt_embeds.dtype,
#             device,
#             generator,
#             latents,
#         )
#     else:
#         # For consistent latent_image_ids generation - need to implement this properly
#         latent_image_ids = pipeline._prepare_latent_image_ids(
#             batch_size,
#             height // pipeline.vae_scale_factor,
#             width // pipeline.vae_scale_factor,
#             device,
#             latents.dtype
#         )

#     # Setup timesteps (reusing existing logic)
#     sigmas = None
#     image_seq_len = latents.shape[1]
#     mu = calculate_shift(
#         image_seq_len,
#         pipeline.scheduler.config.get("base_image_seq_len", 256),
#         pipeline.scheduler.config.get("max_image_seq_len", 4096),
#         pipeline.scheduler.config.get("base_shift", 0.5),
#         pipeline.scheduler.config.get("max_shift", 1.15),
#     )
#     timesteps, num_inference_steps = retrieve_timesteps(
#         pipeline.scheduler,
#         num_inference_steps,
#         device,
#         sigmas=sigmas,
#         mu=mu,
#     )

#     # Determine SDE window
#     if sde_window_size > 0:
#         import random
#         start = random.randint(sde_window_range[0], sde_window_range[1] - sde_window_size)
#         end = start + sde_window_size
#         sde_window = (start, end)
#     else:
#         sde_window = (0, len(timesteps)-1)

#     # Handle guidance
#     if pipeline.transformer.config.guidance_embeds:
#         guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
#         guidance = guidance.expand(latents.shape[0])
#     else:
#         guidance = None

#     # Encode text
#     text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=prompt_embeds.dtype)

#     # Storage for main results
#     main_all_latents = []
#     main_all_log_probs = []
#     main_all_timesteps = []

#     # Storage for intermediate results
#     intermediate_ode_results = []
#     sde_intermediate_latents = []  # Store latents after each SDE step

#     # Main denoising loop (same as original but with intermediate storage)
#     pipeline.scheduler.set_begin_index(0)
#     current_latents = latents.clone()

#     for i, t in enumerate(timesteps):
#         # Determine noise level for current step (same logic as original)
#         if i < sde_window[0]:
#             cur_noise_level = 0
#         elif i == sde_window[0]:
#             cur_noise_level = noise_level
#             main_all_latents.append(current_latents)
#         elif i > sde_window[0] and i < sde_window[1]:
#             cur_noise_level = noise_level
#         else:
#             cur_noise_level = 0

#         # Standard denoising step (same as original)
#         timestep = t.expand(current_latents.shape[0]).to(current_latents.dtype)
#         noise_pred = pipeline.transformer(
#             hidden_states=current_latents,
#             timestep=timestep / 1000,
#             guidance=guidance,
#             pooled_projections=pooled_prompt_embeds,
#             encoder_hidden_states=prompt_embeds,
#             txt_ids=text_ids,
#             img_ids=latent_image_ids,
#             return_dict=False,
#         )[0]

#         latents_dtype = current_latents.dtype
#         current_latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
#             pipeline.scheduler,
#             noise_pred.float(),
#             t.unsqueeze(0).repeat(current_latents.shape[0]),
#             current_latents.float(),
#             noise_level=cur_noise_level,
#             sde_type=sde_type,
#         )
#         if current_latents.dtype != latents_dtype:
#             current_latents = current_latents.to(latents_dtype)

#         # Store main SDE results (same as original)
#         if i >= sde_window[0] and i < sde_window[1]:
#             main_all_latents.append(current_latents)
#             main_all_log_probs.append(log_prob)
#             main_all_timesteps.append(t)

#             # Store intermediate latents for later ODE completion
#             sde_intermediate_latents.append({
#                 'step': i,
#                 'latents': current_latents.clone(),
#                 'timestep_index': i + 1  # Next timestep index for ODE continuation
#             })

#     # Generate intermediate ODE completions
#     for idx, sde_state in enumerate(sde_intermediate_latents):  # 原始：Exclude last (same as main logic) sde_intermediate_latents[:-1]
#         sde_steps_count = idx + 1  # Number of SDE steps completed
#         start_latents = sde_state['latents']
#         start_timestep_idx = sde_state['timestep_index']

#         # ODE completion from this intermediate state
#         remaining_timesteps = timesteps[start_timestep_idx:]
#         if len(remaining_timesteps) == 0:
#             continue

#         ode_latents = start_latents.clone()
#         ode_all_latents = [ode_latents.clone()]
#         ode_all_log_probs = []
#         ode_all_timesteps = []

#         # Reset scheduler for ODE completion
#         pipeline.scheduler.set_begin_index(0)

#         # ODE completion loop
#         for ode_i, ode_t in enumerate(remaining_timesteps):
#             ode_timestep = ode_t.expand(ode_latents.shape[0]).to(ode_latents.dtype)
#             # print(ode_latents.dtype, ode_timestep.dtype)
#             ode_noise_pred = pipeline.transformer(
#                 hidden_states=ode_latents,
#                 timestep=ode_timestep / 1000,
#                 guidance=guidance,
#                 pooled_projections=pooled_prompt_embeds,
#                 encoder_hidden_states=prompt_embeds,
#                 txt_ids=text_ids,
#                 img_ids=latent_image_ids,
#                 return_dict=False,
#             )[0]

#             # Pure ODE step (noise_level=0) but still compute log_prob for consistency
#             ode_latents, ode_log_prob, _, _ = sde_step_with_logprob(
#                 pipeline.scheduler,
#                 ode_noise_pred.float(),
#                 ode_t.unsqueeze(0).repeat(ode_latents.shape[0]),
#                 ode_latents.float(),
#                 noise_level=0,  # Pure ODE
#                 sde_type=sde_type,
#             )

#             if ode_latents.dtype != latents_dtype:
#                 ode_latents = ode_latents.to(latents_dtype)

#             ode_all_latents.append(ode_latents.clone())
#             ode_all_log_probs.append(ode_log_prob)
#             ode_all_timesteps.append(ode_t)

#         # Decode intermediate ODE result
#         ode_decoded_latents = pipeline._unpack_latents(ode_latents, height, width, pipeline.vae_scale_factor)
#         ode_decoded_latents = (ode_decoded_latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
#         ode_decoded_latents = ode_decoded_latents.to(dtype=pipeline.vae.dtype)
#         ode_image = pipeline.vae.decode(ode_decoded_latents, return_dict=False)[0]
#         ode_image = pipeline.image_processor.postprocess(ode_image, output_type=output_type)

#         # Store intermediate result with same structure as main result
#         intermediate_ode_results.append({
#             'sde_steps_count': sde_steps_count,
#             'images': ode_image,
#             'latents': ode_all_latents,  # Exclude last latent (same as main logic) [:-1]
#             'next_latents': ode_all_latents,  # Next latents (same as main logic) [1:]
#             'log_probs': ode_all_log_probs,
#             'timesteps': ode_all_timesteps,
#             'image_ids': latent_image_ids,
#             'text_ids': text_ids,
#             'specific_sde_start': sde_state['step']
#         })

#     # Final decode for main result
#     final_decoded_latents = pipeline._unpack_latents(current_latents, height, width, pipeline.vae_scale_factor)
#     final_decoded_latents = (final_decoded_latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
#     final_decoded_latents = final_decoded_latents.to(dtype=pipeline.vae.dtype)
#     final_image = pipeline.vae.decode(final_decoded_latents, return_dict=False)[0]
#     final_image = pipeline.image_processor.postprocess(final_image, output_type=output_type)

#     # Main results (same structure as original pipeline output)
#     main_results = {
#         'images': final_image,
#         'latents': main_all_latents[:-1],  # Exclude last latent (same as main logic)
#         'next_latents': main_all_latents[1:],  # Next latents (same as main logic)
#         'log_probs': main_all_log_probs,
#         'timesteps': main_all_timesteps,
#         'image_ids': latent_image_ids,
#         'text_ids': text_ids
#     }

#     return main_results, intermediate_ode_results



@torch.no_grad()
def pipeline_mixed_sampling_with_intermediate_ode(
    pipeline,

    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
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
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    noise_level: float = 0.7,
    sde_window_size: int = 0,
    sde_window_range: tuple[int, int] = (0, 5),
    sde_type: Optional[str] = 'sde',
):
    """
    Mixed SDE+ODE sampling with intermediate ODE completions
    Each intermediate result provides the same structure as main results for GRPO training

    Returns:
        main_results: Main SDE+ODE sampling results (same structure as original)
        intermediate_ode_results: List of {sde_steps_count, images, latents, log_probs, timesteps, image_ids, text_ids}
    """
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipeline.check_inputs(
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

    pipeline._guidance_scale = guidance_scale
    pipeline._joint_attention_kwargs = joint_attention_kwargs
    pipeline._current_timestep = None
    pipeline._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipeline._execution_device

    lora_scale = (
        pipeline.joint_attention_kwargs.get("scale", None) if pipeline.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )


    # 4. Prepare latent variables
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents, latent_image_ids = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(pipeline.scheduler.config, "use_flow_sigmas") and pipeline.scheduler.config.use_flow_sigmas:
        sigmas = None
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    # add
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)

    # Determine SDE window
    if sde_window_size > 0:
        start = random.randint(sde_window_range[0], sde_window_range[1] - sde_window_size)
        end = start + sde_window_size
        sde_window = (start, end)
    else:
        sde_window = (0, len(timesteps)-1)

    # Handle guidance
    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    # Storage for main results
    main_all_latents = []
    main_all_log_probs = []
    main_all_timesteps = []

    # Storage for intermediate results
    intermediate_ode_results = []
    sde_intermediate_latents = []  # Store latents after each SDE step

    # Main denoising loop (same as original but with intermediate storage)
    pipeline.scheduler.set_begin_index(0)
    current_latents = latents.clone()

    for i, t in enumerate(timesteps):
        # Determine noise level for current step (same logic as original)
        if i < sde_window[0]:
            cur_noise_level = 0
        elif i == sde_window[0]:
            cur_noise_level = noise_level
            main_all_latents.append(current_latents)
        elif i > sde_window[0] and i < sde_window[1]:
            cur_noise_level = noise_level
        else:
            cur_noise_level = 0

        # Standard denoising step (same as original)
        timestep = t.expand(current_latents.shape[0]).to(current_latents.dtype)
        noise_pred = pipeline.transformer(
            hidden_states=current_latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        latents_dtype = current_latents.dtype
        current_latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            pipeline.scheduler,
            noise_pred.float(),
            t.unsqueeze(0).repeat(current_latents.shape[0]),
            current_latents.float(),
            noise_level=cur_noise_level,
            sde_type=sde_type,
        )
        if current_latents.dtype != latents_dtype:
            current_latents = current_latents.to(latents_dtype)

        # Store main SDE results (same as original)
        if i >= sde_window[0] and i < sde_window[1]:
            main_all_latents.append(current_latents)
            main_all_log_probs.append(log_prob)
            main_all_timesteps.append(t)

            # Store intermediate latents for later ODE completion
            sde_intermediate_latents.append({
                'step': i,
                'latents': current_latents.clone(),
                'timestep_index': i + 1  # Next timestep index for ODE continuation
            })

    # Generate intermediate ODE completions
    # Note: Skip the last SDE state because it will produce the same result as main_results
    # (the last SDE step + remaining ODE steps = main trajectory)
    for idx, sde_state in enumerate(sde_intermediate_latents[:-1]):  # Exclude last to avoid redundant computation
        sde_steps_count = idx + 1  # Number of SDE steps completed
        start_latents = sde_state['latents']
        start_timestep_idx = sde_state['timestep_index']

        # ODE completion from this intermediate state
        remaining_timesteps = timesteps[start_timestep_idx:]
        if len(remaining_timesteps) == 0:
            continue

        ode_latents = start_latents.clone()

        # Reset scheduler for ODE completion
        pipeline.scheduler.set_begin_index(0)

        # ODE completion loop
        for ode_i, ode_t in enumerate(remaining_timesteps):
            ode_timestep = ode_t.expand(ode_latents.shape[0]).to(ode_latents.dtype)
            # print(ode_latents.dtype, ode_timestep.dtype)
            ode_noise_pred = pipeline.transformer(
                hidden_states=ode_latents,
                timestep=ode_timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            # Pure ODE step (noise_level=0) but still compute log_prob for consistency
            ode_latents, ode_log_prob, _, _ = sde_step_with_logprob(
                pipeline.scheduler,
                ode_noise_pred.float(),
                ode_t.unsqueeze(0).repeat(ode_latents.shape[0]),
                ode_latents.float(),
                noise_level=0,  # Pure ODE
                sde_type=sde_type,
            )

            if ode_latents.dtype != latents_dtype:
                ode_latents = ode_latents.to(latents_dtype)


        # Decode intermediate ODE result
        ode_decoded_latents = pipeline._unpack_latents(ode_latents, height, width, pipeline.vae_scale_factor)
        ode_decoded_latents = (ode_decoded_latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        ode_decoded_latents = ode_decoded_latents.to(dtype=pipeline.vae.dtype)
        ode_image = pipeline.vae.decode(ode_decoded_latents, return_dict=False)[0]
        ode_image = pipeline.image_processor.postprocess(ode_image, output_type=output_type)

        # Store intermediate result with same structure as main result
        intermediate_ode_results.append({
            'sde_steps_count': sde_steps_count,
            'images': ode_image,
            'specific_sde_start': sde_state['step']
        })

    # Final decode for main result
    final_decoded_latents = pipeline._unpack_latents(current_latents, height, width, pipeline.vae_scale_factor)
    final_decoded_latents = (final_decoded_latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    final_decoded_latents = final_decoded_latents.to(dtype=pipeline.vae.dtype)
    final_image = pipeline.vae.decode(final_decoded_latents, return_dict=False)[0]
    final_image = pipeline.image_processor.postprocess(final_image, output_type=output_type)

    # Main results (same structure as original pipeline output)
    main_results = {
        'images': final_image,
        'latents': main_all_latents[:-1],  # Exclude last latent (same as main logic)
        'next_latents': main_all_latents[1:],  # Next latents (same as main logic)
        'log_probs': main_all_log_probs,
        'timesteps': main_all_timesteps,
        'image_ids': latent_image_ids,
        'text_ids': text_ids
    }

    # Add the final main result as the last intermediate result
    # This avoids redundant computation: the last SDE step + remaining ODE = main trajectory
    if len(sde_intermediate_latents) > 0:
        last_sde_state = sde_intermediate_latents[-1]
        intermediate_ode_results.append({
            'sde_steps_count': len(sde_intermediate_latents),  # All SDE steps completed
            'images': final_image,  # Reuse main result's image
            'specific_sde_start': last_sde_state['step']
        })

    return main_results, intermediate_ode_results

