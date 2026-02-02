from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from .sd3_pipeline_with_logprob_fast import pipeline_with_logprob
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .sd3_sde_with_logprob import sde_step_with_logprob
import random

@torch.no_grad()
def pipeline_ode_sampling(
    pipeline,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
    num_inference_steps,
    guidance_scale,
    output_type,
    height,
    width,
    generator,
    sde_type,
):
    """
    Pure ODE sampling
    """
    images, _, _, _ = pipeline_with_logprob(
        pipeline,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type=output_type,
        height=height,
        width=width,
        noise_level=0,
        generator=generator,
        sde_window_size=0,
        sde_type=sde_type,
    )
    return images


@torch.no_grad()
def pipeline_mixed_sampling_with_intermediate_ode(
    pipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_layer_guidance_scale: float = 2.8,
    noise_level: float = 0.7,
    sde_window_size: int = 0,
    sde_window_range: tuple = (0, 5),
    sde_type: Optional[str] = 'sde',
):
    """
    Mixed SDE+ODE sampling with intermediate ODE completions for SD3
    Each intermediate result provides the same structure as main results for GRPO training

    Returns:
        main_results: Main SDE+ODE sampling results (same structure as original)
        intermediate_ode_results: List of {sde_steps_count, images, specific_sde_start}
    """
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    # 1. Check inputs
    pipeline.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    pipeline._guidance_scale = guidance_scale
    pipeline._skip_layer_guidance_scale = skip_layer_guidance_scale
    pipeline._clip_skip = clip_skip
    pipeline._joint_attention_kwargs = joint_attention_kwargs
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
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=pipeline.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    if pipeline.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = pipeline.transformer.config.in_channels
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    ) # .float() # 原始代码里有

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)

    # Determine SDE window
    if sde_window_size > 0:
        start = random.randint(sde_window_range[0], sde_window_range[1] - sde_window_size)
        end = start + sde_window_size
        sde_window = (start, end)
    else:
        # No SDE window, last step excluded from training
        sde_window = (0, len(timesteps)-1)

    # Storage for main results
    main_all_latents = []
    main_all_log_probs = []
    main_all_timesteps = []

    # Storage for intermediate results
    intermediate_ode_results = []
    sde_intermediate_latents = []  # Store latents after each SDE step

    # Main denoising loop
    current_latents = latents.clone()

    for i, t in enumerate(timesteps):
        # Determine noise level for current step
        if i < sde_window[0]:
            cur_noise_level = 0
        elif i == sde_window[0]:
            cur_noise_level = noise_level
            main_all_latents.append(current_latents)
        elif i > sde_window[0] and i < sde_window[1]:
            cur_noise_level = noise_level
        else:
            cur_noise_level = 0

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([current_latents] * 2) if pipeline.do_classifier_free_guidance else current_latents
        # Broadcast to batch dimension
        timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype) # 1216添加.to

        noise_pred = pipeline.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=pipeline.joint_attention_kwargs,
            return_dict=False,
        )[0]

        # Perform guidance
        if pipeline.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + pipeline.guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents_dtype = current_latents.dtype

        current_latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            pipeline.scheduler,
            noise_pred.float(),
            t.unsqueeze(0),
            current_latents.float(),
            noise_level=cur_noise_level,
            sde_type=sde_type,
        )

        # 1216增加
        if current_latents.dtype != latents_dtype:
            current_latents = current_latents.to(latents_dtype)

        # Store main SDE results
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
    for idx, sde_state in enumerate(sde_intermediate_latents[:-1]):
        sde_steps_count = idx + 1  # Number of SDE steps completed
        start_latents = sde_state['latents']
        start_timestep_idx = sde_state['timestep_index']

        # ODE completion from this intermediate state
        remaining_timesteps = timesteps[start_timestep_idx:]
        if len(remaining_timesteps) == 0:
            continue

        ode_latents = start_latents.clone()

        # TODO: reset scheduler for ODE completion
        # 没搞清楚作用
        # pipeline.scheduler.set_begin_index(0)

        # ODE completion loop
        for ode_i, ode_t in enumerate(remaining_timesteps):
            # Expand the latents if we are doing classifier free guidance
            ode_latent_model_input = torch.cat([ode_latents] * 2) if pipeline.do_classifier_free_guidance else ode_latents
            ode_timestep = ode_t.expand(ode_latent_model_input.shape[0]).to(ode_latent_model_input.dtype) # 1216添加了.to

            ode_noise_pred = pipeline.transformer(
                hidden_states=ode_latent_model_input,
                timestep=ode_timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=pipeline.joint_attention_kwargs,
                return_dict=False,
            )[0]

            # Perform guidance
            if pipeline.do_classifier_free_guidance:
                ode_noise_pred_uncond, ode_noise_pred_text = ode_noise_pred.chunk(2)
                ode_noise_pred = ode_noise_pred_uncond + pipeline.guidance_scale * (ode_noise_pred_text - ode_noise_pred_uncond)

            # Pure ODE step (noise_level=0)
            ode_latents, ode_log_prob, _, _ = sde_step_with_logprob(
                pipeline.scheduler,
                ode_noise_pred.float(),
                ode_t.unsqueeze(0),
                ode_latents.float(),
                noise_level=0,  # Pure ODE
                sde_type=sde_type,
            )

            if ode_latents.dtype != latents_dtype:
                ode_latents = ode_latents.to(latents_dtype)

        # Decode intermediate ODE result
        ode_decoded_latents = (ode_latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        ode_decoded_latents = ode_decoded_latents.to(dtype=pipeline.vae.dtype)
        ode_image = pipeline.vae.decode(ode_decoded_latents, return_dict=False)[0]
        ode_image = pipeline.image_processor.postprocess(ode_image, output_type=output_type)

        # Store intermediate result
        intermediate_ode_results.append({
            'sde_steps_count': sde_steps_count,
            'images': ode_image,
            'specific_sde_start': sde_state['step']
        })

    # Final decode for main result
    final_latents = (current_latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    final_latents = final_latents.to(dtype=pipeline.vae.dtype)
    final_image = pipeline.vae.decode(final_latents, return_dict=False)[0]
    final_image = pipeline.image_processor.postprocess(final_image, output_type=output_type)

    # Main results (same structure as original pipeline output)
    main_results = {
        'images': final_image,
        'latents': main_all_latents[:-1],  # Exclude last latent
        'next_latents': main_all_latents[1:],  # Next latents
        'log_probs': main_all_log_probs,
        'timesteps': main_all_timesteps,
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

    # Offload all models
    pipeline.maybe_free_model_hooks()

    return main_results, intermediate_ode_results
