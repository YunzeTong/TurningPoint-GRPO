from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import is_compiled_module
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob_fast import pipeline_with_logprob, calculate_shift
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

from flow_grpo.diffusers_patch.flux_pipeline_with_logprob_fast_my import pipeline_ode_sampling, pipeline_mixed_sampling_with_intermediate_ode

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


        
def compute_log_prob(transformer, pipeline, sample, j, config):
    packed_noisy_model_input = sample["latents"][:, j]
    device = packed_noisy_model_input.device
    dtype = packed_noisy_model_input.dtype
    if transformer.module.config.guidance_embeds:
        guidance = torch.tensor([config.sample.guidance_scale], device=device)
        guidance = guidance.expand(packed_noisy_model_input.shape[0])
    else:
        guidance = None

    # Predict the noise residual
    model_pred = transformer(
        hidden_states=packed_noisy_model_input,
        timestep=sample["timesteps"][:, j] / 1000,
        guidance=guidance,
        pooled_projections=sample["pooled_prompt_embeds"],
        encoder_hidden_states=sample["prompt_embeds"],
        txt_ids= torch.zeros(sample["prompt_embeds"].shape[1], 3).to(device=device, dtype=dtype),
        img_ids=sample["image_ids"][0],
        return_dict=False,
    )[0]
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        model_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
        sde_type=config.sample.sde_type,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters, tensorboard_writer=None):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        with autocast():
            with torch.no_grad():
                images, _, _, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.eval_guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                    sde_window_size=0,
                    sde_type=config.sample.sde_type,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            # Log to tensorboard instead of wandb
            if tensorboard_writer:
                # Log evaluation rewards
                for key, value in all_rewards.items():
                    valid_values = value[value != -10]
                    if len(valid_values) > 0:
                        tensorboard_writer.add_scalar(f"eval/reward_{key}", np.mean(valid_values), global_step)

                # Log evaluation images
                for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards)):
                    # Read image and convert to tensor
                    pil = Image.open(os.path.join(tmpdir, f"{idx}.jpg"))
                    img_tensor = torch.from_numpy(np.array(pil)).permute(2, 0, 1)  # HWC to CHW

                    # Add image to tensorboard
                    tensorboard_writer.add_image(f"eval/image_{idx}", img_tensor, global_step, dataformats="CHW")

                # Add text captions
                # caption_text = "\n".join([
                #     f"Image {idx}: {prompt:.100} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10)
                #     for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                # ])
                # tensorboard_writer.add_text("eval/captions", caption_text, global_step)

                # Record each image's reward separately to avoid text length limits
                for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards)):
                    reward_text = " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10)
                    caption_text = f"Prompt: {prompt}\nRewards: {reward_text}"
                    tensorboard_writer.add_text(f"eval/image_{idx}_info", caption_text, global_step)
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        # config.run_name += "_" + unique_id
        pass

    config.save_dir = os.path.join(config.save_dir, config.run_name)

    if config.logdir is None or config.logdir == "":
        config.logdir = os.path.join(config.save_dir, "logs")

    if config.sample.sde_window_size > 0:
        num_train_timesteps = config.sample.sde_window_size
    else:
        num_train_timesteps = config.sample.num_steps - 1

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    # Setup tensorboard writer
    tensorboard_writer = None
    if accelerator.is_main_process:

        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir, exist_ok=True)
        print(f"Saving model checkpoints and logs to {config.save_dir}")

        if not os.path.exists(config.logdir):
            os.makedirs(config.logdir, exist_ok=True)
        print(f"Saving logs to {config.logdir}")

        # config.save_dir = os.path.join(config.save_dir, "checkpoints")
        # if not os.path.exists(config.save_dir):
        #     os.makedirs(config.save_dir, exist_ok=True)
        # print(f"Saving checkpoints to {config.save_dir}")

        # log_dir = os.path.join(config.logdir, config.run_name)
        # os.makedirs(log_dir, exist_ok=True)

        tensorboard_writer = SummaryWriter(
            log_dir=config.logdir,
            comment=f"flux_grpo_fast_{config.run_name}"
        )

        # Save config as text for tensorboard
        config_text = json.dumps(config.to_dict(), indent=2, ensure_ascii=False)
        tensorboard_writer.add_text("config", config_text, 0)

        logger.info(f"TensorBoard logging to: {tensorboard_writer.log_dir}")
        # accelerator.init_trackers(
        #     project_name="flow-grpo",
        #     config=config.to_dict(),
        #     init_kwargs={"wandb": {"name": config.run_name}},
        # )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    if config.use_delta_global:
        logger.info(f"USING DELTA GLOBAL")
    else:
        logger.info(f"USING SDE REWARD ITSELF!!!")

    # load scheduler, tokenizer and models.
    pipeline = FluxPipeline.from_pretrained(
        config.pretrained.model,
        low_cpu_mem_usage=False
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    # For process reward: create one stat tracker per timestep
    # This allows per-prompt grouping while preserving timestep-specific advantages
    if config.per_prompt_stat_tracking:
        stat_trackers_per_timestep = [
            PerPromptStatTracker(config.sample.global_std)
            for _ in range(num_train_timesteps)
        ]
        logger.info(f"Initialized {num_train_timesteps} per-timestep stat trackers for process reward")

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # for deepspeed zero
    if accelerator.state.deepspeed_plugin:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.sample.train_batch_size
    # prepare prompt and reward fn
    if is_deepspeed_zero3_enabled():
        # Using deepspeed zero3 will cause the model parameter `weight.shape` to be empty.
        unset_hf_deepspeed_config()
        reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        set_hf_deepspeed_config(accelerator.state.deepspeed_plugin.dschf)
    else:
        reward_fn, eval_reward_fn = None, None
        reward_type = None
        for key, value in config.reward_fn.items():
            reward_type = key
        if reward_type == "geneval_local":
            local_rank = accelerator.local_process_index
            reward_fn = getattr(flow_grpo.rewards, 'multi_score')("cuda:{}".format(local_rank), config.reward_fn)
            eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')("cuda:{}".format(local_rank), config.reward_fn)
        else:
            reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
            eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    
    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)
    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters, tensorboard_writer)
        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None

            with autocast():
                with torch.no_grad():
                    # Original mixed SDE+ODE sampling with intermediate ODE completions
                    main_results, intermediate_ode_results = pipeline_mixed_sampling_with_intermediate_ode(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        generator=generator,
                        noise_level=config.sample.noise_level,
                        sde_window_size=config.sample.sde_window_size,
                        sde_window_range=config.sample.sde_window_range,
                        sde_type=config.sample.sde_type,
                    )

                    # Pure ODE sampling with same prompt and initial noise
                    ode_images = pipeline_ode_sampling(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        generator=generator,  # Same generator for consistent initial noise
                    )

            # Extract main results
            images = main_results['images']
            # 这里的shape在注释上和原始的对不上是正常的，因为pipeline_mixed_sampling_with_ode_sampling会丢掉最后一步的
            # 原始的没有丢掉，所以latents的num_steps会+1
            latents = torch.stack(main_results['latents'], dim=1)  # (batch_size, num_steps, 16, 96, 96)
            next_latents = torch.stack(main_results['next_latents'], dim=1)  # (batch_size, num_steps, 16, 96, 96)
            log_probs = torch.stack(main_results['log_probs'], dim=1)  # (batch_size, num_steps)
            timesteps = torch.stack(main_results['timesteps']).unsqueeze(0).repeat(config.sample.train_batch_size, 1)


            # Compute rewards for main SDE+ODE result
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)

            # Compute rewards for pure ODE result
            ode_rewards = executor.submit(reward_fn, ode_images, prompts, prompt_metadata, only_strict=True)

            # Compute rewards for intermediate ODE results
            intermediate_rewards = []
            for intermediate_result in intermediate_ode_results:
                intermediate_reward = executor.submit(reward_fn, intermediate_result['images'], prompts, prompt_metadata, only_strict=True)
                intermediate_rewards.append(intermediate_reward)

            # yield to make sure reward computation starts
            time.sleep(0)

            # Main SDE+ODE sample (for GRPO training)
            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "image_ids": main_results['image_ids'].unsqueeze(0).repeat(len(prompt_ids),1,1),
                    "timesteps": timesteps,
                    "latents": latents,  # SDE latents before timestep t
                    "next_latents": next_latents,  # SDE latents after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                    # 下面是新增的
                    "ode_rewards": ode_rewards,  # Pure ODE rewards for comparison
                    "intermediate_results": []  # Will be filled after reward computation
                }
            )

            # Store intermediate results for later processing
            samples[-1]["intermediate_ode_data"] = {
                'results': intermediate_ode_results,
                'rewards': intermediate_rewards
            } # intermediate_ode_results和intermediate_rewards都是列表，每个元素代表一种掇月后的结果

            # 看到这里，要确认intermediate

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # Main SDE+ODE rewards
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

            # Pure ODE rewards
            ode_rewards, _ = sample["ode_rewards"].result()
            sample["ode_rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in ode_rewards.items()
            }

            # Intermediate ODE rewards and prepare additional samples
            intermediate_ode_data = sample["intermediate_ode_data"]
            intermediate_results = []

            for idx, (intermediate_result, intermediate_reward_future) in enumerate(
                zip(intermediate_ode_data['results'], intermediate_ode_data['rewards'])
            ):
                intermediate_rewards, _ = intermediate_reward_future.result()

                # Convert to tensors
                intermediate_rewards_tensor = {
                    key: torch.as_tensor(value, device=accelerator.device).float()
                    for key, value in intermediate_rewards.items()
                }


                intermediate_sample = {
                    "rewards": intermediate_rewards_tensor,
                    "sde_steps_count": intermediate_result['sde_steps_count'],
                    "specific_sde_start": intermediate_result['specific_sde_start']
                }

                intermediate_results.append(intermediate_sample)

            sample["intermediate_results"] = intermediate_results
            # Clean up temporary data
            del sample["intermediate_ode_data"]

        # ==================== Process Reward Computation ====================
        # Compute per-timestep rewards based on intermediate ODE results and global SDE-ODE difference
        for sample in samples:
            batch_size = sample["rewards"]["avg"].shape[0]

            # Get main rewards (all are tensors on device)
            sde_reward = sample["rewards"]["avg"]  # (batch_size,) - final SDE+ODE reward
            ode_reward = sample["ode_rewards"]["avg"]  # (batch_size,) - pure ODE reward

            # Compute global advantage: Δ_global = R_sde - R_ode
            delta_global = sde_reward - ode_reward  # (batch_size,)

            # Get intermediate rewards sorted by sde_steps_count
            intermediate_results = sample["intermediate_results"]
            intermediate_results_sorted = sorted(intermediate_results, key=lambda x: x["sde_steps_count"])

            # Build reward sequence: [R_ode, R_1, R_2, ..., R_n]
            # where R_i is the reward after i SDE steps + ODE completion
            reward_sequence = [ode_reward]  # R_0 = R_ode
            for inter_result in intermediate_results_sorted:
                reward_sequence.append(inter_result["rewards"]["avg"])  # R_1, R_2, ..., R_n

            # Compute d_t = R_t - R_{t-1} for t=1,2,...,n
            # d_t represents the reward change caused by the SDE step from t-1 to t
            d_list = []
            for t in range(1, len(reward_sequence)):
                d_t = reward_sequence[t] - reward_sequence[t-1]
                d_list.append(d_t)  # [d_1, d_2, ..., d_n]

            # Compute s_t = sign(d_t) * sign(Δ_global) for t=1,2,...,n
            # s_t > 0 means the t-th step's trend is consistent with global trend
            s_list = []
            for d_t in d_list:
                s_t = torch.sign(d_t) * torch.sign(delta_global)
                s_list.append(s_t)  # [s_1, s_2, ..., s_n]


            timestep_rewards = []
            B_list = []
            main_list = []
            indicator_list = []
            for t in range(len(s_list)):

                indicator = torch.zeros_like(delta_global)
                if global_step >= config.turn_on_bonus_global_step:
                    if t == 0:
                        # First step bonus based on config
                        if config.apply_first_step_bonus:
                            if config.select_first_step_only_from_consistent_trajectory:
                                indicator = (
                                    ( torch.sign(d_list[0]) * torch.sign(sde_reward - reward_sequence[1]) ) > 0
                                ).float()
                                # sign(R_1 - R_0) * sign(R_sde - R_1)
                            else:
                                indicator = (s_list[0] > 0).float()
                        else:
                            indicator = torch.zeros_like(delta_global)
                    else:
                        # Bonus when previous step was wrong but current step is right
                        s_t = s_list[t]
                        s_last = s_list[t-1]
                        indicator = ((s_last < 0).float() * (s_t > 0).float())
                        if hasattr(config, 'indicator_wo_delta_global_constraint') and config.indicator_wo_delta_global_constraint:
                            indicator = (d_list[t-1] * d_list[t] < 0).float() # 只要是前后不一致的转折点，就算上
                        if config.select_only_positive_intermediate_step: # 只对那些最终提升了的sample给reward
                            indicator = indicator * (torch.sign(delta_global) > 0).float()

                        if config.select_inter_step_only_from_consistent_trajectory:
                            consistent_judge = (
                                ( torch.sign(d_list[t]) * torch.sign(sde_reward - reward_sequence[t+1]) ) > 0
                            ).float()
                            # sign(R_{t-1} - R_{t}) * sign(R_sde - R_{t-1})
                            indicator = indicator * consistent_judge

                indicator_list.append(indicator)

            # Apply balanced bonus if configured
            if config.use_balanced_bonus and global_step >= config.turn_on_bonus_global_step:
                # Stack all indicators and B_t candidates
                all_indicators = torch.stack(indicator_list, dim=1)  # (batch_size, num_timesteps)

                # Compute B_t candidates for all timesteps
                B_t_candidates = []
                for t in range(len(s_list)):
                    if config.take_delta_global_as_main:
                        B_t_cand = d_list[t]
                    else:
                        B_t_cand = delta_global if config.use_delta_global else sde_reward
                    B_t_candidates.append(B_t_cand)
                all_B_t = torch.stack(B_t_candidates, dim=1)  # (batch_size, num_timesteps)

                # Flatten to get all turning points across batch
                turning_point_mask = all_indicators > 0  # (batch_size, num_timesteps)

                # Get all B_t values at turning points (flatten both dimensions)
                B_t_at_turning_points = all_B_t[turning_point_mask]  # (num_total_turning_points,)

                if B_t_at_turning_points.numel() > 0:
                    # Separate positive and negative B_t across all turning points in the batch
                    positive_mask_flat = B_t_at_turning_points > 0
                    negative_mask_flat = B_t_at_turning_points < 0

                    num_positive = positive_mask_flat.sum().item()
                    num_negative = negative_mask_flat.sum().item()

                    if num_positive > 0 and num_negative > 0:
                        # Need to balance: keep min(num_positive, num_negative) of each
                        keep_count = min(num_positive, num_negative)

                        # Get flat indices of all turning points
                        turning_point_flat_indices = torch.where(turning_point_mask.flatten())[0]

                        # Separate into positive and negative groups
                        positive_turning_indices = turning_point_flat_indices[positive_mask_flat]
                        negative_turning_indices = turning_point_flat_indices[negative_mask_flat]

                        # Get B_t values for positive and negative groups
                        positive_B_t_values = B_t_at_turning_points[positive_mask_flat]
                        negative_B_t_values = B_t_at_turning_points[negative_mask_flat]

                        # Select top-k by absolute value from each group
                        balanced_turning_indices = []
                        if num_positive > keep_count:
                            # Select keep_count largest absolute values from positive group
                            _, topk_indices = torch.topk(torch.abs(positive_B_t_values), keep_count)
                            balanced_turning_indices.append(positive_turning_indices[topk_indices])
                        else:
                            balanced_turning_indices.append(positive_turning_indices)

                        if num_negative > keep_count:
                            # Select keep_count largest absolute values from negative group
                            _, topk_indices = torch.topk(torch.abs(negative_B_t_values), keep_count)
                            balanced_turning_indices.append(negative_turning_indices[topk_indices])
                        else:
                            balanced_turning_indices.append(negative_turning_indices)

                        # Combine selected indices
                        balanced_turning_indices = torch.cat(balanced_turning_indices)

                        # Create new indicator mask: zero out everything first, then set selected turning points to 1
                        balanced_indicators_flat = torch.zeros_like(all_indicators.flatten())
                        balanced_indicators_flat[balanced_turning_indices] = 1.0
                        balanced_indicators = balanced_indicators_flat.reshape(all_indicators.shape)

                        # Use balanced indicators
                        indicator_list = [balanced_indicators[:, t] for t in range(len(s_list))]
                    # else: if all same sign or no turning points, keep original indicators

            # Propagate bonus to future steps if configured
            # When a timestep's indicator is True, all subsequent timesteps in that trajectory
            # also receive the same bonus (using B_t value from the triggering timestep)
            if config.propagate_bonus_to_future_steps and global_step >= config.turn_on_bonus_global_step:
                # Stack indicators: shape (batch_size, num_timesteps)
                indicator_tensor = torch.stack(indicator_list, dim=1)  # (batch_size, T)

                # Compute B_t values for all timesteps (before applying indicator)
                B_t_raw_list = []
                for t in range(len(s_list)):
                    if config.take_delta_global_as_main:
                        B_t_raw = d_list[t]
                    else:
                        B_t_raw = delta_global if config.use_delta_global else sde_reward
                        if config.use_sde_minus_rt:
                            B_t_raw = sde_reward - reward_sequence[t]
                    B_t_raw_list.append(B_t_raw)
                B_t_raw_tensor = torch.stack(B_t_raw_list, dim=1)  # (batch_size, T)

                # Create propagated indicator and bonus tensors
                propagated_indicator = indicator_tensor.clone()
                # Store the bonus value to use (from the first triggering timestep)
                propagated_bonus_value = torch.zeros_like(indicator_tensor)

                # For each sample, propagate indicators and bonus values forward
                for b in range(indicator_tensor.shape[0]):


                    neg_accumulated_times = 0
                    neg_is_activated = False
                    # 对于下降的转折点，为其前面的steps全部追加负值
                    for t in reversed(range(indicator_tensor.shape[1])):
                        if (indicator_tensor[b, t] > 0) and ((sde_reward[b] - reward_sequence[t][b]).item() < 0):
                            neg_is_activated = True
                            neg_accumulated_times += 1
                            propagated_bonus_value[b, t] += neg_accumulated_times * (sde_reward[b] - reward_sequence[t][b]).item()
                        elif neg_is_activated:
                            propagated_indicator[b, t] = 1.0
                            propagated_bonus_value[b, t] += neg_accumulated_times * (sde_reward[b] - reward_sequence[t][b]).item()

                    pos_accumulated_times = 0
                    pos_is_activated = False
                    for t in range(indicator_tensor.shape[1]):
                        if (indicator_tensor[b, t] > 0) and ((sde_reward[b] - reward_sequence[t][b]).item() > 0):
                            pos_is_activated = True
                            pos_accumulated_times += 1
                            propagated_bonus_value[b, t] += pos_accumulated_times * (sde_reward[b] - reward_sequence[t][b]).item()
                        elif pos_is_activated:
                            propagated_indicator[b, t] = 1.0
                            propagated_bonus_value[b, t] += pos_accumulated_times * (sde_reward[b] - reward_sequence[t][b]).item()


                # Update indicator_list with propagated values
                indicator_list = [propagated_indicator[:, t] for t in range(len(s_list))]
                # Store propagated bonus values for use in second pass
                propagated_bonus_values_list = [propagated_bonus_value[:, t] for t in range(len(s_list))]

            # Second pass: compute final rewards with (possibly balanced) indicators
            for t in range(len(s_list)):
                main_t = torch.zeros_like(delta_global)
                B_t = torch.zeros_like(delta_global)
                # Configure main and bonus components based on config
                if config.take_delta_global_as_main:
                    main_t = delta_global if config.use_delta_global else sde_reward
                    B_t = d_list[t]
                else:
                    main_t = d_list[t]
                    B_t = delta_global if config.use_delta_global else sde_reward
                    if config.use_sde_minus_rt:
                        B_t = sde_reward - reward_sequence[t]

                # Apply bonus with eta coefficient
                if config.propagate_bonus_to_future_steps and global_step >= config.turn_on_bonus_global_step:
                    # Use pre-computed propagated bonus values
                    # propagated_bonus_values_list[t] contains:
                    # - Original B_t for triggering timesteps
                    # - Accumulated bonus from all previous triggering timesteps for propagated timesteps
                    B_t = config.eta * propagated_bonus_values_list[t]
                else:
                    B_t = config.eta * indicator_list[t] * B_t
                B_list.append(B_t)

                if config.drop_original_reward_when_having_bonus:
                    # 对于B_t中有有效值添加bonus的部分，将其对应的main_t设置为0，使得这些使用bonus reward的步的reward只来源于bonus
                    bonus_mask = (B_t != 0)  # 找出有bonus的位置
                    main_t = main_t.clone()  # 避免修改原始tensor
                    main_t[bonus_mask] = 0   # 将有bonus的位置的main_t设为0

                main_list.append(main_t)

                # Final timestep reward
                R_t = main_t + B_t
                timestep_rewards.append(R_t)

            # Convert to tensor: shape (batch_size, num_sde_timesteps)
            # TODO：这个有问题，现在shape为（3，4），按理说应该(3, 3)，可能跟B_list最后多加了一个有关。 12.5直接修改后目前能跑
            timestep_rewards_tensor = torch.stack(timestep_rewards, dim=1)

            # Store in sample for later advantage computation
            sample["process_rewards"] = timestep_rewards_tensor
            sample["delta_global"] = delta_global
            sample["d_list"] = torch.stack(d_list, dim=1) if len(d_list) > 0 else torch.zeros(batch_size, 0, device=delta_global.device)
            sample["s_list"] = torch.stack(s_list, dim=1) if len(s_list) > 0 else torch.zeros(batch_size, 0, device=delta_global.device)
            sample["B_list"] = torch.stack(B_list, dim=1)
            sample["main_list"] = torch.stack(main_list, dim=1)

        # Tensorboard logging for process rewards (before gathering)
        if accelerator.is_local_main_process and tensorboard_writer and len(samples) > 0:
            # Aggregate across all samples in this epoch (on current device)
            all_delta_global = torch.cat([s["delta_global"] for s in samples], dim=0)
            all_process_rewards = torch.cat([s["process_rewards"] for s in samples], dim=0) # 推测：(n * bsz, 3)
            all_B_list = torch.cat([s["B_list"] for s in samples], dim=0)
            all_d_list = torch.cat([s["d_list"] for s in samples], dim=0)
            all_s_list = torch.cat([s["s_list"] for s in samples], dim=0)
            all_main_list = torch.cat([s["main_list"] for s in samples], dim=0)

            # Log delta_global statistics
            tensorboard_writer.add_scalar("process_reward/delta_global_mean", all_delta_global.mean().item(), global_step)
            tensorboard_writer.add_scalar("process_reward/delta_global_std", all_delta_global.std().item(), global_step)
            tensorboard_writer.add_histogram("process_reward/delta_global_dist", all_delta_global.cpu().numpy(), global_step)

            # Log per-timestep rewards
            num_timesteps = all_process_rewards.shape[1]
            for t in range(num_timesteps):
                tensorboard_writer.add_scalar(f"process_reward/R_t_{t}_mean", all_process_rewards[:, t].mean().item(), global_step)
                tensorboard_writer.add_scalar(f"process_reward/B_t_{t}_mean", all_B_list[:, t].mean().item(), global_step)
                tensorboard_writer.add_scalar(f"process_reward/B_t_{t}_nonzero_ratio", (all_B_list[:, t] != 0).float().mean().item(), global_step)
                tensorboard_writer.add_scalar(f"process_reward/main_t_{t}_mean", all_main_list[:, t].mean().item(), global_step)

            # Log d_t and s_t statistics
            if all_d_list.shape[1] > 0:
                for t in range(all_d_list.shape[1]):
                    tensorboard_writer.add_scalar(f"process_reward/d_{t+1}_mean", all_d_list[:, t].mean().item(), global_step)
                    tensorboard_writer.add_scalar(f"process_reward/d_{t+1}_std", all_d_list[:, t].std().item(), global_step)
                    # s_t consistency: ratio of samples where step aligns with global
                    s_positive_ratio = (all_s_list[:, t] > 0).float().mean().item()
                    tensorboard_writer.add_scalar(f"process_reward/s_{t+1}_positive_ratio", s_positive_ratio, global_step)

            # Log bonus trigger statistics
            for t in range(1, num_timesteps):
                # Check how often bonus condition is triggered: s_t < 0 and s_{t+1} > 0
                if all_s_list.shape[1] > t:
                    s_t = all_s_list[:, t-1]
                    s_next = all_s_list[:, t]
                    trigger_ratio = ((s_t < 0).float() * (s_next > 0).float()).mean().item() # 12.7修改，原来的符号错了
                    tensorboard_writer.add_scalar(f"process_reward/bonus_trigger_{t}_ratio", trigger_ratio, global_step)

            # Log balanced bonus statistics
            if hasattr(config, 'use_balanced_bonus') and config.use_balanced_bonus:
                # Count positive and negative bonuses applied
                positive_bonus_mask = all_B_list > 0
                negative_bonus_mask = all_B_list < 0
                zero_bonus_mask = all_B_list == 0

                tensorboard_writer.add_scalar("process_bonus_balance/positive_bonus_ratio", positive_bonus_mask.float().mean().item(), global_step)
                tensorboard_writer.add_scalar("process_bonus_balance/negative_bonus_ratio", negative_bonus_mask.float().mean().item(), global_step)
                tensorboard_writer.add_scalar("process_bonus_balance/zero_bonus_ratio", zero_bonus_mask.float().mean().item(), global_step)

                # Log per-timestep balance
                for t in range(num_timesteps):
                    num_positive = (all_B_list[:, t] > 0).sum().item()
                    num_negative = (all_B_list[:, t] < 0).sum().item()
                    tensorboard_writer.add_scalar(f"process_bonus_balance/timestep_{t}_positive_count", num_positive, global_step)
                    tensorboard_writer.add_scalar(f"process_bonus_balance/timestep_{t}_negative_count", num_negative, global_step)
                    if num_positive + num_negative > 0:
                        balance_ratio = num_positive / (num_positive + num_negative)
                        tensorboard_writer.add_scalar(f"process_bonus_balance/timestep_{t}_positive_ratio", balance_ratio, global_step)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        # Handle special keys differently
        special_keys = {'intermediate_results', 'ode_rewards', 'images'}
        process_reward_keys = {'process_rewards', 'delta_global', 'd_list', 's_list', 'B_list', 'main_list'}

        samples_main = {}
        for k in samples[0].keys():
            if k in special_keys:
                # Keep as list for visualization
                samples_main[k] = [s[k] for s in samples]
            elif k in process_reward_keys:
                # Concatenate process reward tensors
                samples_main[k] = torch.cat([s[k] for s in samples], dim=0)
            elif isinstance(samples[0][k], dict):
                # Concatenate dict values (e.g., rewards)
                samples_main[k] = {
                    sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                    for sub_key in samples[0][k]
                }
            else:
                # Concatenate regular tensors
                samples_main[k] = torch.cat([s[k] for s in samples], dim=0)

        # 后面这段直到if epoch % 10没搞清楚要干啥
        # Store additional data for logging
        all_ode_rewards = []
        all_intermediate_results = []

        for sample in samples:
            all_ode_rewards.append(sample["ode_rewards"])
            all_intermediate_results.extend(sample["intermediate_results"])

        # Rename to maintain compatibility with existing training code
        samples = samples_main

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]
                sampled_ode_rewards = [ode_rewards['avg'][i] for i in sample_indices]

                # Organize intermediate rewards by timestep
                # We need to extract rewards from all_intermediate_results for the last batch
                sampled_intermediate_rewards = {}
                num_intermediate_per_batch = len(intermediate_ode_results)
                # Get the last batch's intermediate results
                last_batch_intermediate = all_intermediate_results[-num_intermediate_per_batch:] if all_intermediate_results else []

                for inter_sample in last_batch_intermediate:
                    temp_timestep = inter_sample['specific_sde_start']
                    # Extract rewards for sampled indices
                    sampled_intermediate_rewards[temp_timestep] = [
                        inter_sample['rewards']['avg'][i].item() for i in sample_indices
                    ]


                ############### CHECK IMAGE GENERATION #################
                for idx, i in enumerate(sample_indices):
                    image = ode_images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"ode_{idx}.jpg"))

                for idx, i in enumerate(sample_indices):
                    for intermediate_result in intermediate_ode_results:
                        image = intermediate_result['images'][i]
                        temp_timestep = intermediate_result['specific_sde_start']
                        pil = Image.fromarray(
                            (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        )
                        pil = pil.resize((config.resolution, config.resolution))
                        pil.save(os.path.join(tmpdir, f"sde_{idx}_{temp_timestep}.jpg"))
                ############### CHECK IMAGE GENERATION #################

                # Log training images to tensorboard
                if tensorboard_writer:
                    for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards)):
                        # Read image and convert to tensor
                        pil = Image.open(os.path.join(tmpdir, f"{idx}.jpg"))
                        img_tensor = torch.from_numpy(np.array(pil)).permute(2, 0, 1)  # HWC to CHW

                        # Add image to tensorboard
                        tensorboard_writer.add_image(f"train/image_{idx}", img_tensor, global_step, dataformats="CHW")

                        ############### CHECK IMAGE GENERATION #################
                        pil = Image.open(os.path.join(tmpdir, f"ode_{idx}.jpg"))
                        img_tensor = torch.from_numpy(np.array(pil)).permute(2, 0, 1)  # HWC to CHW
                        tensorboard_writer.add_image(f"train/image_ode_{idx}", img_tensor, global_step, dataformats="CHW")

                        # Add reward info for ODE image
                        ode_reward = sampled_ode_rewards[idx]
                        ode_caption = f"Prompt: {prompt}\nODE Reward: {ode_reward:.4f}"
                        tensorboard_writer.add_text(f"train/image_ode_{idx}_info", ode_caption, global_step)

                        for intermediate_result in intermediate_ode_results:
                            temp_timestep = intermediate_result['specific_sde_start']
                            pil = Image.open(os.path.join(tmpdir, f"sde_{idx}_{temp_timestep}.jpg"))
                            img_tensor = torch.from_numpy(np.array(pil)).permute(2, 0, 1)
                            tensorboard_writer.add_image(f"train/image_sde_{idx}_{temp_timestep}", img_tensor, global_step, dataformats="CHW")

                            # Add reward info for intermediate SDE image
                            if temp_timestep in sampled_intermediate_rewards:
                                inter_reward = sampled_intermediate_rewards[temp_timestep][idx]
                                inter_caption = f"Prompt: {prompt}\nSDE Step: {temp_timestep}\nIntermediate Reward: {inter_reward:.4f}"
                                tensorboard_writer.add_text(f"train/image_sde_{idx}_{temp_timestep}_info", inter_caption, global_step)
                        ############### CHECK IMAGE GENERATION #################

                    # Add text captions for training images
                    # train_caption_text = "\n".join([
                    #     f"Image {idx}: {prompt:.100} | avg: {avg_reward:.2f}"
                    #     for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    # ])
                    # tensorboard_writer.add_text("train/captions", train_caption_text, global_step)

                    # Record each image's reward separately to avoid text length limits
                    for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards)):
                        caption_text = f"Prompt: {prompt}\nAvg Reward: {avg_reward:.4f}"
                        tensorboard_writer.add_text(f"train/image_{idx}_info", caption_text, global_step)
        
        # Store original average reward for logging
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]

        # ==================== Use Process Rewards for Advantage Calculation ====================
        # Instead of repeating the same reward across timesteps, we now use per-timestep process rewards
        # process_rewards shape: (batch_size, num_train_timesteps)
        # Each timestep gets a specific reward based on R_t = Δ_global + B_t

        # Gather process rewards across processes
        process_rewards = samples["process_rewards"]  # (batch_size, num_train_timesteps)
        gathered_process_rewards = accelerator.gather(process_rewards)  # (batch_size * num_processes, num_train_timesteps)
        gathered_process_rewards_np = gathered_process_rewards.cpu().numpy()

        # Also gather original rewards for backward compatibility logging
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        # Gather and log ODE rewards
        gathered_ode_rewards = {}
        if all_ode_rewards:
            # Aggregate ODE rewards across batches
            ode_reward_keys = all_ode_rewards[0].keys()
            for key in ode_reward_keys:
                all_values = []
                for batch_ode_rewards in all_ode_rewards:
                    all_values.append(batch_ode_rewards[key])
                combined_values = torch.cat(all_values, dim=0) # 此时的all_values有48个(3,)的tensor, 共144个值，8个进程
                gathered_ode_rewards[key] = accelerator.gather(combined_values).cpu().numpy()
        # gathered_ode_rewards和gathered_rewards在pickscore上一致，但是avg相比，后者(1152,3), 前者(1152,)
        # 合理，参考：samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)

        # Gather and log intermediate ODE rewards
        gathered_intermediate_rewards = {}
        if all_intermediate_results:
            # Group intermediate results by SDE step count
            intermediate_by_steps = {}
            for intermediate_sample in all_intermediate_results:
                sde_start_step = intermediate_sample['specific_sde_start']
                if sde_start_step not in intermediate_by_steps:
                    intermediate_by_steps[sde_start_step] = []
                intermediate_by_steps[sde_start_step].append(intermediate_sample['rewards'])

            # Gather rewards for each SDE step count
            for sde_start_step, reward_list in intermediate_by_steps.items():
                if reward_list:
                    reward_keys = reward_list[0].keys()
                    for key in reward_keys:
                        all_values = []
                        for reward_dict in reward_list:
                            all_values.append(reward_dict[key])
                        combined_values = torch.cat(all_values, dim=0)
                        gathered_value = accelerator.gather(combined_values).cpu().numpy()
                        gathered_intermediate_rewards[f"sde_{sde_start_step}_{key}"] = gathered_value

        # log rewards and images
        if accelerator.is_main_process and tensorboard_writer:
            # Log main SDE+ODE rewards to tensorboard
            tensorboard_writer.add_scalar("train/epoch", epoch, global_step)
            for key, value in gathered_rewards.items():
                if '_strict_accuracy' not in key and '_accuracy' not in key:
                    tensorboard_writer.add_scalar(f"train/reward_sde_ode_{key}", value.mean(), global_step)

            # Log pure ODE rewards
            for key, value in gathered_ode_rewards.items():
                if '_strict_accuracy' not in key and '_accuracy' not in key:
                    tensorboard_writer.add_scalar(f"train/reward_pure_ode_{key}", value.mean(), global_step)

            # Log intermediate ODE rewards
            for key, value in gathered_intermediate_rewards.items():
                if '_strict_accuracy' not in key and '_accuracy' not in key:
                    tensorboard_writer.add_scalar(f"train/reward_intermediate_{key}", value.mean(), global_step)

            # Log reward comparisons
            if gathered_ode_rewards and 'avg' in gathered_rewards and 'avg' in gathered_ode_rewards:
                sde_ode_avg = gathered_rewards['avg'].mean()
                pure_ode_avg = gathered_ode_rewards['avg'].mean()
                reward_diff = sde_ode_avg - pure_ode_avg
                tensorboard_writer.add_scalar("train/reward_diff_sde_vs_ode", reward_diff, global_step)



        # ==================== Compute Advantages Using Process Rewards ====================
        # Compute per-timestep advantages using process rewards
        # For each timestep, normalize across all samples at that timestep

        if config.per_prompt_stat_tracking:
            # Per-timestep per-prompt stat tracking with process rewards
            # This combines the benefits of:
            # 1. Per-prompt grouping (comparing samples from same prompt)
            # 2. Timestep-specific rewards (preserving process reward information)

            # Gather prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )

            # Compute advantages separately for each timestep using its own stat tracker
            advantages_per_timestep = []
            for t in range(gathered_process_rewards_np.shape[1]):
                # Get rewards for timestep t across all samples
                rewards_t = gathered_process_rewards_np[:, t]  # (N,)

                # Use the t-th stat tracker to compute per-prompt advantages for timestep t
                # This normalizes within prompts but allows different timesteps to have different values
                advantages_t = stat_trackers_per_timestep[t].update(prompts, rewards_t)
                advantages_per_timestep.append(advantages_t)

            # Stack advantages: shape (N, T)
            advantages = np.stack(advantages_per_timestep, axis=1)

            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            # Get statistics from the first timestep's tracker (representative)
            group_size, trained_prompt_num = stat_trackers_per_timestep[0].get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process and tensorboard_writer:
                # Log per-prompt statistics to tensorboard
                tensorboard_writer.add_scalar("train/group_size", group_size, global_step)
                tensorboard_writer.add_scalar("train/trained_prompt_num", trained_prompt_num, global_step)
                tensorboard_writer.add_scalar("train/zero_std_ratio", zero_std_ratio, global_step)
                tensorboard_writer.add_scalar("train/reward_std_mean", reward_std_mean, global_step)

                # Log per-timestep group statistics
                for t in range(num_train_timesteps):
                    group_size_t, _ = stat_trackers_per_timestep[t].get_stats()
                    tensorboard_writer.add_scalar(f"train/group_size_timestep_{t}", group_size_t, global_step)

            # Clear all stat trackers
            for tracker in stat_trackers_per_timestep:
                tracker.clear()
        else:
            # Global normalization: compute advantages per timestep independently
            # For each timestep t, normalize R_t across all samples
            advantages_per_timestep = []
            for t in range(gathered_process_rewards_np.shape[1]):
                rewards_t = gathered_process_rewards_np[:, t]  # (N,)
                # Z-score normalization per timestep
                advantages_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-4)
                advantages_per_timestep.append(advantages_t)

            advantages = np.stack(advantages_per_timestep, axis=1)  # (N, T)

        # Log advantage statistics
        if accelerator.is_main_process and tensorboard_writer:
            for t in range(advantages.shape[1]):
                tensorboard_writer.add_scalar(f"advantage/timestep_{t}_mean", advantages[:, t].mean(), global_step)
                tensorboard_writer.add_scalar(f"advantage/timestep_{t}_std", advantages[:, t].std(), global_step)
                tensorboard_writer.add_scalar(f"advantage/timestep_{t}_abs_mean", np.abs(advantages[:, t]).mean(), global_step)
                tensorboard_writer.add_scalar(f"advantage/timestep_{t}_abs_std", np.abs(advantages[:, t]).std(), global_step)
            # Log overall advantage statistics
            tensorboard_writer.add_scalar("advantage/overall_mean", advantages.mean(), global_step)
            tensorboard_writer.add_scalar("advantage/overall_std", advantages.std(), global_step)
            tensorboard_writer.add_scalar("advantage/overall_abs_mean", np.abs(advantages).mean(), global_step)
            tensorboard_writer.add_scalar("advantage/overall_abs_std", np.abs(advantages).std(), global_step)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())
            print("advantages per timestep:", [samples["advantages"][:, t].abs().mean().item() for t in range(samples["advantages"].shape[1])])

        del samples["rewards"]
        del samples["prompt_ids"]

        # Clean up process reward intermediate data to save memory
        del samples["delta_global"]
        del samples["d_list"]
        del samples["s_list"]
        del samples["B_list"]
        del samples["main_list"]
        del samples["process_rewards"]
        del samples["intermediate_results"]
        del samples["ode_rewards"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)

        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0 or true_count == 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process and tensorboard_writer:
            tensorboard_writer.add_scalar(f"train/actual_batch_size", mask.sum().item()//config.sample.num_batches_per_epoch, global_step)
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape

        print("check samples timesteps", samples["timesteps"].dtype)
        print("check shape", total_batch_size, num_timesteps)
        print("samples key", samples.keys())

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, config)

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        print("ratio", ratio)
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process and tensorboard_writer:
                            # Log training info to tensorboard
                            for key, value in info.items():
                                tensorboard_writer.add_scalar(f"loss/{key}", value, global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        
        epoch+=1

    # Close tensorboard writer
    if accelerator.is_main_process and tensorboard_writer:
        tensorboard_writer.close()

if __name__ == "__main__":
    app.run(main)

