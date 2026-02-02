import ml_collections
import imp
import os


base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config



def pickscore_flux_fast():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore_sfw")

    # flux
    config.pretrained.model = "/mnt/workspace/tyz/A_MODELS/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5
    config.sample.eval_guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = False
    config.sample.same_latent = False
    config.sample.noise_level = 0.8
    config.sample.sde_window_size = 5
    config.sample.sde_window_range = (0, 5)
    config.sample.sde_type = "cps"
    config.train.ema = True
    config.mixed_precision = "bf16"
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.save_dir = '/mnt/workspace/tyz/EXP/EXP_6step_flux_pickscore'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_flux_fast_op3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore_sfw")

    # flux
    config.pretrained.model = "/mnt/workspace/tyz/A_MODELS/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5
    config.sample.eval_guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = False
    config.sample.same_latent = False
    config.sample.noise_level = 0.8
    config.sample.sde_window_size = 5
    config.sample.sde_window_range = (0, 5)
    config.sample.sde_type = "cps"
    config.train.ema = True
    config.mixed_precision = "bf16"
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.save_dir = '/mnt/workspace/tyz/EXP/EXP_6step_flux_pickscore'
    config.reward_fn = {
        "pickscore": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True

    # hyper-parameters
    config.apply_first_step_bonus = True # 是否在第一步也施加bonus
    config.use_delta_global = True

    config.take_delta_global_as_main = True
    config.eta = 1.0 # 对于t和t+1步reward之差远小于full sde和ode之差的情况，适当放大弱势项

    config.turn_on_bonus_global_step = 0 # 标准情况下，开始训练后直接开启bonus，不为0时则代表推迟计算bonus
    config.select_first_step_only_from_consistent_trajectory = False # 在第一步施加额外reward的时候，仅挑选consistent trajectory, sign(r_{T-1}-r_T)和sign(r_\text{sde} > r_\text{T-1})同号
    config.select_only_positive_intermediate_step = False

    config.use_balanced_bonus = False

    config.select_inter_step_only_from_consistent_trajectory = False # 对中间去噪步骤，是否只选择consistent trajectory
    config.use_sde_minus_rt = False # 是否用r_sde - r_t代替delta_global

    config.drop_original_reward_when_having_bonus = False
    config.propagate_bonus_to_future_steps = False # 是否将bonus传播到后续时间步：当某个时间步的indicator为true时，该轨迹在这个时间步之后的所有时间步也都获得相同的bonus
    config.indicator_wo_delta_global_constraint = False
    return config


def geneval_flux_fast():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    config.pretrained.model = "/mnt/workspace/tyz/A_MODELS/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5
    config.sample.eval_guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = False # TODO：group算advantage的时候是否是全局在算的，我们之前都是用的False，但是12.13发现默认配置几乎都用True，不过考虑到sde-ode还是用false合理
    config.sample.same_latent = False
    config.sample.noise_level = 0.8
    config.sample.sde_window_size = 6
    config.sample.sde_window_range = (0, 6)
    config.sample.sde_type = "cps"
    config.train.ema = True
    config.mixed_precision = "bf16"
    config.save_freq = 10 # epoch
    config.eval_freq = 10
    config.save_dir = '/mnt/workspace/tyz/EXP/EXP_6step_flux_geneval'
    config.reward_fn = {
        "geneval_local": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def geneval_flux_fast_op3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    config.pretrained.model = "/mnt/workspace/tyz/A_MODELS/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5
    config.sample.eval_guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = False # TODO：group算advantage的时候是否是全局在算的，我们之前都是用的False，但是12.13发现默认配置几乎都用True，不过考虑到sde-ode还是用false合理
    config.sample.same_latent = False
    config.sample.noise_level = 0.8
    config.sample.sde_window_size = 6
    config.sample.sde_window_range = (0, 6)
    config.sample.sde_type = "cps"
    config.train.ema = True
    config.mixed_precision = "bf16"
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.save_dir = '/mnt/workspace/tyz/EXP/EXP_6step_flux_geneval'
    config.reward_fn = {
        "geneval_local": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True

    # hyper-parameters
    config.apply_first_step_bonus = True # 是否在第一步也施加bonus
    config.use_delta_global = True

    config.take_delta_global_as_main = True
    config.eta = 1.0 # 对于t和t+1步reward之差远小于full sde和ode之差的情况，适当放大弱势项

    config.turn_on_bonus_global_step = 0 # 标准情况下，开始训练后直接开启bonus，不为0时则代表推迟计算bonus
    config.select_first_step_only_from_consistent_trajectory = False # 在第一步施加额外reward的时候，仅挑选consistent trajectory, sign(r_{T-1}-r_T)和sign(r_\text{sde} > r_\text{T-1})同号
    config.select_only_positive_intermediate_step = False

    config.use_balanced_bonus = False

    config.select_inter_step_only_from_consistent_trajectory = False # 对中间去噪步骤，是否只选择consistent trajectory
    config.use_sde_minus_rt = False # 是否用r_sde - r_t代替delta_global

    config.drop_original_reward_when_having_bonus = False
    config.propagate_bonus_to_future_steps = False # 是否将bonus传播到后续时间步：当某个时间步的indicator为true时，该轨迹在这个时间步之后的所有时间步也都获得相同的bonus
    config.indicator_wo_delta_global_constraint = False
    return config


def pickscore_sd3_fast():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore_sfw")

    # sd3.5 medium
    config.pretrained.model = "/mnt/workspace/tyz/A_MODELS/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.eval_guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    # config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = False
    config.sample.same_latent = False
    config.sample.sde_window_size = 9
    config.sample.sde_window_range = (0, 9)
    config.sample.sde_type = "cps"
    config.mixed_precision = "fp16"
    # add
    config.sample.noise_level = 0.7
    config.train.ema = True
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.save_dir = '/mnt/workspace/tyz/EXP/EXP_10step_sd3_pickscore'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_fast_op3():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore_sfw")

    # sd3.5 medium
    config.pretrained.model = "/mnt/workspace/tyz/A_MODELS/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.eval_guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    # config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = False
    config.sample.same_latent = False
    config.sample.sde_window_size = 9
    config.sample.sde_window_range = (0, 9)
    config.sample.sde_type = "cps"
    config.mixed_precision = "fp16"
    # add
    config.sample.noise_level = 0.7
    config.train.ema = True
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.save_dir = '/mnt/workspace/tyz/EXP/EXP_10step_sd3_pickscore'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True

    # hyper-parameters
    config.apply_first_step_bonus = True # 是否在第一步也施加bonus
    config.use_delta_global = True

    config.take_delta_global_as_main = True
    config.eta = 1.0 # 对于t和t+1步reward之差远小于full sde和ode之差的情况，适当放大弱势项

    config.turn_on_bonus_global_step = 0 # 标准情况下，开始训练后直接开启bonus，不为0时则代表推迟计算bonus
    config.select_first_step_only_from_consistent_trajectory = False # 在第一步施加额外reward的时候，仅挑选consistent trajectory, sign(r_{T-1}-r_T)和sign(r_\text{sde} > r_\text{T-1})同号
    config.select_only_positive_intermediate_step = False

    config.use_balanced_bonus = False

    config.select_inter_step_only_from_consistent_trajectory = False # 对中间去噪步骤，是否只选择consistent trajectory
    config.use_sde_minus_rt = False # 是否用r_sde - r_t代替delta_global

    config.drop_original_reward_when_having_bonus = False
    config.propagate_bonus_to_future_steps = False # 是否将bonus传播到后续时间步：当某个时间步的indicator为true时，该轨迹在这个时间步之后的所有时间步也都获得相同的bonus
    config.indicator_wo_delta_global_constraint = False
    return config


def geneval_sd3_fast():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "/mnt/workspace/tyz/A_MODELS/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.eval_guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-5
    # config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = False
    config.sample.same_latent = False
    config.sample.noise_level = 0.7
    config.sample.sde_window_size = 9
    config.sample.sde_window_range = (0, 9)
    config.sample.sde_type = "cps"
    config.train.ema = True
    config.mixed_precision = "fp16"
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.save_dir = f'/mnt/workspace/tyz/EXP/EXP_10step_sd3_geneval'
    config.reward_fn = {
        "geneval_local": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3_fast_op3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "/mnt/workspace/tyz/A_MODELS/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.eval_guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-5
    # config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = False
    config.sample.same_latent = False
    config.sample.noise_level = 0.7
    config.sample.sde_window_size = 9
    config.sample.sde_window_range = (0, 9)
    config.sample.sde_type = "cps"
    config.train.ema = True
    config.mixed_precision = "fp16"
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.save_dir = f'/mnt/workspace/tyz/EXP/EXP_10step_sd3_geneval'
    config.reward_fn = {
        "geneval_local": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True

    # hyper-parameters
    config.apply_first_step_bonus = True # 是否在第一步也施加bonus
    config.use_delta_global = True

    config.take_delta_global_as_main = True
    config.eta = 1.0 # 对于t和t+1步reward之差远小于full sde和ode之差的情况，适当放大弱势项

    config.turn_on_bonus_global_step = 0 # 标准情况下，开始训练后直接开启bonus，不为0时则代表推迟计算bonus
    config.select_first_step_only_from_consistent_trajectory = False # 在第一步施加额外reward的时候，仅挑选consistent trajectory, sign(r_{T-1}-r_T)和sign(r_\text{sde} > r_\text{T-1})同号
    config.select_only_positive_intermediate_step = False

    config.use_balanced_bonus = False

    config.select_inter_step_only_from_consistent_trajectory = False # 对中间去噪步骤，是否只选择consistent trajectory
    config.use_sde_minus_rt = False # 是否用r_sde - r_t代替delta_global

    config.drop_original_reward_when_having_bonus = False
    config.propagate_bonus_to_future_steps = False # 是否将bonus传播到后续时间步：当某个时间步的indicator为true时，该轨迹在这个时间步之后的所有时间步也都获得相同的bonus
    config.indicator_wo_delta_global_constraint = False

    return config


def general_ocr_sd3_fast():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "/mnt/workspace/tyz/A_MODELS/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.eval_guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = False
    config.sample.same_latent = False
    config.sample.noise_level = 0.7
    config.sample.sde_window_size = 9
    config.sample.sde_window_range = (0, 9)
    config.sample.sde_type = "cps"
    config.train.ema = True
    config.mixed_precision = "fp16"
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.save_dir = '/mnt/workspace/tyz/EXP/EXP_10step_sd3_ocr'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3_fast_op3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "/mnt/workspace/tyz/A_MODELS/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.eval_guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = False
    config.sample.same_latent = False
    config.sample.noise_level = 0.7
    config.sample.sde_window_size = 9
    config.sample.sde_window_range = (0, 9)
    config.sample.sde_type = "cps"
    config.train.ema = True
    config.mixed_precision = "fp16"
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.save_dir = '/mnt/workspace/tyz/EXP/EXP_10step_sd3_ocr'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True

    # hyper-parameters
    config.apply_first_step_bonus = True # 是否在第一步也施加bonus
    config.use_delta_global = True

    config.take_delta_global_as_main = True
    config.eta = 1.0 # 对于t和t+1步reward之差远小于full sde和ode之差的情况，适当放大弱势项

    config.turn_on_bonus_global_step = 0 # 标准情况下，开始训练后直接开启bonus，不为0时则代表推迟计算bonus
    config.select_first_step_only_from_consistent_trajectory = False # 在第一步施加额外reward的时候，仅挑选consistent trajectory, sign(r_{T-1}-r_T)和sign(r_\text{sde} > r_\text{T-1})同号
    config.select_only_positive_intermediate_step = False

    config.use_balanced_bonus = False

    config.select_inter_step_only_from_consistent_trajectory = False # 对中间去噪步骤，是否只选择consistent trajectory
    config.use_sde_minus_rt = False # 是否用r_sde - r_t代替delta_global

    config.drop_original_reward_when_having_bonus = False
    config.propagate_bonus_to_future_steps = False # 是否将bonus传播到后续时间步：当某个时间步的indicator为true时，该轨迹在这个时间步之后的所有时间步也都获得相同的bonus
    config.indicator_wo_delta_global_constraint = False
    return config



def get_config(name):
    return globals()[name]()
