import argparse
import copy
import logging
import math
import os
import random
import shutil
import time
import warnings

import accelerate
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from omegaconf import OmegaConf
from packaging import version
from torch import nn
from tqdm.auto import tqdm

from memo.datasets.video_dataset import VideoDataset
from memo.models.audio_proj import AudioProjModel
from memo.models.image_proj import ImageProjModel
from memo.models.unet_2d_condition import UNet2DConditionModel
from memo.models.unet_3d import UNet3DConditionModel


warnings.filterwarnings("ignore")

logger = get_logger(__name__, log_level="INFO")


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


class MEMOModel(nn.Module):
    def __init__(
        self,
        reference_net: UNet2DConditionModel,
        diffusion_net: UNet3DConditionModel,
        image_proj,
        audio_proj,
    ):
        super().__init__()
        self.reference_net = reference_net
        self.diffusion_net = diffusion_net
        self.image_proj = image_proj
        self.audio_proj = audio_proj

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        ref_image_latents: torch.Tensor,
        face_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        audio_emotion: torch.Tensor = None,
        uncond_img_fwd: bool = False,
        uncond_audio_fwd: bool = False,
    ):
        face_emb = self.image_proj(face_emb)

        if not uncond_audio_fwd:
            audio_emb = audio_emb.to(device=self.audio_proj.device, dtype=self.audio_proj.dtype)
            audio_emb = self.audio_proj(audio_emb)
        else:
            audio_emb = torch.zeros_like(audio_emb).to(device=audio_emb.device, dtype=audio_emb.dtype)
            audio_emb = self.audio_proj(audio_emb)

        # condition forward
        ref_timesteps = torch.zeros_like(timesteps[0] if isinstance(timesteps, list) else timesteps)
        ref_timesteps = repeat(
            ref_timesteps,
            "b -> (repeat b)",
            repeat=ref_image_latents.size(0) // ref_timesteps.size(0),
        )
        if not uncond_img_fwd:
            ref_features = self.reference_net(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=face_emb,
                return_dict=False,
            )
        else:
            ref_features = self.reference_net(
                torch.zeros_like(ref_image_latents),
                ref_timesteps,
                encoder_hidden_states=face_emb,
                return_dict=False,
            )

        model_pred = self.diffusion_net(
            noisy_latents,
            ref_features,
            timesteps,
            encoder_hidden_states=face_emb,
            audio_embedding=audio_emb,
            audio_emotion=audio_emotion,
        ).sample

        return model_pred


def main():
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir, logging_dir=os.path.join(config.output_dir, "logs")
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if config.seed is not None:
        set_seed(config.seed)

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))

    # Create noise scheduler
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    train_noise_scheduler = FlowMatchEulerDiscreteScheduler(**sched_kwargs)
    noise_scheduler_copy = copy.deepcopy(train_noise_scheduler)

    logger.info("Loading models")
    vae = AutoencoderKL.from_pretrained(config.vae)
    reference_net = UNet2DConditionModel.from_pretrained(
        config.model_name_or_path, subfolder="reference_net", use_safetensors=True
    )
    diffusion_net = UNet3DConditionModel.from_pretrained(
        config.model_name_or_path, subfolder="diffusion_net", use_safetensors=True
    )
    image_proj = ImageProjModel.from_pretrained(
        config.model_name_or_path, subfolder="image_proj", use_safetensors=True
    )
    audio_proj = AudioProjModel.from_pretrained(
        config.model_name_or_path, subfolder="audio_proj", use_safetensors=True
    )

    # Set trainable parameters
    if config.get("train_diffusion_net", False):
        trainable_modules = config.get("trainable_modules", None)
        logger.info(f"Trainable modules: {trainable_modules}")
        if trainable_modules is not None:
            diffusion_net.requires_grad_(False)
            for name, module in diffusion_net.named_modules():
                if any(trainable_mod in name for trainable_mod in trainable_modules):
                    for params in module.parameters():
                        params.requires_grad_(True)
        else:
            diffusion_net.requires_grad_(True)
    else:
        diffusion_net.requires_grad_(False)

    if config.get("train_reference_net", False):
        trainable_modules = config.get("trainable_modules", None)
        if trainable_modules is not None:
            reference_net.requires_grad_(False)
            for name, module in reference_net.named_modules():
                if any(trainable_mod in name for trainable_mod in trainable_modules):
                    for params in module.parameters():
                        params.requires_grad_(True)
        else:
            reference_net.requires_grad_(True)
    else:
        reference_net.requires_grad_(False)

    vae.requires_grad_(False)
    image_proj.requires_grad_(config.get("train_image_proj", False))
    audio_proj.requires_grad_(config.get("train_audio_proj", False))

    # For mixed precision training we cast all non-trainable weights (vae, non-lora image_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        config.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        config.mixed_precision = accelerator.mixed_precision

    model = MEMOModel(
        reference_net,
        diffusion_net,
        image_proj,
        audio_proj,
    ).to(dtype=weight_dtype)
    model.train()

    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            reference_net.enable_xformers_memory_efficient_attention()
            diffusion_net.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    model.reference_net.save_pretrained(os.path.join(output_dir, "reference_net"))
                    model.diffusion_net.save_pretrained(os.path.join(output_dir, "diffusion_net"))
                    model.image_proj.save_pretrained(os.path.join(output_dir, "image_proj"))
                    model.audio_proj.save_pretrained(os.path.join(output_dir, "audio_proj"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_reference_net = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="reference_net", use_safetensors=True
                )
                load_reference_net.register_to_config(**model.reference_net.config)
                load_diffusion_net = UNet3DConditionModel.from_pretrained(
                    input_dir, subfolder="diffusion_net", use_safetensors=True
                )
                load_diffusion_net.register_to_config(**model.diffusion_net.config)
                load_image_proj = ImageProjModel.from_pretrained(
                    input_dir, subfolder="image_proj", use_safetensors=True
                )
                load_image_proj.register_to_config(**model.image_proj.config)
                load_audio_proj = AudioProjModel.from_pretrained(
                    input_dir, subfolder="audio_proj", use_safetensors=True
                )
                load_audio_proj.register_to_config(**model.audio_proj.config)

                load_model = MEMOModel(
                    load_reference_net,
                    load_diffusion_net,
                    load_image_proj,
                    load_audio_proj,
                ).to(dtype=weight_dtype)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if config.gradient_checkpointing:
        reference_net.enable_gradient_checkpointing()
        diffusion_net.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate
            * config.gradient_accumulation_steps
            * config.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    logger.info(f"Total trainable params number: {len(trainable_params)}")
    logger.info(f"Total trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")
    optimizer = optimizer_cls(
        trainable_params,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    train_dataset = VideoDataset(
        num_past_frames=config.data.num_past_frames,
        n_sample_frames=config.data.n_sample_frames,
        img_size=(config.data.width, config.data.height),
        audio_margin=config.data.audio_margin,
        metadata_paths=config.data.metadata_paths,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Move the vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    image_proj.to(accelerator.device, dtype=weight_dtype)
    audio_proj.to(accelerator.device, dtype=weight_dtype)
    reference_net.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = OmegaConf.to_container(config, resolve=True)
        accelerator.init_trackers(config.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from {config.resume_from_checkpoint}")
        accelerator.load_state(config.resume_from_checkpoint)
        path = os.path.basename(config.resume_from_checkpoint)
        global_step = int(path.split("-")[1])
        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, config.num_train_epochs):
        train_loss = 0.0
        accelerator.log({"epoch": epoch}, step=global_step)
        t_data_start = time.time()
        for batch in train_dataloader:
            t_data = time.time() - t_data_start

            with accelerator.accumulate(model):
                # Convert videos to latent space
                with torch.no_grad():
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                    video_length = pixel_values.shape[1]
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                    latents = latents * vae.config.scaling_factor

                    noise = torch.randn_like(latents)

                    # Sample a random timestep for each video
                    def get_timesteps(bsz=None):
                        bsz = bsz or latents.shape[0]
                        u = compute_density_for_timestep_sampling(
                            weighting_scheme=config.weighting_scheme,
                            batch_size=bsz,
                            logit_mean=config.logit_mean,
                            logit_std=config.logit_std,
                            mode_scale=config.mode_scale,
                        )
                        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                        timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)
                        return timesteps

                    timesteps = get_timesteps()

                    start_frame = random.random() < config.start_ratio
                    pixel_values_ref_img = batch["pixel_values_ref_img"].to(dtype=weight_dtype)
                    # Initialize the motion frames as zero maps
                    if start_frame:
                        pixel_values_ref_img[:, 1:] = 0.0

                    ref_img_and_motion = rearrange(pixel_values_ref_img, "b f c h w -> (b f) c h w")
                    ref_image_latents = vae.encode(ref_img_and_motion).latent_dist.sample()
                    ref_image_latents = ref_image_latents * vae.config.scaling_factor
                    image_prompt_embeds = batch["face_emb"].to(dtype=weight_dtype, device=image_proj.device)

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    # Add noise according to flow matching.
                    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                    noisy_latents = sigmas * noise + (1.0 - sigmas) * latents

                    # Predict the noise residual and compute loss
                    uncond_img_fwd = random.random() < config.uncond_img_ratio
                    uncond_audio_fwd = random.random() < config.uncond_audio_ratio

                # ---- Forward!!! -----
                model_pred = model(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    ref_image_latents=ref_image_latents,
                    face_emb=image_prompt_embeds,
                    audio_emb=batch["audio_emb"].to(dtype=weight_dtype),
                    audio_emotion=batch["audio_emotion"],
                    uncond_img_fwd=uncond_img_fwd,
                    uncond_audio_fwd=uncond_audio_fwd,
                )

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                model_pred = model_pred * (-sigmas) + noisy_latents

                # These weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                mse_loss_weights = compute_loss_weighting_for_sd3(
                    weighting_scheme=config.weighting_scheme, sigmas=sigmas
                )

                # Flow matching loss
                target = latents
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights

                # Robust training
                if config.get("robust_training", False):
                    mask = loss <= 0.1
                    num_valid = mask.sum()
                    loss = (loss * mask).sum() / torch.clamp(num_valid, min=1).float()
                else:
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "td": f"{t_data:.2f}s",
                }
                t_data_start = time.time()
                progress_bar.set_postfix(**logs)

                if global_step % config.checkpointing_steps == 0 or global_step == config.max_train_steps:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            if global_step >= config.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.output_dir = os.path.join(config.output_dir, args.exp_name)

    main()
