import os
import subprocess
from typing import Optional
import shutil
import torch
import tempfile
import time
from cog import BasePredictor, Input, Path
from PIL import Image

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
from pydub import AudioSegment

from memo.models.audio_proj import AudioProjModel
from memo.models.image_proj import ImageProjModel
from memo.models.unet_2d_condition import UNet2DConditionModel
from memo.models.unet_3d import UNet3DConditionModel
from memo.pipelines.video_pipeline import VideoPipeline
from memo.utils.audio_utils import (
    extract_audio_emotion_labels,
    preprocess_audio,
    resample_audio,
)
from memo.utils.vision_utils import preprocess_image, tensor_to_video

MODEL_CACHE = "checkpoints"
is_offline_mode = 1
os.environ["HF_DATASETS_OFFLINE"] = str(is_offline_mode)
os.environ["TRANSFORMERS_OFFLINE"] = str(is_offline_mode)
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

BASE_URL = f"https://weights.replicate.delivery/default/memo/{MODEL_CACHE}/"


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL:", url)
    print("[~] Destination path:", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in:", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self):
        """Load the models into memory once at startup."""
        os.makedirs(MODEL_CACHE, exist_ok=True)

        model_files = [
            "audio_proj.tar",
            "diffusion_net.tar",
            "emotion2vec_plus_large.tar",
            "image_proj.tar",
            "misc.tar",
            "reference_net.tar",
            "vae.tar",
            "wav2vec2.tar",
        ]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # Device and precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = torch.bfloat16

        # Load models
        with torch.inference_mode():
            self.vae = AutoencoderKL.from_pretrained(f"{MODEL_CACHE}/vae").to(
                device=self.device, dtype=self.weight_dtype
            )
            self.reference_net = UNet2DConditionModel.from_pretrained(
                MODEL_CACHE, subfolder="reference_net", use_safetensors=True
            )
            self.diffusion_net = UNet3DConditionModel.from_pretrained(
                MODEL_CACHE, subfolder="diffusion_net", use_safetensors=True
            )
            self.image_proj = ImageProjModel.from_pretrained(
                MODEL_CACHE, subfolder="image_proj", use_safetensors=True
            )
            self.audio_proj = AudioProjModel.from_pretrained(
                MODEL_CACHE, subfolder="audio_proj", use_safetensors=True
            )

            self.vae.requires_grad_(False).eval()
            self.reference_net.requires_grad_(False).eval()
            self.diffusion_net.requires_grad_(False).eval()
            self.image_proj.requires_grad_(False).eval()
            self.audio_proj.requires_grad_(False).eval()

            self.reference_net.enable_xformers_memory_efficient_attention()
            self.diffusion_net.enable_xformers_memory_efficient_attention()

            noise_scheduler = FlowMatchEulerDiscreteScheduler()
            self.pipeline = VideoPipeline(
                vae=self.vae,
                reference_net=self.reference_net,
                diffusion_net=self.diffusion_net,
                scheduler=noise_scheduler,
                image_proj=self.image_proj,
            )
            self.pipeline.to(device=self.device, dtype=self.weight_dtype)

    def _process_audio(self, file_path, temp_dir, max_audio_seconds):
        audio = AudioSegment.from_file(file_path)
        max_duration = max_audio_seconds * 1000
        if len(audio) > max_duration:
            audio = audio[:max_duration]
        output_path = os.path.join(temp_dir, "trimmed_audio.wav")
        audio.export(output_path, format="wav")
        return output_path

    def predict(
        self,
        image: Path = Input(description="Input image (e.g. PNG/JPG)."),
        audio: Path = Input(description="Input audio (e.g. WAV/MP3)."),
        resolution: int = Input(
            description="Resolution for generation (square). Default: 512",
            default=512,
            ge=64,
            le=2048,
        ),
        fps: int = Input(
            description="Frames per second of output video. Default: 30",
            default=30,
            ge=1,
            le=60,
        ),
        num_generated_frames_per_clip: int = Input(
            description="Frames per video clip chunk. Default: 16",
            default=16,
            ge=1,
            le=128,
        ),
        inference_steps: int = Input(
            description="Diffusion inference steps. Default: 20",
            default=20,
            ge=1,
            le=200,
        ),
        cfg_scale: float = Input(
            description="Classifier-free guidance scale. Default: 3.5",
            default=3.5,
            ge=1.0,
            le=20.0,
        ),
        max_audio_seconds: int = Input(
            description="Max audio duration (in seconds). Default: 8",
            default=8,
            ge=1,
            le=60,
        ),
        seed: int = Input(
            description="Set a random seed (None or 0 for random)",
            default=0,
        ),
    ) -> Path:
        # Handle seed
        if seed is None or seed == 0:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        generator = torch.manual_seed(seed)

        temp_dir = tempfile.mkdtemp()
        processed_audio = self._process_audio(str(audio), temp_dir, max_audio_seconds)

        # Get original aspect ratio
        img = Image.open(str(image))
        orig_w, orig_h = img.size

        # Fixed past_frames setup
        num_init_past_frames = 2
        num_past_frames = 16

        # Preprocess the image as a square
        pixel_values, face_emb = preprocess_image(
            face_analysis_model=f"{MODEL_CACHE}/misc/face_analysis",
            image_path=str(image),
            image_size=resolution,
        )

        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        cache_dir = os.path.join(output_dir, "audio_preprocess")
        os.makedirs(cache_dir, exist_ok=True)
        processed_audio = resample_audio(
            processed_audio,
            os.path.join(
                cache_dir, f"{os.path.basename(processed_audio).split('.')[0]}-16k.wav"
            ),
        )

        audio_emb, audio_length = preprocess_audio(
            wav_path=processed_audio,
            num_generated_frames_per_clip=num_generated_frames_per_clip,
            fps=fps,
            wav2vec_model=f"{MODEL_CACHE}/wav2vec2",
            vocal_separator_model=f"{MODEL_CACHE}/misc/vocal_separator/Kim_Vocal_2.onnx",
            cache_dir=cache_dir,
            device=self.device,
        )

        audio_emotion, num_emotion_classes = extract_audio_emotion_labels(
            model=MODEL_CACHE,
            wav_path=processed_audio,
            emotion2vec_model=f"{MODEL_CACHE}/emotion2vec_plus_large",
            audio_length=audio_length,
            device=self.device,
        )

        video_frames = []
        num_clips = audio_emb.shape[0] // num_generated_frames_per_clip
        for t in tqdm(range(num_clips), desc="Generating video clips"):
            if len(video_frames) == 0:
                past_frames = pixel_values.repeat(num_init_past_frames, 1, 1, 1).to(
                    dtype=pixel_values.dtype, device=pixel_values.device
                )
                pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)
            else:
                past_frames = video_frames[-1][0].permute(1, 0, 2, 3)
                past_frames = past_frames[-num_past_frames:]
                past_frames = past_frames * 2.0 - 1.0
                past_frames = past_frames.to(
                    dtype=pixel_values.dtype, device=pixel_values.device
                )
                pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)

            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
            audio_tensor = (
                audio_emb[
                    t
                    * num_generated_frames_per_clip : (t + 1)
                    * num_generated_frames_per_clip
                ]
                .unsqueeze(0)
                .to(self.audio_proj.device, dtype=self.audio_proj.dtype)
            )
            audio_tensor = self.audio_proj(audio_tensor)
            audio_emotion_tensor = audio_emotion[
                t
                * num_generated_frames_per_clip : (t + 1)
                * num_generated_frames_per_clip
            ]

            pipeline_output = self.pipeline(
                ref_image=pixel_values_ref_img,
                audio_tensor=audio_tensor,
                audio_emotion=audio_emotion_tensor,
                emotion_class_num=num_emotion_classes,
                face_emb=face_emb,
                width=resolution,
                height=resolution,
                video_length=num_generated_frames_per_clip,
                num_inference_steps=inference_steps,
                guidance_scale=cfg_scale,
                generator=generator,
            )
            video_frames.append(pipeline_output.videos)

        video_frames = torch.cat(video_frames, dim=2).squeeze(0)
        video_frames = video_frames[:, :audio_length]

        # Save the square video
        video_path = os.path.join(output_dir, "memo.mp4")
        tensor_to_video(video_frames, video_path, processed_audio, fps=fps)

        # Rescale the video to match the original aspect ratio
        target_width = int((orig_w / orig_h) * resolution)
        final_video_path = os.path.join(output_dir, "memo_rescaled.mp4")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vf",
                f"scale={target_width}:{resolution}",
                final_video_path,
            ],
            check=True,
        )

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        return Path(final_video_path)
