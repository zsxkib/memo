import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from audio_separator.separator import Separator
from decord import VideoReader
from einops import rearrange
from funasr.download.download_from_hub import download_model
from funasr.models.emotion2vec.model import Emotion2vec
from insightface.app import FaceAnalysis
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

from memo.models.emotion_classifier import AudioEmotionClassifierModel
from memo.models.wav2vec import Wav2VecModel
from memo.utils.audio_utils import load_emotion2vec_model


logger = logging.getLogger("memo")
logger.setLevel(logging.INFO)


class VideoProcessor:
    def __init__(self, face_analysis, det_thresh=0.5, det_size=(640, 640)):
        self.face_analysis = FaceAnalysis(
            name="",
            root=face_analysis,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analysis.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)

    def get_face_embedding_from_video(self, input_path):
        try:
            video_reader = VideoReader(input_path.as_posix())
            assert video_reader is not None and len(video_reader) > 0, "Fail to load video frames"
            video_length = len(video_reader)

            face_emb = None
            for i in range(video_length):
                frame = video_reader[i].asnumpy()
                if face_emb is None:
                    # Detect face
                    faces = self.face_analysis.get(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    assert len(faces) == 1, "Only one face should be detected in face analysis"
                    face_emb = faces[0]["embedding"]

                    if face_emb is not None:
                        break

            if face_emb is None:
                raise ValueError("Face embedding can not be extracted")

        except Exception as e:
            logger.info(f"Error processing video {input_path}: {e}")
            if "video_reader" in locals():
                del video_reader
            return None, None, None

        del video_reader
        return face_emb, video_length


class AudioProcessor:
    def __init__(
        self,
        output_dir,
        wav2vec,
        vocal_separator,
        sample_rate,
        device,
    ):
        self.audio_encoder = Wav2VecModel.from_pretrained(wav2vec, local_files_only=True).to(device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)

        if vocal_separator is None:
            self.vocal_separator = None
        else:
            audio_separator_model_path = os.path.dirname(vocal_separator)
            audio_separator_model_name = os.path.basename(vocal_separator)
            self.vocal_separator = Separator(
                output_dir=output_dir / "vocals",
                output_single_stem="vocals",
                model_file_dir=audio_separator_model_path,
            )
            self.vocal_separator.load_model(audio_separator_model_name)
            assert self.vocal_separator.model_instance is not None, "Fail to load audio separate model."

        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.device = device

    def get_audio_embedding_from_video(self, input_path, video_length):
        if self.vocal_separator is not None:
            # extract audio form the video, and save the audio as wav
            raw_audio_path = self.output_dir / "vocals" / f"{input_path.stem}-raw.wav"
            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "2",
                str(raw_audio_path),
            ]
            subprocess.run(ffmpeg_command, check=True)

            # separate vocal from audio
            outputs = self.vocal_separator.separate(raw_audio_path)
            if len(outputs) <= 0:
                logger.info("Audio separate failed. Using raw audio.")
                speech_array, sr = librosa.load(input_path.as_posix(), sr=self.sample_rate)
            else:
                # resample the vocal to the desired sample rate
                y, sr = librosa.load(self.output_dir / "vocals" / outputs[0], sr=None)
                speech_array = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        else:
            speech_array, sr = librosa.load(input_path.as_posix(), sr=self.sample_rate)

        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=self.sample_rate).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)
        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=video_length, output_hidden_states=True)

        if len(embeddings) == 0:
            logger.info("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.cpu().detach()

        return audio_emb


def process_video(
    input_path,
    output_dir,
    video_processor: VideoProcessor,
    audio_processor: AudioProcessor,
):
    # prepare face embdeding
    face_emb, video_length = video_processor.get_face_embedding_from_video(input_path)
    if face_emb is None:
        logger.info("Fail to extract face embedding")
        return None

    # prepare audio embedding
    try:
        audio_emb = audio_processor.get_audio_embedding_from_video(input_path, video_length)
    except Exception as e:
        logger.info("Error:", e)
        logger.info(f"Fail to extract audio embedding for video {input_path}")
        return None

    if audio_emb is None:
        logger.info(f"Fail to extract audio embedding for video {input_path}")
        return None

    face_emb_path = output_dir / "face_emb" / f"{input_path.stem}.pt"
    audio_emb_path = output_dir / "audio_emb" / f"{input_path.stem}.pt"
    torch.save(face_emb, face_emb_path)
    torch.save(audio_emb, audio_emb_path)
    if video_length != audio_emb.shape[0]:
        logger.info(f"video length {video_length} != audio_emb length {audio_emb.shape}")
        return None

    return face_emb_path, audio_emb_path, video_length


def convert_audio_emb_to_vocals_path(audio_emb_path):
    """
    Convert audio embedding path to the corresponding original vocals path.
    """
    path_parts = Path(audio_emb_path).parts
    filename = path_parts[-1]
    filename_base = filename.replace(".pt", "")
    new_filename = f"{filename_base}-raw_(Vocals)_Kim_Vocal_2.wav"
    new_path = Path(*path_parts[:-2], "vocals", new_filename)
    return new_path


def main():
    parser = argparse.ArgumentParser(description="Process videos for training.")
    parser.add_argument("--input_dir", type=str, help="Directory containing videos", default=None)
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--misc_model_dir", type=str, default=None, help="Path to the face analysis model")
    parser.add_argument("--wav2vec", type=str, default="facebook/wav2vec2-base-960h", help="Path to the wav2vec model")
    parser.add_argument(
        "--emotion2vec_model", type=str, default="iic/emotion2vec_plus_large", help="Model name for emotion2vec"
    )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for audio processing")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the processing on")
    parser.add_argument("--det_size", type=tuple, default=(512, 512))
    parser.add_argument("--det_thresh", type=float, default=0.5)

    args = parser.parse_args()

    # Load the emotion classifier
    kwargs = download_model(model=args.emotion2vec_model)
    kwargs["tokenizer"] = None
    kwargs["input_size"] = None
    kwargs["frontend"] = None
    emotion_model = Emotion2vec(**kwargs, vocab_size=-1).to(args.device)
    init_param = kwargs.get("init_param", None)
    load_emotion2vec_model(
        model=emotion_model,
        path=init_param,
        ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
        oss_bucket=kwargs.get("oss_bucket", None),
        scope_map=kwargs.get("scope_map", []),
    )
    emotion_model.eval()

    emotion_classifier = AudioEmotionClassifierModel.from_pretrained(
        "memoavatar/memo",
        subfolder="misc/audio_emotion_classifier",
        use_safetensors=True,
    ).to(device=args.device)
    emotion_classifier.eval()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir_name in ["face_emb", "audio_emb", "vocals", "audio_emotion"]:
        subdir_path = output_dir / subdir_name
        subdir_path.mkdir(parents=True, exist_ok=True)

    # Initialize video and audio processors
    face_analysis = os.path.join(args.misc_model_dir, "misc/face_analysis")
    os.makedirs(face_analysis, exist_ok=True)
    for model in [
        "1k3d68.onnx",
        "2d106det.onnx",
        "face_landmarker_v2_with_blendshapes.task",
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx",
    ]:
        model_path = os.path.join(face_analysis, "models", model)
        if not os.path.exists(model_path):
            logger.info(f"Downloading {model} to {face_analysis}/models")
            os.system(
                f"wget -P {face_analysis}/models https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/{model}"
            )
            # Check if the download was successful
            if not os.path.exists(model_path):
                raise RuntimeError(f"Failed to download {model} to {model_path}")
            # File size check
            if os.path.getsize(model_path) < 1024 * 1024:
                raise RuntimeError(f"{model_path} file seems incorrect (too small), delete it and retry.")
    logger.info(f"Use face analysis models from {face_analysis}")
    video_processor = VideoProcessor(face_analysis, det_thresh=args.det_thresh, det_size=args.det_size)

    vocal_separator = os.path.join(args.misc_model_dir, "misc/vocal_separator/Kim_Vocal_2.onnx")
    if os.path.exists(vocal_separator):
        logger.info(f"Vocal separator {vocal_separator} already exists. Skipping download.")
    else:
        logger.info(f"Downloading vocal separator to {vocal_separator}")
        os.makedirs(os.path.dirname(vocal_separator), exist_ok=True)
        os.system(
            f"wget -P {os.path.dirname(vocal_separator)} https://huggingface.co/memoavatar/memo/resolve/main/misc/vocal_separator/Kim_Vocal_2.onnx"
        )
    audio_processor = AudioProcessor(
        output_dir,
        args.wav2vec,
        vocal_separator,
        sample_rate=args.sample_rate,
        device=args.device,
    )

    # Define metadata paths
    metadata_path = output_dir / "metadata.jsonl"
    processed_videos = set()

    # Check if metadata_emotion_no_mask_path exists and read the existing content
    if metadata_path.exists():
        with open(metadata_path, "r") as in_f:
            for line in in_f:
                metadata = json.loads(line)
                processed_videos.add(Path(metadata["video"]))

    # Load videos to process
    all_input_video_paths = [file for file in Path(args.input_dir).rglob("*.mp4")]
    logger.info(f"Found {len(all_input_video_paths)} videos")
    input_video_paths = all_input_video_paths

    # Process each video
    for input_path in tqdm(input_video_paths, desc="Processing videos"):
        if not input_path.exists():
            logger.info(f"Video path {input_path} does not exist")
            continue
        if input_path in processed_videos:
            logger.info(f"Video {input_path} has already been processed")
            continue

        outputs = process_video(input_path, output_dir, video_processor, audio_processor)
        if outputs is not None:
            face_emb_path, audio_emb_path, video_length = outputs

            sample_rate = args.sample_rate
            wav_path = convert_audio_emb_to_vocals_path(audio_emb_path)
            wav, sr = torchaudio.load(wav_path)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            wav = wav.view(-1) if wav.dim() == 1 else wav[0].view(-1)

            emotion_labels = torch.full_like(wav, -1, dtype=torch.int32)

            def extract_emotion(x):
                """
                Extract emotion for a given audio segment.
                """
                x = x.to(device=args.device)
                x = F.layer_norm(x, x.shape).view(1, -1)
                feats = emotion_model.extract_features(x)
                x = feats["x"].mean(dim=1)  # average across frames
                x = emotion_classifier(x)
                x = torch.softmax(x, dim=-1)
                return torch.argmax(x, dim=-1)

            # Process start, middle, and end segments
            start_label = extract_emotion(wav[: sample_rate * 2]).item()
            emotion_labels[:sample_rate] = start_label

            for i in range(sample_rate, len(wav) - sample_rate, sample_rate):
                mid_wav = wav[i - sample_rate : i - sample_rate + sample_rate * 3]
                mid_label = extract_emotion(mid_wav).item()
                emotion_labels[i : i + sample_rate] = mid_label

            end_label = extract_emotion(wav[-sample_rate * 2 :]).item()
            emotion_labels[-sample_rate:] = end_label

            # Interpolate to match the target audio length
            emotion_labels = emotion_labels.unsqueeze(0).unsqueeze(0).float()
            emotion_labels = (
                F.interpolate(emotion_labels, size=video_length, mode="nearest").squeeze(0).squeeze(0).int()
            )
            audio_emotion_path = output_dir / "audio_emotion" / f"{input_path.stem}.pt"
            torch.save(emotion_labels, audio_emotion_path)

            processed_videos.add(input_path)

            metadata = {
                "video": str(input_path),
                "face_emb": str(face_emb_path),
                "audio_emb": str(audio_emb_path),
                "audio_emotion": str(audio_emotion_path),
            }

            with open(metadata_path, "a") as f:
                f.write(json.dumps(metadata) + "\n")

    logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
