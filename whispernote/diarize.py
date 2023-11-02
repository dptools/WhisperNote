#!/usr/bin/env python

import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parent
root = None
for parent in file.parents:
    if parent.name == "WhisperNote":
        root = parent
sys.path.append(str(root))

# remove current directory from path
try:
    sys.path.remove(str(parent))
except ValueError:
    pass

import argparse
import logging
from typing import Optional

import pyfiglet
import torchaudio
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core.annotation import Annotation
from rich.console import Console
from rich.logging import RichHandler

import whispernote.helpers.utils as utils

MODULE_NAME = "diarize"

logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.INFO,
    # "format": "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)

console = Console()


def diarize(
    audio_path: str,
    hugging_face_key: str,
    speaker_count: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Annotation:
    logger.info("Loading speaker diarization model from HuggingFace")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=hugging_face_key
    )

    # send pipeline to GPU (when available)
    if torch.cuda.is_available():
        gpu_idx = utils.get_free_gpu_idx()
        device = torch.device(f"cuda:{gpu_idx}")
        logger.info(f"Sending pipeline to GPU {gpu_idx}")
    else:
        device = torch.device("cpu")
        logger.info("Sending pipeline to CPU")

    pipeline.to(device)

    logger.info("Loading audio file")
    waveform, sample_rate = torchaudio.load(audio_path)  # type: ignore

    logger.info("Diarizing audio file")
    with ProgressHook() as hook:
        if speaker_count:
            logger.debug(f"Speaker count: {speaker_count}")
            diarization: Annotation = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                hook=hook,
                num_speakers=speaker_count,
            )
        elif min_speakers and max_speakers:
            logger.debug(f"Speaker count range: {min_speakers} - {max_speakers}")
            diarization = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                hook=hook,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        else:
            logger.debug("No speaker count or speaker count range specified")
            diarization: Annotation = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate}, hook=hook
            )

    logger.info("Diarization complete")
    return diarization


def write_output(output: str, diarization: Annotation) -> None:
    """Write output to file

    Args:
        output (str): output file
        diarization (pyannote.core.SlidingWindowFeature): result of diarization
    """
    with open(output, "w") as text_file:
        for segment, _, label in diarization.itertracks(yield_label=True):  # type: ignore
            speaker = label
            start, end, _ = segment.start, segment.end, segment.duration

            start_ms = int(start * 1000)
            end_ms = int(end * 1000)

            text_file.write(f"{start_ms},{end_ms},{speaker}\n")


def get_huggingface_key():
    """Get HuggingFace API key from config file"""
    config_path = utils.get_config_file()
    params = utils.config(config_path, section="huggingface")

    if "api_key" not in params:
        logger.error("HuggingFace API key not found in config file")
        raise Exception("HuggingFace API key not found in config file")

    return params["api_key"]


def main():
    parser = argparse.ArgumentParser(description="Diarize audio file")

    parser.add_argument("--input", type=str, help="input audio file", required=True)
    parser.add_argument("--output", type=str, help="output file", required=True)
    parser.add_argument(
        "--speaker-count", type=int, help="number of speakers, if known", required=False
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="minimum number of speakers, if known",
        required=False,
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="maximum number of speakers, if known",
        required=False,
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="log file",
        required=False,
    )

    args = parser.parse_args()

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    title = pyfiglet.figlet_format("WhisperNote", font="slant")
    console.print(f"[bold red]{title}")
    console.rule("[bold red]Diarization")

    has_speaker_count = args.speaker_count is not None
    if has_speaker_count:
        logger.info(f"Speaker count: {args.speaker_count}")

    has_speaker_count_range = (
        args.min_speakers is not None and args.max_speakers is not None
    )
    if has_speaker_count_range:
        logger.info(f"Speaker count range: {args.min_speakers} - {args.max_speakers}")

    if has_speaker_count and has_speaker_count_range:
        raise Exception(
            "Speaker count and speaker count range cannot both be specified"
        )

    logger.info("Loading HuggingFace API key from config file")
    hugging_face_key = get_huggingface_key()

    diarization = diarize(
        audio_path=args.input,
        hugging_face_key=hugging_face_key,
        speaker_count=args.speaker_count,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    logger.info("Writing output to file")
    write_output(args.output, diarization)


if __name__ == "__main__":
    main()
