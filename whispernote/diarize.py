#!/usr/bin/env python
"""
Diarize audio file using pyannote speaker diarization model.
"""

import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parent
ROOT = None
for parent in file.parents:
    if parent.name == "WhisperNote":
        ROOT = parent
sys.path.append(str(ROOT))

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

pipeline: Pipeline = None  # type: ignore


def load_model(hugging_face_key: str, threads: int = 8) -> None:
    """
    Loads model to memory. Sends model to GPU if available, falls back to CPU if not.

    Args:
        hugging_face_key (str): HuggingFace API key
        threads (int): number of threads to use

    Returns:
        None
    """
    global pipeline  # pylint: disable=global-statement
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
        torch.set_num_threads(threads)

    pipeline.to(device)


def diarize(
    audio_path: str,
    hugging_face_key: str,
    speaker_count: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    threads: int = 8,
) -> Annotation:
    """
    Diarize audio file. Returns result of diarization as Annotation object,
    use write_output to write to file.

    Args:
        audio_path (str): path to audio file
        hugging_face_key (str): HuggingFace API key
        speaker_count (int): number of speakers, if known
        min_speakers (int): minimum number of speakers, if known
        max_speakers (int): maximum number of speakers, if known
        threads (int): number of threads to use

    Returns:
        pyannote.core.Annotation: result of diarization
    """
    if pipeline is None:
        load_model(hugging_face_key, threads)

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

    Returns:
        None
    """
    with open(output, "w", encoding="utf-8") as text_file:
        for segment, _, label in diarization.itertracks(yield_label=True):  # type: ignore
            speaker = label
            start, end, _ = segment.start, segment.end, segment.duration

            start_ms = int(start * 1000)
            end_ms = int(end * 1000)

            text_file.write(f"{start_ms},{end_ms},{speaker}\n")


def get_huggingface_key():
    """Get HuggingFace API key from config file"""
    config_path = utils.get_config_file()
    params = utils.config(config_path, section="whispernote")

    if "huggingface_api_key_file" not in params:
        logger.error("HuggingFace API key file not found in config file")
        raise ValueError("HuggingFace API key file not found in config file")

    hugging_face_key_file = params["huggingface_api_key_file"]
    logger.debug(f"HuggingFace API key file: {hugging_face_key_file}")

    with open(hugging_face_key_file, "r", encoding="utf-8") as key_file:
        hugging_face_key = key_file.read().strip()

    return hugging_face_key


def main() -> None:
    """
    Diarize audio file
    """
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
        raise ValueError(
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
