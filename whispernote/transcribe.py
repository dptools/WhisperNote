#!/usr/bin/env python

import sys
from pathlib import Path
from typing import Optional

import torch

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
import json
import logging
from typing import Any, Dict

import pyfiglet
import whisper
from whispernote.helpers import ffprobe
from rich.console import Console
from rich.logging import RichHandler

import whispernote.helpers.utils as utils

MODULE_NAME = "transcribe"

logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.INFO,
    # "format": "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)

console = Console()

whisper_model: whisper.Whisper = None  # type: ignore


def load_model(
    model: str, in_memory: bool = True, force_cpu: bool = False, threads: int = 8
):
    global whisper_model
    if force_cpu:
        device = "cpu"
        logger.info("Sending transcription model to CPU (Overridden)")
        torch.set_num_threads(threads)
    else:
        if utils.check_gpu():
            gpu_idx = utils.get_free_gpu_idx()
            device = f"cuda:{gpu_idx}"
            logger.info(f"Sending transcription model to GPU {gpu_idx}")
        else:
            device = "cpu"
            logger.info("Sending transcription model to CPU")
            torch.set_num_threads(threads)
    logger.info(f"Loading transcription model: '{model}'")
    whisper_model = whisper.load_model(model, in_memory=in_memory, device=device)


def transcribe(
    input: str,
    language: Optional[str] = None,
    model: str = "base",
    beam_size: Optional[int] = None,
    condition_on_previous_text: bool = True,
    word_timestamps: bool = True,
    load_model_in_memory: bool = True,
    force_cpu: bool = False,
    threads: int = 8,
) -> Dict[str, Any]:
    """Transcribe audio file with Whisper

    Args:
        input (str): input audio file
        language (str, optional): language of the audio file, if known. Defaults to None.
        model (str, optional): model to use for transcription. Defaults to "base".
        word_timestamps (bool, optional): include word timestamps in output. Defaults to False.
    """
    logger.info(f"Transcribing {input} with Whisper")

    logger.info("Starting transcription")
    logger.debug(f"language: '{language}'")
    logger.debug(f"word_timestamps: {word_timestamps}")

    global whisper_model
    if whisper_model is None:
        load_model(model, in_memory=load_model_in_memory, threads=threads, force_cpu=force_cpu)
    result = whisper_model.transcribe(
        input,
        language=language,
        word_timestamps=word_timestamps,
        verbose=False,
        beam_size=beam_size,
        condition_on_previous_text=condition_on_previous_text,
    )

    logger.info("Transcription complete")
    return result


def write_output(output: str, result: dict) -> None:
    """Write output to file

    Args:
        output (str): output file
        result (dict): result of transcription
    """
    logger.info(f"Writing output to {output}")
    with open(output, "w") as output_file:
        json.dump(result, output_file, indent=4)


def print_duration(input_file: str):
    """
    Prints the duration of an input audio file.

    Args:
        input_file (str): The path to the input audio file.

    Returns:
        None
    """
    metadata = ffprobe.get_metadata(input_file)
    duration = metadata["format"]["duration"]
    duration = float(duration)
    logger.debug(f"Duration of input file is {duration:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio file with Whisper")

    parser.add_argument("--input", type=str, help="input audio file", required=True)
    parser.add_argument("--output", type=str, help="output file", required=True)
    parser.add_argument(
        "--language", type=str, help="language of the audio file, if known"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model to use for transcription",
        choices=["tiny", "base", "small", "medium", "large"],
        default="large",
    )
    parser.add_argument(
        "--word-timestamps",
        type=bool,
        help="include word timestamps in output",
        default=True,
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="log file",
        required=False,
    )
    parser.add_argument(
        "--condition-on-previous-text",
        type=bool,
        help="condition on previous text. False helps prevent the model from repeating itself.",
        default=True,
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        help="beam size",
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

    input_file = args.input

    title = pyfiglet.figlet_format("WhisperNote", font="slant")
    console.print(f"[bold red]{title}")
    console.rule("[bold red]Trascription")

    # Check if input file exists
    if not Path(input_file).is_file():
        logger.error(f"Input file {input_file} does not exist")
        raise FileNotFoundError(f"Input file {input_file} does not exist")

    # Print duration of input file
    print_duration(input_file)

    results = transcribe(
        input=args.input,
        language=args.language,
        model=args.model,
        word_timestamps=args.word_timestamps,
        condition_on_previous_text=args.condition_on_previous_text,
        beam_size=args.beam_size,
    )

    write_output(args.output, results)

    sys.exit(0)


if __name__ == "__main__":
    main()
