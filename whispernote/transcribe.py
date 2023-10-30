#!/usr/bin/env python

import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parent
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

import pyfiglet
import whisper
from whispernote.helpers import ffprobe
from rich.console import Console
from rich.logging import RichHandler

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


def transcribe(
    input: str,
    language: str = None,
    model: str = "base",
    word_timestamps: bool = True,
    load_model_in_memory: bool = True,
) -> None:
    """Transcribe audio file with Whisper

    Args:
        input (str): input audio file
        language (str, optional): language of the audio file, if known. Defaults to None.
        model (str, optional): model to use for transcription. Defaults to "base".
        word_timestamps (bool, optional): include word timestamps in output. Defaults to False.
    """
    logger.info(f"Transcribing {input} with Whisper")

    logger.debug(f"Loading model: '{model}'")
    model: whisper.Whisper = whisper.load_model(model, in_memory=load_model_in_memory)

    logger.info("Starting transcription")
    logger.debug(f"language: '{language}'")
    logger.debug(f"word_timestamps: {word_timestamps}")
    result = model.transcribe(
        input, language=language, word_timestamps=word_timestamps, verbose=False
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

    args = parser.parse_args()

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
    )

    write_output(args.output, results)

    sys.exit(0)


if __name__ == "__main__":
    main()
