#!/usr/bin/env python
from rich.console import Console

console = Console()

with console.status("[green]Loading...") as status:
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

    status.update("[bold green]Importing modules...")
    import argparse
    import concurrent.futures
    import logging
    import os
    import subprocess
    import tempfile
    from typing import Dict, List, Optional

    import pyfiglet
    from rich.logging import RichHandler

    import whispernote.helpers.utils as utils
    from whispernote import diarize, subtitle, transcribe

MODULE_NAME = "whispernote"

logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.INFO,
    # "format": "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True, level=logging.INFO)],
}
logging.basicConfig(**logargs)


def run_parallel(args, log_file: str) -> Dict[str, concurrent.futures.Future]:
    """
    Runs transcription and diarization jobs in parallel using ThreadPoolExecutor.

    Args:
        args: An object containing the command line arguments.

    Returns:
        A dictionary containing the futures of the transcription and diarization tasks.
    """
    futures: Dict[str, concurrent.futures.Future] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="whispernote"
    ) as executor:
        if args.transcript_output:
            logger.info(f"Submitting transcription job for {args.input}")

            command_array = [
                "python",
                os.path.join(utils.get_repo_root(), "WhisperNote", "whispernote", "transcribe.py"),
                "--input",
                args.input,
                "--output",
                args.transcript_output,
                "--model",
                args.transcript_model,
                "--log-file",
                log_file,
                "--condition-on-previous-text",
                str(args.condition_on_previous_text),
                "--beam-size",
                str(args.beam_size),
            ]
            if args.language:
                command_array.extend(["--language", args.language])

            transcription_task = executor.submit(
                utils.execute_commands,
                command_array,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            futures["transcription"] = transcription_task

        if args.diarization_output:
            logger.info(f"Submitting diarization job for {args.input}")
            command_array = [
                "python",
                os.path.join(utils.get_repo_root(), "WhisperNote", "whispernote", "diarize.py"),
                "--input",
                args.input,
                "--output",
                args.diarization_output,
                "--log-file",
                log_file,
            ]
            if args.speaker_count:
                command_array.extend(["--speaker-count", str(args.speaker_count)])
            if args.min_speakers:
                command_array.extend(["--min-speakers", str(args.min_speakers)])
            if args.max_speakers:
                command_array.extend(["--max-speakers", str(args.max_speakers)])

            diarization_task = executor.submit(
                utils.execute_commands,
                command_array,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            futures["diarization"] = diarization_task

        # Run the processes
        tasks_completed = 0
        for task in concurrent.futures.as_completed(futures.values()):
            tasks_completed += 1
            task_name = [k for k, v in futures.items() if v == task][0]
            logger.info(
                f"Completed {tasks_completed} of {len(futures)} tasks: {task_name}"
            )

    return futures


def run_whispernote(
    audio_input: str,
    transcript_output: str,
    diarization_output: str,
    srt_output: str,
    transcript_model: str,
    language: Optional[str] = None,
    speaker_count: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
):
    if transcript_output:
        logger.info(f"Running transcription for {audio_input}")
        transcript = transcribe.transcribe(
            input=audio_input,
            model=transcript_model,
            language=language,
        )
        transcribe.write_output(transcript_output, transcript)
        logger.info(f"Generated transcript output at {transcript_output}")

    if diarization_output:
        logger.info(f"Running diarization for {audio_input}")
        hugging_face_key = diarize.get_huggingface_key()
        diarization = diarize.diarize(
            audio_path=audio_input,
            hugging_face_key=hugging_face_key,
            speaker_count=speaker_count,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        diarize.write_output(diarization_output, diarization)
        logger.info(f"Generated Diarization output at {diarization_output}")

    if srt_output:
        logger.info(f"Generating Diarized SRT file for {audio_input}")
        subtitle_params = utils.config(utils.get_config_file(), "subtitles")
        max_words_per_line = int(subtitle_params["max_words_per_line"])
        subtitle.generate_diarized_subtitles(
            whisper_json=transcript_output,
            diarization_path=diarization_output,
            srt_path=srt_output,
            max_words_per_line=max_words_per_line,
        )
        logger.info(f"Generated Diarized SRT at {srt_output}")


def main():
    log_params = utils.config(utils.get_config_file(), "logging")
    log_file = log_params[MODULE_NAME]
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    parser = argparse.ArgumentParser(description="WhisperNote")

    parser.add_argument("--input", type=str, help="input audio file", required=True)
    parser.add_argument(
        "--parallel",
        type=bool,
        help="run processes in parallel",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--transcript-output",
        type=str,
        help="output file with transcript from Whisper (JSON)",
        required=False,
    )
    parser.add_argument(
        "--diarization-output",
        type=str,
        help="output file with speaker diarization from puannote.audio (CSV)",
        required=False,
    )
    parser.add_argument(
        "--srt-output",
        type=str,
        help="output file with SRT(subtitle) file generated from transcript, and diarization (SRT)",
        required=False,
    )
    parser.add_argument(
        "--transcribeme-output",
        type=str,
        help="output file with transcribeMe style file generated from transcript, and diarization (TXT)",
        required=False,
    )
    parser.add_argument(
        "--transcript-model",
        type=str,
        help="model to use for transcription",
        choices=["tiny", "base", "small", "medium", "large"],
        default="large",
    )
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
        "--language",
        type=str,
        help="language of the audio file, if known",
        required=False,
    )

    args = parser.parse_args()
    params = utils.config(utils.get_config_file(), "whispernote")
    args.condition_on_previous_text = params["condition_on_previous_text"]
    if (
        args.condition_on_previous_text == "True"
        or args.condition_on_previous_text == "true"
    ):
        args.condition_on_previous_text = True
    else:
        args.condition_on_previous_text = False
    args.beam_size = int(params["beam_size"])

    # Check what outputs are requested
    transcript_output = args.transcript_output
    diarization_output = args.diarization_output
    srt_output = args.srt_output

    # If no outputs are requested, then exit
    if not transcript_output and not diarization_output and not srt_output:
        logger.error("No outputs requested. Exiting.")
        logger.info("Provide at least one of the following:")
        logger.info("\t--transcript-output")
        logger.info("\t--diarization-output")
        logger.info("\t--srt-output")
        logger.info("Use --help for more information.")
        sys.exit(0)

    logger.info(f"Arguments: {args}")

    title = pyfiglet.figlet_format("WhisperNote", font="slant")
    console.print(f"[bold red]{title}")

    temp_files: List[tempfile._TemporaryFileWrapper] = []

    if srt_output:
        if not transcript_output:
            logger.info(
                "SRT output requested, but transcript output not requested. Using temp file"
            )
            temp_file = tempfile.NamedTemporaryFile(suffix=".json")
            transcript_output = temp_file.name
            logger.debug(f"Using temp file {transcript_output}")
            temp_files.append(temp_file)
        if not diarization_output:
            logger.info(
                "SRT output requested, but diarization output not requested. Using temp file"
            )
            temp_file = tempfile.NamedTemporaryFile(suffix=".csv")
            diarization_output = temp_file.name
            logger.debug(f"Using temp file {diarization_output}")
            temp_files.append(temp_file)

    if args.parallel:
        logger.info("Running transcription and diarization in parallel")
        run_parallel(args, log_file=log_file)
    else:
        if transcript_output:
            logger.info(f"Running transcription for {args.input}")
            transcript = transcribe.transcribe(
                input=args.input,
                model=args.transcript_model,
                language=args.language,
                condition_on_previous_text=args.condition_on_previous_text,
                beam_size=args.beam_size,
            )
            transcribe.write_output(transcript_output, transcript)
            logger.info(f"Generated transcript output at {transcript_output}")

        if diarization_output:
            logger.info(f"Running diarization for {args.input}")
            hugging_face_key = diarize.get_huggingface_key()
            diarization = diarize.diarize(
                audio_path=args.input,
                hugging_face_key=hugging_face_key,
                speaker_count=args.speaker_count,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
            diarize.write_output(diarization_output, diarization)
            logger.info(f"Generated Diarization output at {diarization_output}")

    if srt_output:
        logger.info(f"Generating Diarized SRT file for {args.input}")
        transcribeme_output = args.transcribeme_output
        subtitle_max_words_per_line = int(params["subtitle_max_words_per_line"])
        logger.info(f"Max words per subtitle line: {subtitle_max_words_per_line}")
        if not transcribeme_output:
            logger.info(
                "SRT output requested, but transcribeMe output not requested. Using temp file"
            )
            temp_file = tempfile.NamedTemporaryFile(suffix=".txt")
            transcribeme_output = temp_file.name
            logger.debug(f"Using temp file {transcribeme_output}")
            temp_files.append(temp_file)
        subtitle.generate_diarized_subtitles(
            whisper_json=transcript_output,
            diarization_path=diarization_output,
            srt_path=srt_output,
            transcribeMe_path=transcribeme_output,
            max_words_per_line=subtitle_max_words_per_line,
        )
        logger.info(f"Generated Diarized SRT at {srt_output}")
        logger.info(f"Generated Diarized TranscribeMe at {transcribeme_output}")

    for temp_file in temp_files:
        logger.debug(f"Deleting temp file {temp_file.name}")
        temp_file.close()


if __name__ == "__main__":
    main()
