# init file for module

import logging
import multiprocessing
import time
from typing import List, Optional

from rich.logging import RichHandler

from . import diarize, subtitle, transcribe
from .helpers import utils

MODULE_NAME = "transcribe"

logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.INFO,
    # "format": "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)


def run_whispernote_parallel(
    audio_input: str,
    transcript_output: str,
    diarization_output: str,
    srt_output: str,
    transcript_model: str,
    language: Optional[str] = None,
    speaker_count: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    transcribeMe_path: Optional[str] = None,
):
    params = utils.config(utils.get_config_file(), "whispernote")

    def transcribe_wrapper(input, model, language, transcript_output):
        transcript = transcribe.transcribe(
            input=input,
            model=model,
            language=language,
        )
        transcribe.write_output(transcript_output, transcript)

    def diarize_wrapper(
        audio_path,
        hugging_face_key,
        speaker_count,
        min_speakers,
        max_speakers,
        diarization_output,
    ):
        diarization = diarize.diarize(
            audio_path=audio_path,
            hugging_face_key=hugging_face_key,
            speaker_count=speaker_count,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        diarize.write_output(diarization_output, diarization)

    processes: List[multiprocessing.Process] = []

    if transcript_output:
        logger.info(f"Running transcription for {audio_input}")
        transcript_process = multiprocessing.Process(
            target=transcribe_wrapper,
            args=(audio_input, transcript_model, language, transcript_output),
        )
        processes.append(transcript_process)
        transcript_process.start()

    if diarization_output:
        # Sleep for 30 seconds to allow the transcription process to start
        time.sleep(30)

        logger.info(f"Running diarization for {audio_input}")
        hugging_face_key = diarize.get_huggingface_key()
        diarization_process = multiprocessing.Process(
            target=diarize_wrapper,
            args=(
                audio_input,
                hugging_face_key,
                speaker_count,
                min_speakers,
                max_speakers,
                diarization_output,
            ),
        )
        processes.append(diarization_process)
        diarization_process.start()

    for process in processes:
        process.join()

    if srt_output:
        logger.info(f"Generating Diarized SRT file for {audio_input}")
        max_words_per_line = int(params["subtitle_max_words_per_line"])
        subtitle.generate_diarized_subtitles(
            whisper_json=transcript_output,
            diarization_path=diarization_output,
            srt_path=srt_output,
            max_words_per_line=max_words_per_line,
            transcribeMe_path=transcribeMe_path,
        )
        logger.info(f"Generated Diarized SRT at {srt_output}")


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
    transcribeMe_path: Optional[str] = None,
):
    params = utils.config(utils.get_config_file(), "whispernote")

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
        max_words_per_line = int(params["subtitle_max_words_per_line"])
        subtitle.generate_diarized_subtitles(
            whisper_json=transcript_output,
            diarization_path=diarization_output,
            srt_path=srt_output,
            max_words_per_line=max_words_per_line,
            transcribeMe_path=transcribeMe_path,
        )
        logger.info(f"Generated Diarized SRT at {srt_output}")
