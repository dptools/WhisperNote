import json
from io import StringIO
from typing import Dict

from whispernote.models.SubtitleElement import SubtitleElement
from whispernote.models.Subtitles import Subtitles


def process_whisper_transcript(whisper_json: str) -> str:
    with open(whisper_json) as json_file:
        file_contents = json_file.read()
        parsed_json = json.loads(file_contents)

    segments = parsed_json["segments"]
    srt_file = StringIO()

    for segment in segments:
        start, end, _ = int(segment["start"]), int(segment["end"]), segment["text"]

        words = segment["words"]
        for word in words:
            start, end, word = (word["start"]), (word["end"]), word["word"]
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            srt_file.write(f"{start_ms},{end_ms},{word}\n")

    return srt_file.getvalue()


def generate_diarized_subtitles(
    whisper_json: str, diarization_path: str, srt_path: str
) -> str:
    transcript = process_whisper_transcript(whisper_json).strip().split("\n")
    diarization = open(diarization_path).read().strip().split("\n")

    subtitles = Subtitles()

    for transcript_line in transcript:
        transcript_parts = transcript_line.split(",")
        transcript_start = transcript_parts[0]
        transcript_end = transcript_parts[1]
        transcript_text = ",".join(transcript_parts[2:])

        transcript_start, transcript_end = int(transcript_start), int(transcript_end)

        cumulative_durations: Dict[str, int] = {}

        for diarization_line in diarization:
            diarization_start, diarization_end, speaker = diarization_line.split(",")

            if speaker not in cumulative_durations:
                cumulative_durations[speaker] = 0

            diarization_start, diarization_end = int(diarization_start), int(
                diarization_end
            )

            if diarization_start > transcript_end:
                # cumulative_durations[speaker] += transcript_end - transcript_start
                break
            elif diarization_end < transcript_start:
                continue

            if diarization_end < transcript_start:
                cumulative_durations[speaker] += diarization_end - diarization_start
            elif diarization_start > transcript_end:
                pass
            else:
                cumulative_durations[speaker] += transcript_end - max(
                    diarization_start, transcript_start
                )

        primary_speaker = max(cumulative_durations, key=cumulative_durations.get)

        subtitle_element = SubtitleElement(
            start_ms=transcript_start,
            end_ms=transcript_end,
            text=transcript_text,
            speaker=primary_speaker,
        )
        subtitles.add_element(subtitle_element)

    subtitles.join_adjacent_elements()
    subtitles.to_file(srt_path)
