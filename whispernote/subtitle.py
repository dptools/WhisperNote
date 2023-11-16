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
import copy
import json
from io import StringIO
from typing import Dict, Optional, List

import pandas as pd

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


def get_transcript_df(
    whisper_json_path: str,
) -> pd.DataFrame:
    transcript = process_whisper_transcript(whisper_json_path).strip().split("\n")
    data = [line.split(",", maxsplit=2) for line in transcript]
    transcript_df = pd.DataFrame(data, columns=["start", "end", "text"])

    transcript_df["start"] = transcript_df["start"].astype(int)
    transcript_df["end"] = transcript_df["end"].astype(int)

    transcript_df["text"] = transcript_df["text"].str.strip()

    return transcript_df


def get_diarization_df(
    diarization_path: str,
) -> pd.DataFrame:
    diarization = open(diarization_path).read().strip().split("\n")

    data = [line.split(",") for line in diarization]
    # If the diarization file has a header, remove it
    if data[0][0] == "start":
        data = data[1:]

    diarization_df = pd.DataFrame(data, columns=["start", "end", "speaker"])

    diarization_df["start"] = diarization_df["start"].astype(int)
    diarization_df["end"] = diarization_df["end"].astype(int)

    return diarization_df


def merge_transcript_json(
    whisper_json_path: str,
) -> Dict[str, Any]:
    df = get_transcript_df(whisper_json_path=whisper_json_path)
    stop_characters = [".", "?", "!"]

    groups = df.groupby(df["text"].shift().str[-1].isin(stop_characters).cumsum())
    # groups = [group for _, group in groups]

    json_data: Dict[str, Any] = {}
    json_data["segments"] = []
    for idx, group in groups:
        group_json_data = {}
        group_json_data["idx"] = idx
        group_json_data["start"] = group["start"].min()
        group_json_data["end"] = group["end"].max()
        group_json_data["text"] = " ".join(group["text"].tolist())

        group_json_data_parts = []
        for part_idx, part in group.iterrows():
            part_json_data = {}
            part_json_data["idx"] = part_idx
            part_json_data["start"] = part["start"]
            part_json_data["end"] = part["end"]
            part_json_data["text"] = part["text"]

            group_json_data_parts.append(part_json_data)

        group_json_data["parts"] = group_json_data_parts
        json_data["segments"].append(group_json_data)

    return json_data


def transcript_json_to_df(transcript_json: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(transcript_json["segments"])

    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)

    df["text"] = df["text"].str.strip()

    # Drop idx column
    df.drop(columns=["idx"], inplace=True)

    return df


def combine_transcript_diarization(
    whisper_json_path: str,
    diarization_csv_path: str,
):
    transcript_json = merge_transcript_json(whisper_json_path=whisper_json_path)
    transcript_df = transcript_json_to_df(transcript_json=transcript_json)
    diarization_df = get_diarization_df(diarization_path=diarization_csv_path)

    # Add speaker column to transcript_df with dtype str
    transcript_df["speaker"] = ""

    for diarization_line in diarization_df.iterrows():
        start = diarization_line[1]["start"]
        end = diarization_line[1]["end"]
        speaker = diarization_line[1]["speaker"]

        transcript_df.loc[
            (transcript_df["start"] >= start) & (transcript_df["end"] <= end), "speaker"
        ] = speaker

    return transcript_df


def process_combined_transcript_diarization(
    df: pd.DataFrame,
) -> pd.DataFrame:
    # Replace empty speaker values with NaN
    df["speaker"].replace("", float("NaN"), inplace=True)

    # Fill in missing speaker values
    df["speaker"].ffill(inplace=True)
    df["speaker"].bfill(inplace=True)

    # Drop rows with empty text
    df.dropna(subset=["text"], inplace=True)

    return df


def construct_diarised_subtitles(whisper_json: str, diarization_path: str) -> Subtitles:
    diarized_transcript_df = combine_transcript_diarization(
        whisper_json_path=whisper_json,
        diarization_csv_path=diarization_path,
    )

    diarized_transcript_df = process_combined_transcript_diarization(
        df=diarized_transcript_df
    )

    subtitles = Subtitles()

    for _, row in diarized_transcript_df.iterrows():
        subtitle_element = SubtitleElement(
            start_ms=row["start"],
            end_ms=row["end"],
            text=row["text"],
            speaker=row["speaker"],
        )
        subtitles.add_element(subtitle_element)

    return subtitles

def generate_diarized_subtitles(
    whisper_json: str,
    diarization_path: str,
    srt_path: str,
    transcribeMe_path: Optional[str] = None,
    max_words_per_line: int = 7,
) -> List[str]:
    subtitles = construct_diarised_subtitles(
        whisper_json=whisper_json, diarization_path=diarization_path
    )
    
    subtitles_srt = copy.deepcopy(subtitles)

    subtitles_srt.join_adjacent_elements(max_words_per_line=max_words_per_line)
    subtitles_srt.to_file(srt_path)

    if transcribeMe_path is None:
        return [srt_path]

    subtitles_transcribeMe = copy.deepcopy(subtitles)
    subtitles_transcribeMe.display_mode = "transcribeMe"

    subtitles_transcribeMe.join_adjacent_elements(max_words_per_line=max_words_per_line)
    subtitles_transcribeMe.to_file(transcribeMe_path)

    return [srt_path, transcribeMe_path]


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--whisper-json",
        type=str,
        required=True,
        help="Path to the Whisper JSON file",
    )
    arg_parser.add_argument(
        "--diarization-csv",
        type=str,
        required=True,
        help="Path to the diarization CSV file",
    )
    arg_parser.add_argument(
        "--srt",
        type=str,
        required=True,
        help="Path to the Output SRT file",
    )

    args = arg_parser.parse_args()

    print("Generating diarized subtitles...")
    generate_diarized_subtitles(
        whisper_json=args.whisper_json,
        diarization_path=args.diarization_csv,
        srt_path=args.srt,
    )

    print("Done!")
