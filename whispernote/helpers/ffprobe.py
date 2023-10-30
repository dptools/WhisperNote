# Reference:
# https://stackoverflow.com/questions/9896644/getting-ffprobe-information-with-python

import argparse
import json
import subprocess
from typing import NamedTuple
import sys
from pathlib import Path


class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def get_metadata(file_path_to_process: tuple) -> dict:
    ffprobe_result = ffprobe(file_path_to_process)

    if ffprobe_result.return_code != 0:
        print("Error: ffprobe failed.")
        print(ffprobe_result.error, file=sys.stderr)

    metadata = json.loads(ffprobe_result.json)

    return metadata


def ffprobe(file_path) -> FFProbeResult:
    command_array = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        f"'{file_path}'",
    ]

    result = subprocess.run(
        " ".join(command_array),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
    )

    return FFProbeResult(
        return_code=result.returncode, json=result.stdout, error=result.stderr
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View ffprobe output")
    parser.add_argument("-i", "--input", help="File Name", required=True)
    args = parser.parse_args()
    if not Path(args.input).is_file():
        print("could not read file: " + args.input)
        exit(1)
    print("File:       {}".format(args.input))
    ffprobe_result = ffprobe(file_path=args.input)
    if ffprobe_result.return_code == 0:
        # Print the raw json string
        print(ffprobe_result.json)

        # or print a summary of each stream
        d = json.loads(ffprobe_result.json)
        streams = d.get("streams", [])
        for stream in streams:
            print(
                f'{stream.get("codec_type", "unknown")}: {stream.get("codec_long_name")}'
            )

    else:
        print("ERROR")
        print(ffprobe_result.error, file=sys.stderr)
