"""
SubtitleElement class that represents a single subtitle element.
"""
from typing import Optional


def ms_to_HMSms(ms: float, delim: str = '.') -> str:
    """
    Converts milliseconds to HH:MM:SS<delim>MIL format.

    Args:
        ms (float): The time in milliseconds.
        delim (str): The delimiter between seconds and milliseconds. Default is '.'.

    Returns:
        str: The time in HH:MM:SS<delim>MIL format.
    """
    s = ms / 1000

    h = int(s // 3600)
    m = int((s % 3600) // 60)
    s = int(s % 60)

    mil = int((ms % 1000))
    # Format the timestamp as HH:MM:SS<delim>MIL
    return f"{h:02d}:{m:02d}:{s:02d}{delim}{mil:03d}"


class SubtitleElement:
    """
    SubtitleElement class that represents a single subtitle element.

    Attributes:
        index (int): The index of the subtitle element.
        start_ms (int): The start time of the subtitle element in milliseconds.
        end_ms (int): The end time of the subtitle element in milliseconds.
        text (str): The text content of the subtitle element.
        speaker (str): The speaker of the subtitle element.
        display_mode (str): The display mode of the subtitle element. Default is "srt".

    Methods:
        srt_string: Returns the string representation of the subtitle element in SRT format.
        transcript_string: Returns the string representation of the subtitle element in transcript
            format.
    """
    def __init__(
        self,
        start_ms: int,
        end_ms: int,
        text: str,
        speaker: str,
        index: Optional[int] = None,
        display_mode: str = "srt",
    ) -> None:
        self.index = index
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.text = text.strip()
        self.speaker = speaker
        self.display_mode = display_mode

    def srt_string(self) -> str:
        """
        Returns the string representation of the subtitle element in SRT format.

        SRT format:
        idx
        HH:MM:SS,MMM --> HH:MM:SS,MMM
        [speaker]

        Returns:
            str: The string representation of the subtitle element in SRT format.
        """
        delim = ','
        string_representation = f"""{self.index}
{ms_to_HMSms(self.start_ms, delim=delim)} --> {ms_to_HMSms(self.end_ms, delim=delim)}
[{self.speaker}]
{self.text.strip()}\n"""

        return string_representation

    def transcript_string(self) -> str:
        """
        Returns the string representation of the subtitle element in transcript format.

        Transcript format:
        [speaker] HH:MM:SS.MMM text

        Returns:
            str: The string representation of the subtitle element in transcript format.
        """
        string_representation = (
            f"""{self.speaker} {ms_to_HMSms(self.start_ms)} {self.text.strip()}\n"""
        )
        return string_representation

    def __str__(self) -> str:
        match self.display_mode:
            case "srt":
                return self.srt_string()
            case "transcribeMe":
                return self.transcript_string()
            case _:
                return self.srt_string()

    def __repr__(self) -> str:
        return self.__str__()
