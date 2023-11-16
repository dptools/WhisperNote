from typing import Optional


def ms_to_HMSms(ms: float, delim: str = '.') -> str:
    s = ms / 1000

    h = int(s // 3600)
    m = int((s % 3600) // 60)
    s = int(s % 60)

    mil = int((ms % 1000))
    # Format the timestamp as HH:MM:SS<delim>MIL
    return f"{h:02d}:{m:02d}:{s:02d}{delim}{mil:03d}"


class SubtitleElement:
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
        delim = ','
        string_representation = f"""{self.index}
{ms_to_HMSms(self.start_ms, delim=delim)} --> {ms_to_HMSms(self.end_ms, delim=delim)}
[{self.speaker}]
{self.text.strip()}\n"""

        return string_representation

    def transcript_string(self) -> str:
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
