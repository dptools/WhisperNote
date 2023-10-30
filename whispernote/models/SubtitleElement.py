from typing import Optional


def ms_to_srt(ms: float) -> str:
    s = ms / 1000

    h = int(s // 3600)
    m = int((s % 3600) // 60)
    s = int(s % 60)

    mil = int((ms % 1000))
    # Format the timestamp as HH:MM:SS,MIL
    return f"{h:02d}:{m:02d}:{s:02d},{mil:03d}"


class SubtitleElement:
    def __init__(
        self,
        start_ms: int,
        end_ms: int,
        text: str,
        speaker: str,
        index: Optional[int] = None,
    ) -> None:
        self.index = index
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.text = text.strip()
        self.speaker = speaker

    def __str__(self) -> str:
        string_representation = f"""{self.index}
{ms_to_srt(self.start_ms)} --> {ms_to_srt(self.end_ms)}
[{self.speaker}]
{self.text.strip()}\n"""
        return string_representation

    def __repr__(self) -> str:
        return self.__str__()
