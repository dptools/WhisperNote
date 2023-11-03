from typing import List

from whispernote.models.SubtitleElement import SubtitleElement


class Subtitles:
    def __init__(self, display_mode: str = "srt") -> None:
        self.index = 0
        self.elements: List[SubtitleElement] = []
        self._display_mode = display_mode

    @property
    def display_mode(self) -> str:
        return self._display_mode

    @display_mode.setter
    def display_mode(self, value: str) -> None:
        self._display_mode = value
        for element in self.elements:
            element.display_mode = value

    def add_element(self, element: SubtitleElement) -> None:
        element.index = self.index
        self.index += 1
        self.elements.append(element)

    def join_adjacent_elements(self, max_words_per_line: int = 7) -> None:
        # If speaker is the same, and the first element doesn't end with a stop symbol, join them
        stop_characters = [".", "?", "!"]
        max_words_per_line = max_words_per_line

        idx = 0
        while idx < len(self.elements) - 1:
            element = self.elements[idx]
            next_element = self.elements[idx + 1]

            if (
                element.speaker == next_element.speaker
                and element.text[-1] not in stop_characters
                and len(element.text.split(" ")) < max_words_per_line
            ):
                element.text = element.text.strip() + " " + next_element.text.strip()
                element.end_ms = next_element.end_ms
                self.elements.remove(next_element)
            else:
                idx += 1

    def __str__(self) -> str:
        string_representation = ""
        for element in self.elements:
            string_representation += str(element) + "\n"
        return string_representation

    def __repr__(self) -> str:
        return self.__str__()

    def to_file(self, path: str) -> None:
        match self.display_mode:
            case "srt":
                for element in self.elements:
                    element.display_mode = "srt"
            case "transcribeMe":
                for element in self.elements:
                    element.display_mode = "transcribeMe"
            case _:
                pass

        with open(path, "w") as text_file:
            text_file.write(str(self))
