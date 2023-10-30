from typing import List

from whispernote.models.SubtitleElement import SubtitleElement


class Subtitles:
    def __init__(self) -> None:
        self.index = 0
        self.elements: List[SubtitleElement] = []

    def add_element(self, element: SubtitleElement) -> None:
        element.index = self.index
        self.index += 1
        self.elements.append(element)

    def join_adjacent_elements(self):
        # If speaker is the same, and the first element doesn't end with a stop symbol, join them
        stop_characters = [".", "?", "!"]
        max_words_per_line = 7

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
        with open(path, "w") as text_file:
            text_file.write(str(self))
