"""
Subtitles class, which contains a list of SubtitleElement objects.
"""
from typing import List

from whispernote.models.SubtitleElement import SubtitleElement


class Subtitles:
    """
    Subtitles class that contains a list of SubtitleElement objects.

    Attributes:
        index (int): The index of the subtitle element.
        elements (List[SubtitleElement]): The list of subtitle elements.
        display_mode (str): The display mode of the subtitles. Default is "srt".

    Methods:
        add_element: Adds a SubtitleElement to the list of subtitle elements.
        join_adjacent_elements: Joins adjacent subtitle elements with the same speaker if the first
            element doesn't end with a stop symbol.
        to_file: Writes the subtitles to a file in the specified display mode format.
    """
    def __init__(self, display_mode: str = "srt") -> None:
        self.index = 0
        self.elements: List[SubtitleElement] = []
        self._display_mode = display_mode

    @property
    def display_mode(self) -> str:
        """
        Current display mode of the subtitles.

        Returns:
            str: The display mode of the subtitles.
        """
        return self._display_mode

    @display_mode.setter
    def display_mode(self, value: str) -> None:
        self._display_mode = value
        for element in self.elements:
            element.display_mode = value

    def add_element(self, element: SubtitleElement) -> None:
        """
        Adds a SubtitleElement to the list of subtitle elements.

        Args:
            element (SubtitleElement): The SubtitleElement to be added.

        Returns:
            None
        """
        element.index = self.index
        self.index += 1
        self.elements.append(element)

    def join_adjacent_elements(self, max_words_per_line: int = 7) -> None:
        """
        Joins adjacent subtitle elements with the same speaker if the first
        element doesn't end with a stop symbol.

        Args:
            max_words_per_line (int): Maximum number of words per line. Default is 7.

        Returns:
            None
        """
        # If speaker is the same, and the first element doesn't end with a stop symbol, join them
        stop_characters = [".", "?", "!"]

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
        """
        Writes the subtitles to a file in the specified display mode format.

        Args:
            path (str): The path to the file to write the subtitles to.

        Returns:
            None
        """
        match self.display_mode:
            case "srt":
                for element in self.elements:
                    element.display_mode = "srt"
            case "transcribeMe":
                for element in self.elements:
                    element.display_mode = "transcribeMe"
            case _:
                pass

        with open(path, "w", encoding="utf-8") as text_file:
            text_file.write(str(self))
