from typing import List
from dataclasses import dataclass


@dataclass
class Verdict:
    """A base class to hold all details about the answer aspect in question answering.

    Attributes:
        _text (str): The text of the answer.
        _idx (int): The ID of the answer.
    Usage:
        verdict = Verdict("True")
        text = verdict.text()
    """

    _text: str
    _idx: int

    def flatten(self) -> List[str]:
        """Flattens the answer structure if complex into a simple list of answer texts."""
        return [self._text]
