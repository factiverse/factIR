from dataclasses import dataclass


@dataclass
class Claim:
    """A base class to hold all details about the claim.

    Attributes:
        _text (str): The text of the claim.
        _idx (int): The ID of the question.
    """

    text: str
    idx: int

    def set_attention_mask(self, attention_mask) -> None:
        self.attention_mask = attention_mask
