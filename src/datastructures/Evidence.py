from typing import List

import pandas as pd
from src.config.constants import Separators
from dataclasses import dataclass


@dataclass
class Evidence:
    """Data class to hold evidence/context for Fact Checking

    Args:
        text : text of evidence passage
        title : title of evidence passage
        idx : index of evidence passage
    """

    text: str
    idx: int
    title: str
    url: str
