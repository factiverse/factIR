import json
from typing import Any, Dict
import os


def open_file_wrapped(filepath: str, **kwargs) -> Any:
    return open(filepath, **kwargs)


def read_from_jsonlines(filepath: str) -> list[dict[Any, Any]]:
    """Reads in the jsonlines file at the filepath."""
    with open_file_wrapped(filepath, mode="r") as f:
        dict_list = [json.loads(line) for line in f]
        return dict_list
