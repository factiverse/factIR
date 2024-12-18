"""CrossEncoder class.

    Usage:
        cross_encoder = CrossEncoder("model_name")
    """

from sentence_transformers.cross_encoder import CrossEncoder as CrossEnc
from typing import List, Tuple, Dict


class CrossEncoder:
    def __init__(self, model_name: str, **kwargs):
        """initializes cross encoder

        Args:
            model_name (str): cross encoder model name
        """        
        self.model = CrossEnc(model_name, **kwargs)

    def predict(
        self,
        sentences: List[Tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> List[float]:
        """Cross encoder that predicts a score for query doc pair.

        Args:
            sentences (List[Tuple[str, str]]):
            batch_size (int, optional): Defaults to 32.
            show_progress_bar (bool, optional): Defaults to True.

        Returns:
            List[float]: score for query doc pair
        """
        return self.model.predict(
            sentences=sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
