"""This class contains the hyperparameters to setup the Dense Passage Retriever"""

from src.datastructures.basehyperparams import BaseHyperParameters
from typing import Dict


class DenseHyperParams(BaseHyperParameters):
    def __init__(
        self,
        query_max_length: int = 128,
        query_encoder_path: str = "facebook/dpr-question_encoder-multiset-base",
        document_encoder_path: str = "facebook/dpr-ctx_encoder-multiset-base",
        learning_rate: float = 1e-5,
        num_negative_samples: int = 5,
        ann_search: str = "faiss_search",
        convert_to_tensor: bool = True,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        batch_size: int = 64,
    ) -> None:
        """Initializes the hyperparameters for dense retrieval.

        Args:
            query_max_length (int, optional): _description_. Defaults to 128.
            query_encoder_path (str, optional): _description_. Defaults to "facebook/dpr-question_encoder-multiset-base".
            document_encoder_path (str, optional): _description_. Defaults to "facebook/dpr-ctx_encoder-multiset-base".
            learning_rate (float, optional): _description_. Defaults to 1e-5.
            num_negative_samples (int, optional): _description_. Defaults to 5.
            ann_search (str, optional): _description_. Defaults to "faiss_search".
            convert_to_tensor (bool, optional): _description_. Defaults to True.
            show_progress_bar (bool, optional): _description_. Defaults to True.
            convert_to_numpy (bool, optional): _description_. Defaults to True.
            batch_size (int, optional): _description_. Defaults to 64.
        """
        super().__init__()

        self.query_max_length = query_max_length
        self.query_encoder_path = query_encoder_path
        self.document_encoder_path = document_encoder_path
        self.learning_rate = learning_rate
        self.num_negative_samples = num_negative_samples
        self.ann_search = ann_search
        self.convert_to_tensor = convert_to_tensor
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.convert_to_numpy = convert_to_numpy

    def get_all_params(self) -> Dict:
        """returns dense hyperparameters

        Returns:
            dict: dictionary of hyperparams
        """
        config = {
            "query_length": self.query_max_length,
            "query_encoder_path": self.query_encoder_path,
            "document_encoder_path": self.document_encoder_path,
            "learning_rate": self.learning_rate,
            "num_negative_samples": self.num_negative_samples,
            "ann_search": self.ann_search,
            "convert_to_tensor": self.convert_to_tensor,
            "convert_to_numpy": self.convert_to_numpy,
            "batch_size": self.batch_size,
            "show_progress_bar": self.show_progress_bar,
        }
        return config
