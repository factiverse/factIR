import tqdm
import json
from src.config.constants import Split
from typing import Dict, Any, List
from src import utils
from src.datastructures.Claim import Claim
from src.datastructures.Evidence import Evidence
from src.dataloader.CorpusLoader import CorpusLoader


class RegularClaimsLoader:
    def __init__(
        self,
        dataset: str,
        tokenizer="bert-base-uncased",
        config_path="test_config.ini",
        split=Split.TRAIN,
        batch_size=None,
        corpus=None,
    ):
        """Data loader class to load claims dataset.

        Arguments:
        dataset (str): string containing the dataset alias
        tokenzier (str) : name of the tokenizer model. Set tokenizer as None,
        if only samples to be loaded but not tokenized and stored.
        This can help save time if only the raw dataset is needed.
        config_path (str) : path to the configuration file containing various parameters
        split (Split) : Split of the dataset to be loaded
        batch_size (int) : batch size to process the dataset.
        corpus: corpus containing all needed passages.
        """
        self.corpus = []
        self.queries = []
        self.corpus_loader = CorpusLoader()

    def load_raw_dataset(self, filepath: str, split=Split.TEST) -> Dict:
        """Loads a raw json file.

        Args:
             filepath : str The path to datafile
        Returns:
         The raw json data
        """
        dataset = utils.reader_utils.read_from_jsonlines(split)
        return dataset

    def read_json(self, path) -> Dict[Any, Any]:
        """Reads the fact check dataset from the given path.

        Args:
            path: The path to the dataset.

        Returns:
            A list of facts/fact checker urls.
        """
        with open(path) as f:
            data = json.load(f)
        return data

    def read_claims(self, filepath: str, split=Split.TEST) -> List[Claim]:
        """Reads the claims.

        Args:
            filepath: The path to the dataset.

        Returns:
            A list of claims.
        """
        raw_claims = self.read_json(filepath)
        for index, claim in enumerate(raw_claims):
            self.queries.append(Claim(idx=index, text=claim["text"]))
        return self.queries

    def read_qrels(self, filepath: str, split=Split.TEST) -> Dict:
        """Reads the qrels.

        Args:
            filepath: The path to the qrel file.

        Returns:
            A dict of qrels.
        """
        raw_qrels = self.read_json(filepath)
        return raw_qrels

    def read_corpus(self, filepath: str, split=Split.TEST):
        """Reads the corpus.

        Args:
            filepath: The path to the dataset.

        Returns:
            A list of claims.
        """
        raw_corpus = self.read_json(filepath)
        self.corpus = self.corpus_loader.load_corpus(raw_corpus)
        return self.corpus

    def read_retrieval_data(
        self, claim_filepath: str, qrel_filepath: str, corpus_filepath: str
    ):
        claims = self.read_claims(claim_filepath)
        corpus = self.read_corpus(corpus_filepath)
        qrels = self.read_qrels(qrel_filepath)
        return claims, corpus, qrels
