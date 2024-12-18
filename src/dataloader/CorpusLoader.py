"""Loads the retrieval Corpus.

Returns:
    List[Evidence]: The retrieval corpus
"""

import json
from typing import Dict, List, Any
from src.datastructures.Evidence import Evidence


class CorpusLoader:

    def load_corpus(self, corpus: List[Dict[Any, Any]]) -> List[Evidence]:
        """Loads the corpus for retrieval.

        Args:
            corpus (List[Dict[str]]): retrieval corpus

        Returns:
            List[Evidence]: list of passage instances
        """
        corpus_object = []
        for index, evidence in enumerate(corpus):
            if evidence["title"] == None:
                evidence_title = ""
            else:
                evidence_title = evidence["title"]
            evidence_instance = Evidence(
                idx=index,
                text=evidence["text"],
                title=evidence_title,
                url=evidence["url"],
            )
            corpus_object.append(evidence_instance)
        return corpus_object
