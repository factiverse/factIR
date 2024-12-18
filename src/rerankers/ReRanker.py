"""ReRanker class: Instantiates a cross-encoder for re-ranking.

    Usage:
        re_ranker = ReRanker("model_name")
    """
import logging
import torch
from typing import List, Dict
from src.datastructures.Claim import Claim
from src.datastructures.Evidence import Evidence

from src.rerankers.CrossEncoder import CrossEncoder


class Reranker:

    def __init__(self, model_name, batch_size: int = 128, **kwargs):
        """Initializes a re-ranker.

        Args:
            model_name (_type_): cross encoder model name
            batch_size (int, optional): Defaults to 128.
        """
        self.cross_encoder = CrossEncoder(model_name, trust_remote_code=True)
        self.batch_size = batch_size
        self.rerank_results = {}

    def rerank(
        self,
        corpus: List[Evidence],
        queries: List[Claim],
        results: Dict[str, Dict[str, float]],
        top_k: int,
    ) -> Dict[str, Dict[str, float]]:
        """Reanks given query document pairs.

        Args:
            corpus (List[Evidence]): retrieval corpus
            queries (List[Claim]): queries to retrieve documents for
            results (Dict[str, Dict[str, float]]): query document pairs with scores
            top_k (int): number of documents to rank and return

        Returns:
            Dict[str, Dict[str, float]]: re-ranked documents for queries
        """
        sentence_pairs, pair_ids = [], []
        corpus_final = {}
        for evidence in corpus:
            corpus_final[str(evidence.idx)] = {
                "title": evidence.title,
                "text": evidence.text,
            }

        queries_final = {}
        for index, query in enumerate(queries):
            queries_final[str(query.idx)] = query.text
        print("queries***********",len(queries),len(results))
        # print("query_ids**",query_ids)
        # queries = [query.text for query in list(queries)]

        for query_id in results:
            print("results[query_id]",len(results[query_id]))
            for doc_id in list(results[query_id].keys()):
                pair_ids.append([query_id, doc_id])
                corpus_text = corpus_final[doc_id]["title"]+ "[SEP]"+ corpus_final[doc_id]["text"]
                corpus_text = corpus_text.strip()
                sentence_pairs.append([queries_final[query_id], corpus_text])

        logging.info("Intitating Rerank Top-{}....".format(top_k))
        rerank_scores = [
            float(score)
            for score in self.cross_encoder.predict(
                sentence_pairs, batch_size=self.batch_size
            )
        ]
        print("rerank_scores",len(rerank_scores),len(pair_ids),len(sentence_pairs),sentence_pairs[:5])

        self.rerank_results = {query_id: {} for query_id in results}
        sorted_list = sorted(
            zip(pair_ids, rerank_scores), key=lambda x: x[1], reverse=True
        )
        # print(sorted_list[:10])
        for pair, score in sorted_list:
            query_id, doc_id = pair[0], pair[1]
            self.rerank_results[str(query_id)][str(doc_id)] = score

        return self.rerank_results
