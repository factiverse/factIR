from typing import List
import torch
import heapq
from sentence_transformers import SentenceTransformer
from src.datastructures.Evidence import Evidence
from src.datastructures.DenseHyperParams import DenseHyperParams
from src.datastructures.Claim import Claim

import logging

from src.retrievers.dense.indexer.indexer import AnnSearch
from src.metrics.SimilarityMatch import SimilarityMetric

logger = logging.getLogger(__name__)


class DprSentSearch:
    def __init__(self, config: DenseHyperParams, dataset_name=None):
        self.query_encoder = SentenceTransformer(
            config.query_encoder_path, device="cuda"
        )
        self.document_encoder = SentenceTransformer(
            config.document_encoder_path, device="cuda"
        )
        self.documents = {}
        self.titles = {}
        self.data = []
        self.args = config
        self.ann_search = AnnSearch()
        self.ann_algo = None
        self.sep = "[SEP]"
        self.config = config
        self.dataset_name = dataset_name
        self.logger = logging.getLogger(__name__)

    def get_passage_embeddings(self, passages: List[str] = None):
        # print("self.args.show_progress_bar",self.args.show_progress_bar)
        return self.document_encoder.encode(
            passages,
            convert_to_tensor=self.args.convert_to_tensor,
            convert_to_numpy=self.args.convert_to_numpy,
            show_progress_bar=self.args.show_progress_bar,
            batch_size=self.config.batch_size,
        )

    def get_ann_algo(self, emb_dim, num_trees: int = None, metric: str = None):
        self.ann_algo = self.ann_search.get_ann_instance(
            self.args.ann_search, self.data, emb_dim, num_trees, metric
        )
        return self.ann_algo

    def create_index(self, corpus=None):
        for data in list(corpus):
            self.documents[data.idx] = data.text
            self.titles[data.idx] = data.title

        self.index_mapping = list(self.documents.keys())

        passages = [
            self.documents[idx] + "[SEP]" + self.titles[idx]
            for idx in self.index_mapping
        ]
        # print("passages",len(passages))
        index_exists = self.ann_algo.load_index_if_available(self.dataset_name)
        ##TODO: Uncomment below for index usage
        # index_exists = False
        if index_exists:
            logger.info(
                f"Index already exists. Loading {self.args.ann_search} index"
            )
        else:
            passage_vectors = self.get_passage_embeddings(passages)
            assert len(passage_vectors) == len(self.index_mapping)
            self.ann_algo.create_index(passage_vectors, self.dataset_name)

    def retrieve_in_chunks(
        self,
        corpus: List[Evidence],
        queries: List[Claim],
        top_k: int,
        score_function: SimilarityMetric,
        return_sorted: bool = True,
        chunksize: int = 400000,
        **kwargs,
    ):
        """Retrieves in chunks for large collections.

        Args:
            corpus (List[Evidence]): retrieval corpus
            queries (List[Claim]): claims/questions
            top_k (int): number of documents to retrieve
            score_function (SimilarityMetric): decide the similarity fucntion to use
            return_sorted (bool, optional): return a sorted list of documents? Defaults to True.
            chunksize (int, optional): Defaults to 400000.

        Returns:
            Dict: retrieval results
        """
        queries_text = [query.text() for query in queries]
        corpus_ids = [doc.idx for doc in corpus]
        query_embeddings = self.query_encoder.encode(
            queries_text,
            batch_size=self.config.batch_size,
            show_progress_bar=self.args.show_progress_bar,
            convert_to_tensor=self.args.convert_to_tensor,
        )
        query_ids = [query.idx for query in queries]
        result_heaps = {
            qid: [] for qid in query_ids
        }  # Keep only the top-k docs for each query
        self.results = {qid: {} for qid in query_ids}
        batches = range(0, len(corpus), chunksize)
        for batch_num, corpus_start_idx in enumerate(batches):
            self.logger.info(
                "Encoding Batch {}/{}...".format(batch_num + 1, len(batches))
            )
            corpus_end_idx = min(corpus_start_idx + chunksize, len(corpus))

            sub_corpus = corpus[corpus_start_idx:corpus_end_idx]
            # Encode chunk of corpus
            contexts = []
            for evidence in sub_corpus:
                context = ""
                if evidence.title:
                    context = (
                        evidence.title + self.sep + evidence.text()
                    ).strip()
                else:
                    context = evidence.text().strip()
                contexts.append(context)
            sub_corpus_embeddings = self.get_passage_embeddings(contexts)

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = score_function.evaluate(
                query_embeddings, sub_corpus_embeddings
            )
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k + 1, len(cos_scores[1])),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr],
                    cos_scores_top_k_values[query_itr],
                ):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(
                                result_heaps[query_id], (score, corpus_id)
                            )
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(
                                result_heaps[query_id], (score, corpus_id)
                            )

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
        return self.results

    def retrieve(
        self,
        queries: List[Claim],
        top_k,
        chunk: bool = False,
        chunksize: int = 400000,
    ):
        """Retrieves from a collection to retruen relevant passages for a query.

        Args:
            queries (List[Claim]): claims/questions
            top_k (_type_): top_k documents to retrieve
            chunk (bool, optional): Indicator of whether to do chunked retrieval. Defaults to False.
            chunksize (int, optional): Defaults to 400000.

        Returns:
            Dict: retrieval results
        """
        query_vector = self.query_encoder.encode(
            [query.text for query in queries],
            convert_to_tensor=self.args.convert_to_tensor,
            show_progress_bar=self.args.show_progress_bar,
        )
        top_neighbours = self.ann_algo.get_top_n_neighbours(query_vector, top_k)
        response = {}
        for idx, q in enumerate(queries):
            response[str(q.idx)] = {}
            for index, c_idx in enumerate(top_neighbours["ids"][idx]):
                if idx >= 0:
                    response[str(q.idx)][str(self.index_mapping[c_idx])] = (
                        float(top_neighbours["distances"][idx][index])
                    )
        return response
