import json
import os
from src.retrievers.dense.Contriever import Contriever
from src.config.constants import Split
from src.retrievers.lexical.bm25 import BM25Search
from src.dataloader.RegularClaimsLoader import RegularClaimsLoader
from src.datastructures.DenseHyperParams import DenseHyperParams
from src.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from src.metrics.SimilarityMatch import CosineSimilarity, DotScore
from src.metrics.SimilarityMatch import CosineSimilarity as CosScore
from src.rerankers.ReRanker import Reranker

if __name__ == "__main__":
    # config = config_instance.get_all_params()

    loader = RegularClaimsLoader("factiverse")

    queries, corpus, qrels = loader.read_retrieval_data(
        "data/claims.json", "data/qrel_processed.json", "data/final_corpus.json"
    )
    cert_path = os.environ["ca_certs"]
    password = os.environ["elastic_password"]
    bm25_search = BM25Search(
        index_name="factiverse",
        initialize=False,
        cert_path=cert_path,
        elastic_password=password,
    )
    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    response = bm25_search.retrieve(corpus, queries, 100)
    metrics = RetrievalMetrics(k_values=[1, 5, 10, 100])

    print(metrics.evaluate_retrieval(qrels=qrels, results=response))
    reranker = Reranker("snowflake/snowflake-arctic-embed-s")
    # cross-encoder/ms-marco-MiniLM-L-6-v2
    results = reranker.rerank(corpus, queries, response, 100)
    print("results", results["0"], response["0"], qrels["0"])
    metrics = RetrievalMetrics(k_values=[1, 5, 10, 100])
    print(metrics.evaluate_retrieval(qrels=qrels, results=results))
