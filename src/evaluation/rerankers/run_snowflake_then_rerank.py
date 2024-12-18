from src.dataloader.RegularClaimsLoader import RegularClaimsLoader
from src.config.constants import Split
from src.datastructures.DenseHyperParams import DenseHyperParams
from src.retrievers.dense.DenseFullSearch import DenseFullSearch
from src.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from src.metrics.SimilarityMatch import CosineSimilarity as CosScore
from src.rerankers.ReRanker import Reranker


if __name__ == "__main__":
    config_instance = DenseHyperParams(
        query_encoder_path="Snowflake/snowflake-arctic-embed-s",
        document_encoder_path="Snowflake/snowflake-arctic-embed-s",
        batch_size=32,
        show_progress_bar=True,
    )

    loader = RegularClaimsLoader("factiverse")

    queries, corpus, qrels = loader.read_retrieval_data(
        "data/claims.json", "data/qrel_processed.json", "data/final_corpus.json"
    )
    print(len(qrels), len(corpus))
    mpnet_search = DenseFullSearch(config_instance)
    similarity_measure = CosScore()

    response = mpnet_search.retrieve(
        corpus, queries, top_k=100, score_function=similarity_measure
    )
    metrics = RetrievalMetrics(k_values=[1, 5, 10, 100])

    print(metrics.evaluate_retrieval(qrels=qrels, results=response))

    reranker = Reranker("Alibaba-NLP/gte-multilingual-reranker-base")
    # cross-encoder/ms-marco-MiniLM-L-6-v2
    results = reranker.rerank(corpus, queries, response, 100)
    #print("results", results["0"], response["0"], qrels["0"])
    print(metrics.evaluate_retrieval(qrels=qrels, results=results))
