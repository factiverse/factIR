from src.dataloader.RegularClaimsLoader import RegularClaimsLoader
from src.config.constants import Split
from src.datastructures.DenseHyperParams import DenseHyperParams
from src.retrievers.dense.DenseFullSearch import DenseFullSearch
from src.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from src.metrics.SimilarityMatch import CosineSimilarity as CosScore


if __name__ == "__main__":
    config_instance = DenseHyperParams(
        query_encoder_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        document_encoder_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size=32,
        show_progress_bar=True,
    )

    loader = RegularClaimsLoader("factiverse")

    queries, corpus, qrels = loader.read_retrieval_data(
        "data/claims.json", "data/qrel_processed.json", "data/final_corpus.json"
    )
    mpnet_search = DenseFullSearch(config_instance)
    similarity_measure = CosScore()

    response = mpnet_search.retrieve(
        corpus, queries, top_k=100, score_function=similarity_measure
    )
    print("indices", len(response))
    metrics = RetrievalMetrics(k_values=[1, 10, 100])
    print(metrics.evaluate_retrieval(qrels=qrels, results=response))
