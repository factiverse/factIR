from src.dataloader.RegularClaimsLoader import RegularClaimsLoader
from src.config.constants import Split
from src.datastructures.DenseHyperParams import DenseHyperParams
from src.retrievers.dense.DprSentSearch import DprSentSearch
from src.metrics.retrieval.RetrievalMetrics import RetrievalMetrics


if __name__ == "__main__":
    config_instance = DenseHyperParams(
        query_encoder_path="facebook-dpr-question_encoder-multiset-base",
        document_encoder_path="facebook-dpr-ctx_encoder-multiset-base",
        ann_search="faiss_search",
        convert_to_tensor=False,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    loader = RegularClaimsLoader("factiverse")

    queries, corpus, qrels = loader.read_retrieval_data(
        "data/claims.json", "data/qrel_processed.json", "data/final_corpus.json"
    )
    dpr_sent_search = DprSentSearch(config_instance, dataset_name="factiverse")
    _ = dpr_sent_search.get_ann_algo(768, 100, "euclidean")
    dpr_sent_search.create_index(corpus)
    response = dpr_sent_search.retrieve(queries, 100)
    print("indices", len(response))
    metrics = RetrievalMetrics(k_values=[1, 5, 10, 100])
    print(metrics.evaluate_retrieval(qrels=qrels, results=response))
