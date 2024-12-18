from src.dataloader.RegularClaimsLoader import RegularClaimsLoader
from src.config.constants import Split
from src.retrievers.lexical.bm25 import BM25Search
from src.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
import os

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

    response = bm25_search.retrieve(corpus, queries, 100)
    print("indices", len(response), len(queries), len(corpus))
    print(response)
    metrics = RetrievalMetrics(k_values=[1, 5, 10, 100])
    print(metrics.evaluate_retrieval(qrels=qrels, results=response))
