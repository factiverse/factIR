from src.retrievers.dense.Contriever import Contriever
from src.config.constants import Split
from src.dataloader.RegularClaimsLoader import RegularClaimsLoader
from src.datastructures.DenseHyperParams import DenseHyperParams
from src.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from src.metrics.SimilarityMatch import CosineSimilarity

import os

if __name__ == "__main__":
    config_instance = DenseHyperParams(
        query_encoder_path="facebook/contriever",
        document_encoder_path="facebook/contriever",
        batch_size=32,
        show_progress_bar=True,
    )

    loader = RegularClaimsLoader("factiverse")

    queries, corpus, qrels = loader.read_retrieval_data(
        "data/claims.json", "data/qrel_processed.json", "data/final_corpus.json"
    )
    print("queries", len(queries), len(qrels), len(corpus), queries[0])
    contriever_search = Contriever(config_instance)

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = CosineSimilarity()
    response = contriever_search.retrieve(
        corpus,
        queries,
        100,
        similarity_measure,
        chunk=False,
        chunksize=60000,
        data_name="factiverse_search",
    )
    print("indices", len(response))
    metrics = RetrievalMetrics(k_values=[1,5, 10, 100])
    print(response)
    print(metrics.evaluate_retrieval(qrels=qrels, results=response))
