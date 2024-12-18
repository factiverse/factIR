import json
from src.retrievers.dense.Contriever import Contriever
from src.config.constants import Split
from src.dataloader.RegularClaimsLoader import RegularClaimsLoader
from src.datastructures.DenseHyperParams import DenseHyperParams
from src.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from src.metrics.SimilarityMatch import CosineSimilarity, DotScore

from colbert.infra import ColBERTConfig
from src.retrievers.dense.TCTColBERT import TCTColBERT


if __name__ == "__main__":
    config_instance = ColBERTConfig(doc_maxlen=256, nbits=2, kmeans_niters=4,bsize=16, gpus="")

    loader = RegularClaimsLoader("factiverse")

    queries, corpus, qrels = loader.read_retrieval_data(
        "data/claims.json", "data/qrel_processed.json", "data/final_corpus.json"
    )
    tasb_search = TCTColBERT(config_instance,checkpoint="colbert-ir/colbertv2.0")