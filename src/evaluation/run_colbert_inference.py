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

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = DotScore()
    response = tasb_search.retrieve(corpus,queries,100)
    metrics = RetrievalMetrics(k_values=[1,5,10,100])
    #print(response)
    print("indices",len(response))
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))
    wiki_docs = {}
    with open("factiverse_colbert.json","w") as f:
        json.dump(response,f)
    for index, key in enumerate(list(response.keys())):
        wiki_docs[key] = []
        for idx in list(response[key].keys()):
            corpus_id = int(idx)
            wiki_docs[key].append(corpus[corpus_id].text())
    with open("factiverse_colbert.json","w") as f:
        json.dump(wiki_docs,f)
