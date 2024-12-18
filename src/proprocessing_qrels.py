import json
from typing import List, Dict

from utils.reader_utils import read_from_jsonlines


def unicode_processing(snippet: str):
    strencode = snippet.encode("ascii", "ignore")
    strdecode = strencode.decode()
    return strdecode.lower().strip()


def form_qrels(filepath: str):
    qrels = read_from_jsonlines(filepath)
    claims = []
    with open("../data/final_corpus.json") as f:
        corpus = json.load(f)
    qrel_final = {}
    unique_claims = []
    count = 0
    for qrel in qrels:
        if qrel["claim"] not in unique_claims:
            claims.append({"text": qrel["claim"]})
            qrel_final[str(count)] = {}
            unique_claims.append(qrel["claim"])
            count += 1

        evidence_index = [
            i
            for i, evidence in enumerate(corpus)
            if evidence["text"] == unicode_processing(qrel["evidence.snippet"])
        ]
        print("evidence_index", evidence_index)
        qrel_final[str(count - 1)][str(evidence_index[0])] = qrel[
            "relevanceLabel"
        ]

    with open("../data/qrel_processed.json", "w") as f:
        json.dump(qrel_final, f, indent=4)

    with open("../data/claims.json", "w") as f:
        json.dump(claims, f)


if __name__ == "__main__":
    form_qrels("../data/qrels_raw.jsonl")
