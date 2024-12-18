import json
from typing import List, Dict

from utils.reader_utils import read_from_jsonlines


def unicode_processing(snippet: str):
    strencode = snippet.encode("ascii", "ignore")
    strdecode = strencode.decode()
    return strdecode.lower().strip()


def get_corpus(filepath: str) -> List[Dict]:
    corpus = read_from_jsonlines(filepath)
    deduplicated_snippets = list()
    evidences = list()
    for evidence in corpus:
        if not evidence["snippet"] or evidence["snippet"].strip() == "":
            continue
        evidence_snippet = unicode_processing(evidence["snippet"])
        if evidence_snippet in evidences:
            continue
        evidence_object = {
            "text": evidence_snippet,
            "url": evidence["url"],
            "title": evidence["title"],
        }
        deduplicated_snippets.append(evidence_object)
        evidences.append(evidence_snippet)

    print("deduplicated_snippets", len(deduplicated_snippets))
    return deduplicated_snippets


def form_corpus(corpus):
    relevance_snippets = read_from_jsonlines(
        "/home/venktesh/retrieval_benchmarking/data/qrels_raw.jsonl"
    )
    evidences = [snippet["text"] for snippet in corpus]
    final_evidence = []
    for relevance_snippet in relevance_snippets:
        evidence_snippet = unicode_processing(
            relevance_snippet["evidence.snippet"]
        )
        if evidence_snippet in evidences:
            continue

        evidence_object = {
            "text": evidence_snippet,
            "url": relevance_snippet["evidence.url"],
            "title": unicode_processing(relevance_snippet["evidence.title"]),
        }
        final_evidence.append(evidence_object)
        evidences.append(evidence_snippet)
    for snippet_obj in corpus:
        final_evidence.append(snippet_obj)

    print("final_deduplicated_corpus", len(final_evidence))
    return final_evidence


if __name__ == "__main__":
    corpus = get_corpus(
        "/home/venktesh/retrieval_benchmarking/data/corpus_stance.jsonl"
    )
    final_deduplicated_corpus = form_corpus(corpus)
    with open("../data/final_corpus.json", "w") as f:
        json.dump(final_deduplicated_corpus, f, indent=4)
    print("final_deduplicated_corpus", len(final_deduplicated_corpus))
