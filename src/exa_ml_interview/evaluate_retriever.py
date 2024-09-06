from dataclasses import asdict, dataclass
import json

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from models import BaseModelMixin, BartBase, WorstModel, CheatSentenceTransformer
from retriever import Retriever, Result


@dataclass
class SummaryStats:
    total: float
    mean: float
    stddev: float
    min: float
    max: float
    median: float

    @classmethod
    def new_from_list(cls, entries: list[float]):
        min_value: float = float('inf')
        max_value: float = float('-inf')
        total = 0.0
        for e in entries:
            total += e
            min_value = min(e, min_value)
            max_value = max(e, max_value)
        mean = total / len(entries)
        median = sorted(entries)[int(len(entries) / 2)]
        stddev = 0.0
        for e in entries:
            stddev += (e - mean) ** 2
        stddev = (stddev / len(entries)) ** 0.5
        return cls(total, mean, stddev, min_value, max_value, median)


@dataclass
class EvaluationRun:
    raw_time_to_compute_embeddings_seconds: list[float]
    raw_time_to_perform_lookup_seconds: list[float]
    index_of_correct_match: list[int]  # If we fetch at most k entries, this is the position of the correct entry, else -1.  Let's us easily compute the top-k results.
    correct_item_scores: list[float]  # What is the score of the correct item?  -1 if we don't have an item.
    item_zero_scores: list[float]  # What is the score of the item in place zero?
    total_queries: int


def run_benchmark(model: BaseModelMixin, documents: Dataset, qid_to_text: dict[int, str], test_matches: dict[int, int], k: int) -> EvaluationRun:
    # test_matches should be query_id to corpus_id
    # NOTE: These are not batched.  Batching will (should) decrease the average latency.
    retriever = Retriever(model)
    timings = retriever.embed_and_store_corpus(documents, return_timings=True)

    total_queries = 0
    correct_recalls = 0
    missed_recalls = 0
    correct_indices = list()  # If we find a result, push it, else push -1
    item_zero_score = list()
    correct_item_scores = list()
    lookup_times = list()

    for qid, cid in tqdm(test_matches.items()):
        if qid not in qid_to_text:
            continue
        qtext = qid_to_text[qid]
        total_queries += 1
        results = retriever.search(qtext, k)
        found_expected_doc = False
        for idx, r in enumerate(results):
            lookup_times.append(r.lookup_time_seconds)
            if r.document_id == cid:
                found_expected_doc = True
                correct_indices.append(idx)
                correct_item_scores.append(r.score)
            if idx == 0:
                item_zero_score.append(r.score)
        if found_expected_doc:
            correct_recalls += 1
        else:
            missed_recalls += 1

    return EvaluationRun(
        raw_time_to_compute_embeddings_seconds=[], #timings['embed_times'],
        raw_time_to_perform_lookup_seconds=lookup_times,
        index_of_correct_match=correct_indices,
        item_zero_scores=item_zero_score,
        correct_item_scores=correct_item_scores,
        total_queries=total_queries,
    )


if __name__ == '__main__':
    print("E2E with worst encoder.")
    print("Loading model...")
    #model = BartBase(device=torch.device('cuda'))
    model = CheatSentenceTransformer()

    print("Loading data...")
    #corpus = load_dataset("mteb/msmarco", "corpus")['corpus']
    #queries = load_dataset("mteb/msmarco", "queries")['queries']
    #query_corpus_matches = {int(qcm['query-id']): int(qcm['corpus-id']) for qcm in load_dataset("mteb/msmarco", "default")['test']}
    corpus = load_dataset("mteb/msmarco", "corpus", split="corpus")
    query_id_to_text = {int(q['_id']): q['text'] for q in load_dataset("mteb/msmarco", "queries", split="queries")}
    query_corpus_matches = load_dataset("mteb/msmarco", "default", split='test')
    query_corpus_top_match = dict()
    for qcm in query_corpus_matches:
        qid = int(qcm['query-id'])
        cid = int(qcm['corpus-id'])
        score = float(qcm['score'])
        if (qid, cid) not in query_corpus_top_match or query_corpus_top_match[(qid, cid)] < score:
            query_corpus_top_match[(qid, cid)] = score
    query_corpus_matches = {qc[0]: qc[1] for qc, score in query_corpus_top_match.items() if score > 0.0}

    print("Running benchmark...")
    out = run_benchmark(model, corpus, query_id_to_text, query_corpus_matches, 10)
    with open(f"results_k5_bartbase.json", 'wt') as fout:
        json.dump(asdict(out), fout, indent=2)
