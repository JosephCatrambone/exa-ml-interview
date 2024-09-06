import json
import sys
from dataclasses import asdict, dataclass

from datasets import load_dataset, Dataset
from tqdm import tqdm

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


def run_interactive():
    k = 10
    corpus_cap = "2000"  # Docs.  Can also do %

    from models import CheatSentenceTransformer
    model = CheatSentenceTransformer()
    retriever = Retriever(model, "sent_tf_2k.db")

    if retriever.corpus_size() == 0:
        corpus = load_dataset("mteb/msmarco", "corpus", split=f"corpus[:{corpus_cap}]")
        retriever.embed_and_store_corpus(corpus)

    while True:
        query = input(">")
        if not query:
            break
        results = retriever.search(query, k)
        print(f"Q: {query}")
        print("-----")
        for idx, r in enumerate(results):
            print(f"{idx}: {r.score} : {r.document_text[:100]}")
        print("-----")


def run_benchmark():
    print("Loading model...")
    from models import CheatSentenceTransformer
    # model = BartBase(device=torch.device('cuda'))
    model = CheatSentenceTransformer()
    # model = BertFinetunedBiencoder()
    k = 5
    corpus_cap = "2000"
    retriever = Retriever(model, "sent_tf_2k.db")

    print("Loading data...")
    # corpus = load_dataset("mteb/msmarco", "corpus")['corpus']
    # queries = load_dataset("mteb/msmarco", "queries")['queries']
    # query_corpus_matches = {int(qcm['query-id']): int(qcm['corpus-id']) for qcm in load_dataset("mteb/msmarco", "default")['test']}
    corpus = load_dataset("mteb/msmarco", "corpus", split=f"corpus[:{corpus_cap}]")
    corpus_ids = set()
    for c in corpus:
        corpus_ids.add(int(c['_id']))
    query_id_to_text = {int(q['_id']): q['text'] for q in load_dataset("mteb/msmarco", "queries", split="queries")}
    query_corpus_matches_raw = load_dataset("mteb/msmarco", "default", split='test')
    query_corpus_match_scores = dict()  # (qid,cid) -> score
    query_corpus_matches = list()
    for qcm in query_corpus_matches_raw:
        qid = int(qcm['query-id'])
        cid = int(qcm['corpus-id'])
        score = float(qcm['score'])
        query_corpus_match_scores[(qid, cid)] = score
        if cid not in corpus_ids:
            continue
        query_corpus_matches.append({'query-id': qid, 'corpus-id': cid, 'score': score})

    print("Running benchmark...")
    # test_matches should be query_id to corpus_id
    # NOTE: These are not batched.  Batching will (should) decrease the average latency.
    timings = retriever.embed_and_store_corpus(corpus, return_timings=True)

    total_queries = 0
    tp = tn = fp = fn = 0
    correct_recalls = 0
    missed_recalls = 0
    correct_rejections = 0
    missed_rejections = 0
    lookup_times = list()

    for qcm in tqdm(query_corpus_matches):
        qid = qcm['query-id']
        cid = qcm['corpus-id']
        score = qcm['score']
        qtext = query_id_to_text[qid]
        total_queries += 1

        results = retriever.search(qtext, k)

        doc_in_top_k = False
        for idx, r in enumerate(results):
            lookup_times.append(r.lookup_time_seconds)
            if r.document_id == cid:
                doc_in_top_k = True
            #item_zero_score.append(r.score)
        if doc_in_top_k:
            if score > 0.0:
                tp += 1
                correct_recalls += 1
            else:
                fp += 1
                missed_rejections += 1
        else:
            if score > 0.0:
                fn += 1
                missed_recalls += 1
            else:
                tn += 1
                correct_rejections += 1
    print(f"TP: {tp}\nFP: {fp}\nTN: {tn}\nFN: {fn}")


if __name__ == '__main__':
    if "--interactive" in sys.argv:
        run_interactive()
    else:
        run_benchmark()
