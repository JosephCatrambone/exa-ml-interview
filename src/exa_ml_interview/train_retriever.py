from dataclasses import asdict, dataclass

import torch
import wandb
from datasets import Dataset, load_dataset

from models import BartBase

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None


@dataclass
class ExperimentalRun:
    learning_rate: float
    epochs: int


run_details = ExperimentalRun(
    learning_rate=0.2,
    epochs=10,
)


try:
    import wandb
    wandb.init(project="vector_search", config=asdict(run_details))
except ImportError:
    wandb = None


def train():
    # Load and preprocess data.
    #corpus = load_dataset("mteb/msmarco", "corpus")['corpus']
    #queries = load_dataset("mteb/msmarco", "queries")['queries']
    #query_corpus_matches = {int(qcm['query-id']): int(qcm['corpus-id']) for qcm in load_dataset("mteb/msmarco", "default")['train']}
    corpus = load_dataset("mteb/msmarco", "corpus", split="corpus")
    query_id_to_text = {int(q['_id']): q['text'] for q in load_dataset("mteb/msmarco", "queries", split="queries")}
    query_corpus_matches = load_dataset("mteb/msmarco", "default", split='train')


    # Load our base model.
    model = BartBase(device="cuda")

if __name__ == "__main__":
    train()