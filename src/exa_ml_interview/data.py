import itertools
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import Subset
from tqdm import tqdm


def load_train_data(percentage_train: float, percentage_validate: float, seed: Optional[int] = None):
    split_kwargs = {}
    if seed is not None:
        split_kwargs['generator'] = torch.Generator().manual_seed(seed)
    query_corpus_matches = load_dataset("mteb/msmarco", "default", split='train')
    qcm_train, qcm_validate, _ = torch.utils.data.random_split(
        query_corpus_matches,
        [percentage_train, percentage_validate, 1 - percentage_train - percentage_validate],
        **split_kwargs
    )

    #corpus_id_to_text = {q['_id']: q['text'] for q in tqdm(load_dataset("mteb/msmarco", "corpus", split="corpus"))}
    #query_id_to_text = {q['_id']: q['text'] for q in tqdm(load_dataset("mteb/msmarco", "queries", split="queries"))}
    # query_corpus_matches = {int(qcm['query-id']): int(qcm['corpus-id']) for qcm in load_dataset("mteb/msmarco", "default")['train']}
    corpus_id_to_text = dict()
    query_id_to_text = dict()
    for q in tqdm(load_dataset("mteb/msmarco", "corpus", split="corpus[:25%]")):  # NOTE: 25%!!!
        corpus_id_to_text[q['_id']] = q['text']

    for q in tqdm(load_dataset("mteb/msmarco", "queries", split="queries")):
        query_id_to_text[q['_id']] = q['text']

    return corpus_id_to_text, query_id_to_text, qcm_train, qcm_validate
