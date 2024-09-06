import itertools
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import Subset, DataLoader
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

    # Only load the queries and documents we need.
    # Still have to iterate over the sets, but at least we save ourselves the trouble.
    required_qids = set()
    required_cids = set()
    for qcm in itertools.chain(qcm_train, qcm_validate):
        required_qids.add(int(qcm['query-id']))
        required_cids.add(int(qcm['corpus-id']))

    #corpus_id_to_text = {q['_id']: q['text'] for q in tqdm(load_dataset("mteb/msmarco", "corpus", split="corpus"))}
    #query_id_to_text = {q['_id']: q['text'] for q in tqdm(load_dataset("mteb/msmarco", "queries", split="queries"))}
    # query_corpus_matches = {int(qcm['query-id']): int(qcm['corpus-id']) for qcm in load_dataset("mteb/msmarco", "default")['train']}
    corpus_id_to_text = dict()
    query_id_to_text = dict()
    for q in tqdm(load_dataset("mteb/msmarco", "corpus", split="corpus")):
        if int(q['_id']) in required_cids:
            corpus_id_to_text[int(q['_id'])] = q['text']

    for q in tqdm(load_dataset("mteb/msmarco", "queries", split="queries")):
        if int(q['_id']) in required_qids:
            query_id_to_text[int(q['_id'])] = q['text']

    return corpus_id_to_text, query_id_to_text, qcm_train, qcm_validate
