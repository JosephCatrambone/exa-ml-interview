import os
from typing import Optional, Union

import numpy
import torch

from .base import BaseModelMixin


class BertFinetunedBiencoder(BaseModelMixin):
    def __init__(self, saved_path: Optional[str] = None, device: torch.device = torch.device('cpu'),):
        self.device = device
        if saved_path is None:
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
            self.query_encoder = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased').to(self.device)
            self.doc_encoder = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased').to(self.device)
        else:
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', os.path.join(saved_path, 'tokenizer'))
            self.query_encoder = torch.hub.load('huggingface/pytorch-transformers', 'model', os.path.join(saved_path, 'query_encoder')).to(self.device)
            self.doc_encoder = torch.hub.load('huggingface/pytorch-transformers', 'model', os.path.join(saved_path, 'doc_encoder')).to(self.device)
            # self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', local_path)  # Points to './test/bert_saved_model/', same place as was saved using `save_pretrained('./test/saved_model/')`
        self.query_encoder.eval()
        self.doc_encoder.eval()
        #self.base_size = self.base_model(**self.tokenizer("Initializing...", return_tensors="pt")).last_hidden_state.shape[-1]
        self.embedding_size = self.embed_documents(["test"]).shape[-1]

    # TODO: Hoist all the set train/eval methods to the parent class.
    def set_train_mode(self):
        self.query_encoder.train()
        self.doc_encoder.train()

    def set_eval_mode(self):
        self.query_encoder.eval()
        self.doc_encoder.eval()

    def save(self, save_path):
        self.tokenizer.save_pretrained(os.path.join(save_path, 'tokenizer'))
        self.query_encoder.save_pretrained(os.path.join(save_path, 'query_encoder'))
        self.doc_encoder.save_pretrained(os.path.join(save_path, 'doc_encoder'))

    def get_model_identifier(self) -> str:
        return "bertbasebiencoder"

    def embed_queries(self, queries: list[str]) -> numpy.ndarray:
        return self._enc(self.query_encoder, queries).cpu().detach().numpy()

    def embed_documents(self, docs: list[str]) -> numpy.ndarray:
        return self._enc(self.doc_encoder, docs).cpu().detach().numpy()

    def infer_training_batch(self, queries: list[str], documents: list[str], targets: list[float], loss_fn) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        query_embeddings = self._enc(self.query_encoder, queries)
        document_embeddings = self._enc(self.doc_encoder, documents)
        target = torch.Tensor(targets).to(self.device)
        return query_embeddings, document_embeddings, loss_fn(query_embeddings, document_embeddings, target)

    def _enc(self, model, texts: list[str]) -> torch.Tensor:
        tokenizer_out = self.tokenizer.batch_encode_plus(texts, padding=True, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        out = model(**tokenizer_out)
        return out.last_hidden_state[:, -1, :]

    def get_embedding_size(self) -> int:
        return self.embedding_size

    def embed(self, text: list[str]) -> numpy.ndarray:
        raise Exception("Call encode_documents or encode_queries instead.")
