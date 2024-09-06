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
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', saved_path)
            self.query_encoder = torch.hub.load('huggingface/pytorch-transformers', 'model', saved_path).to(self.device)
            self.doc_encoder = torch.hub.load('huggingface/pytorch-transformers', 'model', saved_path).to(self.device)
            # self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', local_path)  # Points to './test/bert_saved_model/', same place as was saved using `save_pretrained('./test/saved_model/')`
        self.query_encoder.eval()
        self.doc_encoder.eval()
        #self.base_size = self.base_model(**self.tokenizer("Initializing...", return_tensors="pt")).last_hidden_state.shape[-1]
        self.embedding_size = self.embed_doc(["test"]).shape[-1]

    @classmethod
    def init_new(cls):
        pass

    def get_model_identifier(self) -> str:
        return "bartbasebiencoder"

    def embed_queries(self, queries: list[str]) -> numpy.ndarray:
        return self._enc(self.query_encoder, queries).cpu().detach().numpy()

    def embed_doc(self, docs: list[str]) -> numpy.ndarray:
        return self._enc(self.doc_encoder, docs).cpu().detach().numpy()

    def infer_training_batch(self, queries: list[str], documents: list[str], targets: list[float], loss_fn) -> torch.Tensor:
        query_embeddings = self._enc(self.query_encoder, queries)
        document_embeddings = self._enc(self.doc_encoder, documents)
        target = torch.Tensor(targets).to(self.device)
        return loss_fn(query_embeddings, document_embeddings, target)

    def _enc(self, model, texts: list[str]) -> torch.Tensor:
        tokenizer_out = self.tokenizer.batch_encode_plus(texts, padding=True, return_tensors='pt').to(self.device)
        out = model(**tokenizer_out)
        return out.last_hidden_state[:, -1, :]

    def get_embedding_size(self) -> int:
        return self.embedding_size

    def embed(self, text: list[str]) -> numpy.ndarray:
        return None
