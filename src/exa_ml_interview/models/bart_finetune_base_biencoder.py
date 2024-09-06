from typing import Optional

import numpy
import torch

from .base import BaseModelMixin


class BartFinetunedBiencoder(BaseModelMixin):
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        #self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        #self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', local_path)  # Points to './test/bert_saved_model/', same place as was saved using `save_pretrained('./test/saved_model/')`
        self.query_encoder = torch.hub.load('pytorch/fairseq', 'bart.base').to(self.device)
        self.doc_encoder = torch.hub.load('pytorch/fairseq', 'bart.large').to(self.device)
        self.query_encoder.eval()
        self.doc_encoder.eval()
        #self.base_size = self.base_model(**self.tokenizer("Initializing...", return_tensors="pt")).last_hidden_state.shape[-1]
        self.embedding_size = self.doc_encoder.extract_features(self.doc_encoder("test")).shape[-1]

    @classmethod
    def init_new(cls):
        pass

    def get_model_identifier(self) -> str:
        return "bartbasebiencoder"

    def embed_query(self, query: str) -> numpy.ndarray:
        last_layer_features = self.query_encoder.extract_features(self.query_encoder.encode(query))
        return last_layer_features[0, -1, :]

    def embed_doc(self, doc: str) -> numpy.ndarray:
        return self.doc_encoder.extract_features(self.doc_encoder.encode(doc))[0, -1, :]

    def get_embedding_size(self) -> int:
        return self.embedding_size

    def embed(self, text: str) -> numpy.ndarray:
        return None
