# A pre-made sentence transformer to get a rough estimate of the performance we might expect for a really good tune.
from typing import Optional

import numpy
import torch
from sentence_transformers import SentenceTransformer

from .base import BaseModelMixin


class CheatSentenceTransformer(BaseModelMixin):
    def get_model_identifier(self) -> str:
        return "mpnetbasecheatsentencetransformer"

    def get_embedding_size(self) -> int:
        return self.embed_size

    def embed(self, text: list[str]) -> numpy.ndarray:
        return self.model.encode(text)

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka").to(device)
        self.embed_size = self.embed(["test"])[0].shape[-1]
