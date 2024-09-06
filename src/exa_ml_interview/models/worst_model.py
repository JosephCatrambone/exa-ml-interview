import numpy

from .base import BaseModelMixin


class WorstModel(BaseModelMixin):

    def get_model_identifier(self) -> str:
        return "worstmodel"

    def get_embedding_size(self) -> int:
        return 4

    def embed(self, docs: list[str]) -> numpy.ndarray:
        return numpy.asarray([[0.0, 0.0, 0.0, 1.0/(1+len(doc))] for doc in docs])
