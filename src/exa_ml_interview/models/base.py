from abc import ABC, abstractmethod

import numpy
import torch


class BaseModelMixin(ABC):
    @abstractmethod
    def get_model_identifier(self) -> str:
        """Return a unique string identifier for the model. [a-z]+"""
        ...

    @abstractmethod
    def get_embedding_size(self) -> int:
        # The DOCUMENT/CORUPUS embedding size.
        ...

    def embed_queries(self, query: list[str]) -> numpy.ndarray:
        """Generate the embedding vector for the given query string.  If this model doesn't have a separate query encoder, fall back to the 'embed' method."""
        return self.embed(query)

    def embed_documents(self, document: list[str]) -> numpy.ndarray:
        """Generate the embedding vector for the given query string.  If this model doesn't have a separate document encoder, fall back to the 'embed' method."""
        return self.embed(document)

    @abstractmethod
    def embed(self, text: list[str]) -> numpy.ndarray:
        ...

    def infer_training_batch(self, queries: list[str], documents: list[str], targets: list[float], loss_fn) -> torch.Tensor:
        """
        query_embeddings = self.embed_queries(queries)
        doc_embeddings = self.embed_documents(documents)
        out = self.score_match(query_embeddings, doc_embeddings)
        return loss_fn(out, targets)
        """
        raise NotImplementedError("Fine tuning not implemented for this model.")

    def score_match(self, query_vector: numpy.ndarray, document_vector: numpy.ndarray) -> numpy.ndarray:
        """Return a score from 0 to 1, with 0 being less relevant and 1 being more relevant.
        We should probably leave this as cosine_similarity since generally the vec DB will be using that for lookup.
        It's only a variable if we want to do something like 'ranking' in the future."""
        # TODO: Maybe this is a bad idea?
        return cosine_similarity(query_vector, document_vector)


class FunctionCallEmbedModel(BaseModelMixin):
    def __init__(self, embed_fn, score_match_fn, embedding_size: int, name: str):
        self.name = name
        self.embedding_size = embedding_size
        self.embed_fn = embed_fn
        self.score_match_fn = score_match_fn

    def get_model_identifier(self) -> str:
        return self.name

    def get_embedding_size(self) -> int:
        return self.embedding_size

    def embed(self, text: list[str]) -> numpy.ndarray:
        return self.embed_fn(text)


class FunctionCallQCEmbedModel(BaseModelMixin):
    def __init__(self, query_embed_fn, document_embed_fn, score_match_fn, embedding_size: int, name: str):
        self.name = name
        self.embedding_size = embedding_size
        self.query_embedding_fn = query_embed_fn
        self.document_embedding_fn = document_embed_fn
        self.score_match_fn = score_match_fn

    def get_model_identifier(self) -> str:
        return self.name

    def get_embedding_size(self) -> int:
        return self.embedding_size

    def embed(self, text: list[str]) -> numpy.ndarray:
        return None

    def embed_queries(self, query: list[str]) -> numpy.ndarray:
        return self.query_embedding_fn(query)

    def embed_documents(self, document: list[str]) -> numpy.ndarray:
        return self.document_embedding_fn(document)

    def score_match(self, query_vector: numpy.ndarray, document_vector: numpy.ndarray) -> numpy.ndarray:
        return self.score_match_fn(query_vector, document_vector)


def cosine_similarity(vec1: numpy.ndarray, vec2: numpy.ndarray, assume_normalized: bool = False) -> numpy.ndarray:
    if assume_normalized:
        vec1_magnitude = 1.0
        vec2_magnitude = 1.0
    else:
        vec1_magnitude = vec1.dot(vec1)**0.5
        vec2_magnitude = vec2.dot(vec2)**0.5
    return numpy.dot(vec1/vec1_magnitude, vec2/vec2_magnitude)
