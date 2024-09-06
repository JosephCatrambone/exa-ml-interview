# A pre-made sentence transformer to get a rough estimate of the performance we might expect for a really good tune.

from sentence_transformers import SentenceTransformer

from .base import FunctionCallEmbedModel, cosine_similarity


class CheatSentenceTransformer(FunctionCallEmbedModel):
    def __init__(self):
        model = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka").to('cuda')
        def embed(text: list[str]):
            return model.encode(text)
        super().__init__(
            embed_fn=embed,
            score_match_fn=cosine_similarity,
            embedding_size=embed(["test"])[0].shape[-1],
            name="mpnetbasecheatsentencetransformer"
        )
