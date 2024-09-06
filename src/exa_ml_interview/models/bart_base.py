from typing import Optional

import numpy
import torch
from transformers import BartTokenizerFast, BartModel
#tqdm boto3 requests regex sentencepiece sacremoses

from .base import BaseModelMixin


class BartBase(BaseModelMixin):
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        model_name = 'facebook/bart-base'
        # The model to beat: SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka", truncate_dim=768)
        self.tokenizer = BartTokenizerFast.from_pretrained(model_name)
        self.base_model = BartModel.from_pretrained(model_name)
        self.base_model.eval()
        self.base_size = self.base_model(**self.tokenizer("Initializing...", return_tensors="pt")).last_hidden_state.shape[-1]
        self.base_model.to(device)

    def get_model_identifier(self) -> str:
        return "bartuntrainedbasic"

    def get_embedding_size(self) -> int:
        return self.base_size

    def embed(self, text: list[str]) -> numpy.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        #encoded = self.base_model(**inputs).last_hidden_state[0, -1, :].detach().numpy()
        encoded = self.base_model(**inputs).encoder_last_hidden_state[:, -1, :].detach().cpu().numpy()
        return encoded
