import itertools
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn
from datasets import load_dataset  # NOTE: NOT THE HF DATASETS!
from tqdm import tqdm

from data import load_train_data
from models import BertFinetunedBiencoder

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None


@dataclass
class ExperimentalRun:
    learning_rate: float
    batch_size: int
    epochs: int
    cosine_loss_margin: float


run_details = ExperimentalRun(
    learning_rate=0.1,
    batch_size=8,
    epochs=10,
    cosine_loss_margin=0.5,
)


try:
    import wandb
    wandb.init(project="vector_search", config=asdict(run_details))
except ImportError:
    wandb = None


def train():
    # Keep this locked so we don't have to worry about cheating by seeing our validation set?
    fixed_generator = torch.Generator().manual_seed(42)

    # Load our base model.
    device = torch.device('cuda')
    model = BertFinetunedBiencoder(saved_path=None, device=device, )

    # Load and preprocess data.
    print("Preprocessing data...")
    corpus_id_to_text, query_id_to_text, qcm_train, qcm_validate = load_train_data(0.01, 0.3, seed=42)
    qcm_train = torch.utils.data.DataLoader(qcm_train, batch_size=run_details.batch_size, shuffle=True)
    qcm_validate = torch.utils.data.DataLoader(qcm_validate, batch_size=run_details.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(list(model.doc_encoder.parameters()) + list(model.query_encoder.parameters()), lr=run_details.learning_rate)
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.5, reduction='mean')

    tick = 0
    for epoch_idx in range(0, run_details.epochs):
        total_epoch_loss = 0.0
        for batch_idx, batch in enumerate(qcm_train):
            optimizer.zero_grad()

            # Basic and easy, but not performant because all scores are '1' in training.
            query_texts = [query_id_to_text[b] for b in batch['query-id']]
            corpus_texts = [corpus_id_to_text[b] for b in batch['corpus-id']]
            scores = batch['score']

            # Hack for contrastive loss:
            # We take the query and corpus texts in order, then duplicate the query texts and offset the corpus texts by one.
            # query_texts = [query_id_to_text[b] for b in itertools.chain(batch['query-id'], batch['query-id'])]
            # corpus_texts = [corpus_id_to_text[b] for b in itertools.chain(batch['corpus-id'], reversed(batch['corpus-id']))]
            # target_cosine_similarity = torch.Tensor([1] * len(batch) + [0]*len(batch)).to(device)

            batch_loss = model.infer_training_batch(query_texts, corpus_texts, scores, loss_fn)
            batch_loss.backward()
            total_epoch_loss += batch_loss.item()
            optimizer.step()

            if wandb is not None and batch_idx % 100 == 0:
                tick += 1
                wandb.log({
                        "batch_loss": batch_loss.item(),
                        "batch_idx": batch_idx,
                        "epoch_idx": epoch_idx,
                    },
                    tick,
                    commit=True
                )
            elif batch_idx % 1000 == 0:
                print(f"Batch idx: {batch_idx} - Loss: {batch_loss.item()}")

            # TODO: Run our validation here.
    torch.save(model.doc_encoder, "./doc_encoder")
    torch.save(model.query_encoder, "./query_encoder")


if __name__ == "__main__":
    train()
