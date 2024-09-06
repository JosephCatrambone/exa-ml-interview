import itertools
import os
import random
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn
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
    learning_rate=0.01,
    batch_size=16,
    epochs=10,
    cosine_loss_margin=0.5,
)


try:
    import wandb
    wandb.init(project="vector_search", config=asdict(run_details))
except ImportError:
    wandb = None


def constrastive_learning(a: torch.Tensor, b: torch.Tensor, diagonal_scores: torch.Tensor) -> torch.Tensor:
    # Want A to match B along the diagonal.
    # Assumes that A is of the shape (batch_size, embedding_length)
    assert a.shape == b.shape
    target_diagonal = torch.diag(diagonal_scores)
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    model_diagonal = (a @ b.T)
    # TODO: This is not real contrastive loss, but we're still using cosine embeddings.
    return torch.nn.functional.mse_loss(model_diagonal, target_diagonal, reduction='mean')


def train(device_id: int):
    # Load our base model.
    device = torch.device(f'cuda:{device_id}')
    model = BertFinetunedBiencoder(saved_path=None, device=device, )

    # Load and preprocess data.
    print("Preprocessing data...")
    corpus_id_to_text, query_id_to_text, qcm_train, qcm_validate = load_train_data(0.7, 0.299, seed=42)
    qcm_train = torch.utils.data.DataLoader(qcm_train, batch_size=run_details.batch_size, shuffle=True)
    qcm_validate = torch.utils.data.DataLoader(qcm_validate, batch_size=run_details.batch_size, shuffle=True)

    # Hack for randomly choosing entries:
    all_corpus_keys = list(set(list(corpus_id_to_text.keys())))
    all_query_keys = list(set(list(query_id_to_text.keys())))

    optimizer = torch.optim.Adam(list(model.doc_encoder.parameters()) + list(model.query_encoder.parameters()), lr=run_details.learning_rate)
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=run_details.cosine_loss_margin, reduction='mean')

    print("Training...")
    tick = 0
    best_validation_loss = float('+inf')
    try:
        for epoch_idx in tqdm(range(0, run_details.epochs)):
            total_epoch_loss = 0.0
            model.set_train_mode()
            for batch_idx, batch in tqdm(enumerate(qcm_train)):
                optimizer.zero_grad()

                # If we don't have a document in the corpus, we replace that with a random match and give it a cosine of zero.
                # This gives us our negative examples.
                found_positive = False
                found_negative = False
                query_texts = list()
                corpus_texts = list()
                scores = list()
                for qid, cid, s in zip(batch['query-id'], batch['corpus-id'], batch['score']):
                    if qid in query_id_to_text and cid in corpus_id_to_text:
                        query_texts.append(query_id_to_text[qid])
                        corpus_texts.append(corpus_id_to_text[cid])
                        scores.append(s)
                        found_positive = True
                    else:
                        query_texts.append(query_id_to_text[random.choice(all_query_keys)])
                        corpus_texts.append(corpus_id_to_text[random.choice(all_corpus_keys)])
                        scores.append(0)
                        found_negative = True
                if not found_positive and found_negative:
                    print(f"WARNING: BATCH {batch_idx} CONTAINED ALL POSITIVE OR ALL NEGATIVE!  {found_positive} / {found_negative}")

                # Infer and backprop:
                query_embeddings, doc_embeddings, batch_loss = model.infer_training_batch(query_texts, corpus_texts, scores, loss_fn)
                #batch_loss = constrastive_learning(query_embeddings, doc_embeddings, torch.Tensor(scores).to(model.device))
                batch_loss.backward()
                total_epoch_loss += batch_loss.item()
                optimizer.step()

                # Log results:
                if wandb is not None:
                    tick += 1
                    wandb.log(
                        {"batch_loss": batch_loss.item(), },
                        step=tick,
                        commit=(tick % 100 == 0)
                    )
                elif batch_idx % 100 == 0:
                    print(f"Batch idx: {batch_idx} - Loss: {batch_loss.item()}")

            model.set_eval_mode()
            total_validation_loss = 0.0
            for batch in qcm_validate:
                query_texts = list()
                corpus_texts = list()
                scores = list()
                for qid, cid in zip(batch['query-id'], batch['corpus-id']):
                    if qid in query_id_to_text and cid in corpus_id_to_text:
                        query_texts.append(query_id_to_text[qid])
                        corpus_texts.append(corpus_id_to_text[cid])
                        scores.append(1)
                    else:
                        query_texts.append(query_id_to_text[random.choice(all_query_keys)])
                        corpus_texts.append(
                            corpus_id_to_text[random.choice(all_corpus_keys)])
                        scores.append(0)
                _, _, validation_loss = model.infer_training_batch(query_texts, corpus_texts, scores, loss_fn)
                total_validation_loss += validation_loss.item()
                if wandb is not None:
                    tick += 1
                    wandb.log(
                        {"validation_loss": validation_loss.item()},
                        step=tick,
                        commit=True
                    )
            print(f"Total validation loss: {total_validation_loss}")
            if total_validation_loss < best_validation_loss:
                best_validation_loss = total_validation_loss
                model.save(f"checkpoint_best_{epoch_idx}")
                #torch.save(model.doc_encoder, "./doc_encoder_{epoch_idx}.pt")
                #torch.save(model.query_encoder, "./query_encoder_{epoch_idx}.pt")
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as e:
        model.save(f"model_crash_epoch_{epoch_idx}_save_{random.randint(0, 2**64)}")
        raise e

    model_save_name = f"model_final_{random.randint(0, 2**64)}"
    print(f"Saving model as {model_save_name}")
    model.save(model_save_name)


if __name__ == "__main__":
    train(int(sys.argv[1]))
