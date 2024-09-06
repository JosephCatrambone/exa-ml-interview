# Retrospective

## What was finished:

- train_reviewer will, as per the name, start to train a biencoder (based on bert cased) using the mteb/msmarco dataset.
- retriever will index and fetch docs when provided with a model.
- evaluate_retriever will try and run the top-k on the test set. 

## What remains to do:

- Cosine similarity training can overfit but doesn't give good results.  Validation flat, but can overfit on a single example.  Same with contrastive learning.
- Need to check logic on the evaluate top-k.  
- Three models still training.

# Model things attempted:

- 'The dumb thing' -- just a plain untrained set of bert embeddings for a baseline.
- 'Monoencoder' with a single encoder for query and document.
- Cosine Similarity loss on cased bert embeddings.
- Contrastive loss on cased bert.  (This did not work as well as I'd like.)
- (Also tried pre-trained embed model for sanity on the retrieval and scoring system -- not used for training or fine tuning.)

## What went well:

- No issues with host machine stability!
- Easy to get started, clone repo, pull dataset, etc.
- Retriever (interactive) came together nicely.

## What could have been better:

- A few training crashes (3x OOM, 1 or 2 tokenizer problems.)
- Performance plateaus on cosine sim and contrastive loss models.  Results are subpar in practice.  Need to investigate, but hopeful for model 38!
- Might have spent too long hammering out the base model -- mentally splitting the difference between the engineering side and the ML practice side instead of focusing on training from the first moment.
- Some regrets over starting with Adam instead of SGD -- may have interacted badly with some things?

---

Original document:

# Exa ML Interview

This is the starter repo for the onsite Exa ML interview! You will fork this and work on it yourself.

Your goal is to train a model/system for document retrieval over the [MS Marco V1](https://huggingface.co/datasets/mteb/msmarco) dataset. In particular, at the end of this you will implement a class `Retriever`

```python
@dataclass
class Result:
    document_id: str
    document_text: str
    score: float

class Retriever:
    def __init__(self):
        pass

    def embed_and_store_corpus(self, corpus_path):
        pass

    def search(query: str, k: int) -> List[Result]:
        pass

```

After the interview, I will call search() on 1000 queries from the MS-Marco test set, and compute the recall@1. Your goal is to produce a system that has the highest recall@1. 

You are welcome to use any approach you want for this problem -- it will likely involve a transformer-backed bi-encoder at some level, but are welcome to use any architecture or other system to improve the performance. You do have the following constraints:

1. search must return a result in <1 second (95% of the time)
2. You will have access to a 4 A10G's.
3. You are welcome to used pretrained LLMs in any capacity here, but you can't start with a pretrained embedding-specific model
4. You should only train on the [MS-Marco](https://huggingface.co/datasets/mteb/msmarco-v2/viewer/queries) train set -- or any synthetic data you generate

JC Notes:
- Latency < Top k retrieval
- Boring to just grab a pretrained encoder; show good data science.

## Evaluation

What we're looking for in this interview is

1) Well written, readable, working code
2) That gets good accuracy on the evaluation
3) That indicates good ML research -- you should try out approaches, from the literature or your own ideas, and validate they work

In particular, I recommend getting a simple bi-encoder retriever training first, and once you have that running then try out novel improvements, either to the retrieval architecture, the data, the training code, or the model code. We are looking for a system that gets good accuracy, but beyond that we're really looking for evidence that you can do good applied ML research -- we're interested in seeing what experiments and novel solutions you come up with!

Feel free to ask questions! This is meant to be a collaborative exercise.

Plan:
- Pull dataset and look at how it fits together.
- Get an evaluation setup running that looks at latency, recall.
- Get a baseline running so we have some numbers to compare.
