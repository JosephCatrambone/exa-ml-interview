import heapq
import json
import os
import struct
import time
from dataclasses import dataclass

import numpy
import sqlite3
import sqlite_vec
from tqdm import tqdm

from models import BaseModelMixin


@dataclass
class Result:
    document_id: int
    document_text: str
    score: float
    embed_time_seconds: float = 0.0
    lookup_time_seconds: float = 0.0


class Retriever:
    def __init__(self, model: BaseModelMixin):
        self.model = model
        self.id_vec_tuples = list()
        self.doc_id_to_text = dict()
        self.restore()

    def restore(self):
        db_file = self.model.get_model_identifier() + ".json"
        if os.path.isfile(db_file):
            with open(db_file, 'rt') as fin:
                data = json.load(fin)
                self.id_vec_tuples = [(int(did[0]), numpy.asarray(did[1])) for did in data['id_vec_tuples']]
                self.doc_id_to_text = {int(k):v for k,v in data['doc_id_to_text']}

    def save(self):
        db_file = self.model.get_model_identifier() + ".json"
        with open(db_file, 'wt') as fout:
            json.dump({
                "id_vec_tuples": [(str(did[0]), did[1].tolist()) for did in self.id_vec_tuples],
                "doc_id_to_text": {str(k):v for k,v in self.doc_id_to_text.items()},
            },
                fout
            )

    def embed_and_store_corpus(self, corpus, return_timings: bool = False):
        """corpus_path is assumed to be a dataset (or iterable) with '_id', and 'text'.
        If return_timings is true, returns a dictionary with {'embed_times' and 'insert_times'}."""
        embed_times = list()
        insert_times = list()
        for idx, entry in tqdm(enumerate(corpus)):
            doc_id = int(entry['_id'])
            text = entry['text']
            # TODO: Should we separate embeddings?
            start_compute = time.time()
            doc_vec = self.model.embed_documents(text)
            end_compute = time.time()

            start_insert = time.time()
            self.id_vec_tuples.append((doc_id, doc_vec))
            self.doc_id_to_text[doc_id] = text
            end_insert = time.time()

            embed_times.append(end_compute - start_compute)
            insert_times.append(end_insert - start_insert)

            if idx % 10000 == 0:
                self.save()
        self.save()

        if return_timings:
            return {'embed_times': embed_times, 'insert_times': insert_times}

    def search(self, query: str, k: int) -> list[Result]:
        start_embed = time.time()
        query_vec = self.model.embed_queries(query)
        end_embed = time.time()

        start_search = time.time()
        result_heap = list()
        for (doc_id, doc_vec) in self.id_vec_tuples:
            score = self.model.score_match(query_vec, doc_vec)
            doc_text = self.doc_id_to_text[doc_id]
            heapq.heappush(result_heap, (score, doc_id, doc_text))
            if len(result_heap) > k:
                heapq.heappop(result_heap)
        end_search = time.time()

        results = list()
        while result_heap:
            r = heapq.heappop(result_heap)
            results.append(Result(
                document_id=r[1],
                document_text=r[2],
                score=r[0],
                embed_time_seconds=end_embed-start_embed,
                lookup_time_seconds=end_search-start_search,
            ))
        return results


class RetrieverSQLite:
    def __init__(self, biencoder: BaseModelMixin):
        self.model = biencoder

        # Init vec storage:
        #self.db = sqlite3.connect(":memory:")
        self.db = sqlite3.connect(self.model.get_model_identifier() + ".db")
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)

        self.db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS corpus_vectors USING vec0(embedding float[{self.model.get_embedding_size()}]);")
        self.db.execute(f"CREATE TABLE IF NOT EXISTS corpus (doc_id INTEGER PRIMARY KEY, text TEXT);")

    def embed_and_store_corpus(self, corpus, return_timings: bool = False):
        """corpus_path is assumed to be a dataset (or iterable) with '_id', and 'text'.
        If return_timings is true, returns a dictionary with {'embed_times' and 'insert_times'}."""
        embed_times = list()
        insert_times = list()
        for entry in tqdm(corpus):
            doc_id = int(entry['_id'])
            text = entry['text']
            # TODO: Should we separate embeddings?
            start_compute = time.time()
            doc_vec = self.model.embed_documents(text)
            end_compute = time.time()
            start_insert = time.time()
            self.db.execute(
                "INSERT INTO corpus_vectors(rowid, embedding) VALUES (?, ?);",
                (doc_id, Retriever.list_to_sqlitevec(doc_vec.tolist())),
            )
            self.db.execute(
                "INSERT INTO corpus (doc_id, text) VALUES (?, ?);", (doc_id, text)
            )
            end_insert = time.time()
            embed_times.append(end_compute - start_compute)
            insert_times.append(end_insert - start_insert)
        if return_timings:
            return {'embed_times': embed_times, 'insert_times': insert_times}

    def search(self, query: str, k: int) -> list[Result]:
        embed_start = time.time()
        query_vec = self.model.embed_queries(query)
        embed_end = time.time()

        search_start = time.time()
        rows = self.db.execute("""
              SELECT
                rowid,
                distance
              FROM corpus_vectors
              WHERE embedding MATCH ?
              ORDER BY distance
              LIMIT ?
            """,
            (Retriever.list_to_sqlitevec(query_vec.tolist()), k),
        ).fetchall()
        search_end = time.time()

        format_start = time.time()
        results = list()
        for row in rows:
            # TODO: Do a join above.
            postprocess_start = time.time()
            doc = self.db.execute("SELECT doc_id, text FROM corpus WHERE doc_id = ?;", (row['rowid'],)).fetchone()
            postprocess_end = time.time()
            results.append(Result(
                document_id=doc['doc_id'],
                document_text=doc['text'],
                score=1.0/(1.0+row['distance']),
                lookup_time_seconds=search_end - search_start,
                embed_time_seconds=embed_end - embed_start,
                #postprocess_time_seconds=postprocess_end - postprocess_start,
            ))
        format_end = time.time()

        #print(f"Embed took {embed_end-embed_start} seconds. Search took {search_end-search_start} seconds. Postprocessing took {format_end-format_start} seconds.")

        return results

    @staticmethod
    def list_to_sqlitevec(vector: list[float]):
        return struct.pack("%sf" % len(vector), *vector)
