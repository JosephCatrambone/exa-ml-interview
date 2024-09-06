import heapq
import json
import os
import struct
import time
from dataclasses import dataclass
from typing import Optional

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
    def __init__(self, biencoder: BaseModelMixin, filename: Optional[str] = None):
        self.model = biencoder
        self.embedding_size = biencoder.get_embedding_size()

        # Init vec storage:
        if filename is None:
            self.db = sqlite3.connect(":memory:")
        else:
            self.db = sqlite3.connect(filename)
        self.db.row_factory = sqlite3.Row

        embedding_descriptor = ",".join([f"d{i} REAL" for i in range(0, self.embedding_size)])
        self.db.execute(f"CREATE TABLE IF NOT EXISTS corpus_vectors (doc_id INTEGER PRIMARY KEY, {embedding_descriptor});")
        self.db.execute(f"CREATE TABLE IF NOT EXISTS corpus (doc_id INTEGER PRIMARY KEY, text TEXT);")
        vector_insert_columns = ", ".join([f"d{i}" for i in range(0, self.embedding_size)])
        qs = ",".join(["?"] * self.embedding_size)
        self.vector_insert_statement = f"INSERT INTO corpus_vectors(doc_id, {vector_insert_columns}) VALUES (?, {qs});"

    def corpus_size(self):
        doc_count = self.db.execute("SELECT COUNT(*) AS count FROM corpus;").fetchone()['count']
        return doc_count

    def insert_document(self, document_id: int, document_text: str, commit: bool = True):
        start_compute = time.time()
        doc_vec = self.model.embed_documents([document_text])[0]
        end_compute = time.time()

        start_insert = time.time()
        self.db.execute(self.vector_insert_statement, [document_id, ] + doc_vec.tolist())
        self.db.execute("INSERT INTO corpus (doc_id, text) VALUES (?, ?);", (document_id, document_text))
        end_insert = time.time()
        if commit:
            self.db.commit()
        return end_compute - start_compute, end_insert - start_insert


    def embed_and_store_corpus(self, corpus, return_timings: bool = False):
        """corpus_path is assumed to be a dataset (or iterable) with '_id', and 'text'.
        If return_timings is true, returns a dictionary with {'embed_times' and 'insert_times'}."""
        embed_times = list()
        insert_times = list()
        for entry in tqdm(corpus):
            doc_id = int(entry['_id'])
            text = entry['text']
            embed_time, insert_time = self.insert_document(doc_id, text, commit=False)  # One commit at the end.
            embed_times.append(embed_time)
            insert_times.append(insert_time)
        self.db.commit()
        if return_timings:
            return {'embed_times': embed_times, 'insert_times': insert_times}

    def search(self, query: str, k: int) -> list[Result]:
        embed_start = time.time()
        query_vec = self.model.embed_queries([query])[0].tolist()
        embed_end = time.time()

        search_start = time.time()
        #drange_string = "AND ".join([f"d{i} > {v - VRANGE} AND d{i} < {v + VRANGE} " for i, v in enumerate(query_vec)])
        distance_query = "+".join([f"ABS(d{i}-({v}))" for i, v in enumerate(query_vec)])
        rows = self.db.execute(f"SELECT doc_id, {distance_query} AS distance FROM corpus_vectors ORDER BY distance ASC LIMIT ?", (k,)).fetchall()
        search_end = time.time()

        results = list()
        for row in rows:
            # TODO: Do a join above.
            doc = self.db.execute("SELECT doc_id, text FROM corpus WHERE doc_id = ?;", (row['doc_id'],)).fetchone()
            results.append(Result(
                document_id=doc['doc_id'],
                document_text=doc['text'],
                score=1.0/(1.0+row['distance']),
                lookup_time_seconds=search_end - search_start,
                embed_time_seconds=embed_end - embed_start,
            ))

        #print(f"Embed took {embed_end-embed_start} seconds. Search took {search_end-search_start} seconds. Postprocessing took {format_end-format_start} seconds.")

        return results


class RetrieverSQLiteVec:
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
        self.db.commit()
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
