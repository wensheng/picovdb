# picovdb
-----

An extremely fast, ultra-lightweight local vector database in Python.

**"extremely fast"**: sub-millisecond query

**"ultra-lighweight"**: One file with only Numpy and one optional dependency [faiss-cpu](https://pypi.org/project/faiss-cpu/). (See faiss note at the end)

## Install

```shell
pip install picovdb
```

## Usage

**Create a db:**

(Use SentenceTransformer embedding as example)
```python
from picovdb import PicoVectorDB  # On Mac, import before any libs that use pytorch
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 256
model = SentenceTransformer('all-MiniLM-L6-v2')
dim = model.get_sentence_embedding_dimension()

with open('A_Christmas_Carol.txt', encoding='UTF8') as f:
    content = f.read()
    num_chunks = len(content) // CHUNK_SIZE + 1
    chunks = [content[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE] for i in range(num_chunks)]
    embeddings = model.encode(chunks)
    data = [
        {
            "_vector_": embeddings[i],
            "_id_": i,
            "content": chunks[i],
        }
        for i in range(num_chunks)
    ]
    db = PicoVectorDB(embedding_dim=dim, storage_file='_acc')
    db.upsert(data)
    db.save()
```

**Query**
```python
from picovdb import PicoVectorDB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
dim = model.get_sentence_embedding_dimension()

db = PicoVectorDB(embedding_dim=dim, storage_file='_acc')
txt = "Are there no prisons? Are there no workhouses?"
emb = model.encode(txt)
q = db.query(emb, top_k=3)
print('query results:', q)
```

## Benchmark

> Embedding Dim: 1024. 

Environment: M3 MacBook Air

1. Pure Python:
   - Inserting `100,000` vectors took about `0.5`s
   - Doing 100 queries from `100,000` vectors took roughly `0.8`s (`0.008`s per quiry).
   - Doing 1000 queries from `100,000` vectors in batch mode took `1.0`s (`0.001`s or `1 millisecond` per quiry).

2. With FAISS(cpu):
   - Inserting `100,000` vectors took `110`s
   - Doing 100 queries from `100,000` vectors took `0.04`s (`0.0004`s or `0.4 millisecond` per quiry).
   - Doing 1000 queries from `100,000` vectors in batch mode took `0.1`s (`0.0001`s or `0.1 millisecond` per quiry).

Environment: Windows PC with CPU Core i7-12700k and old-gen M2 Nvme SSD

1. Pure Python:
   - Inserting `100,000` vectors took about `0.7`s
   - Doing 100 queries from `100,000` vectors took roughly `1.5`s (`0.015`s per quiry).
   - Doing 1000 queries from `100,000` vectors in batch mode took `1.0`s (`0.001`s or `1 millisecond` per quiry).


2. With FAISS(cpu):
   - Inserting `100,000` vectors took `50`s
   - Doing 100 queries from `100,000` vectors took `0.04`s (`0.0004`s or `0.4 millisecond` per quiry).
   - Doing 1000 queries from `100,000` vectors in batch mode took `0.16`s (`0.00016`s or `0.16 millisecond` per quiry).

## FAISS Note

On MacOS, if you use FAISS, please do one of following:

- import `picovdb` before any libraries that use `pytorch` (e.g. `sentence_transformers`, `transformers`, etc) or any packages that use OpenMP.
- set faiss_threads to 1 when initializing PicoVectorDB, e.g. `PicoVectorDB(..., faiss_threads=1)`.
- set env var `PICOVDB_FAISS_THREADS=1` before running your script.

Faiss >=1.10 will segfault on Darwin when using HNSW index with OpenMP multithreading. This is a known issue with FAISS on macOS.

On Windows and Linux, Faiss works fine with other libs that use OpenMP, no special care is needed.
