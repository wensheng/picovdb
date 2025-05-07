# picovdb
-----

An extremely fast, ultra-lightweight local vector database in Python.

**"extremely fast"**: sub-millisecond query

**"ultra-lighweight"**: One file with only Numpy and one optional dependency [faiss-cpu](https://pypi.org/project/faiss-cpu/).

## Install

```shell
pip install picovdb
```

## Usage

**Create a db:**

(Use SentenceTransformer embedding as example)
```python
from sentence_transformers import SentenceTransformer
from picovdb import PicoVectorDB

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
db = PicoVectorDB(embedding_dim=dim, storage_file='_acc')
txt = "Are there no prisons? Are there no workhouses?"
emb = model.encode(txt)
q = db.query(emb, top_k=3)
print('query results:', q)
```

## Benchmark

> Embedding Dim: 1024. 

Hardware: M3 MacBook Air

1. Pure Python:
   - Inserting `100,000` vectors took about `0.5`s
   - Doing 100 queries from `100,000` vectors took roughly `0.6`s (`0.006`s per quiry).

2. With FAISS(cpu):
   - Inserting `100,000` vectors took `110`s
   - Doing 100 queries from `100,000` vectors took `0.05`s (`0.0005`s or `0.5 millisecond` per quiry).
   - Doing 1000 queries from `100,000` vectors in batch mode took `0.2`s (`0.0002`s or `0.2 millisecond` per quiry).

Hardware: PC with CPU Core i7-12700k and old-gen M2 Nvme SSD

1. Pure Python:
   - Inserting `100,000` vectors took about `0.7`s
   - Doing 100 queries from `100,000` vectors took roughly `1.3`s (`0.013`s per quiry).

2. With FAISS(cpu):
   - Inserting `100,000` vectors took `50`s
   - Doing 100 queries from `100,000` vectors took `0.05`s (`0.0005`s or `0.5 millisecond` per quiry).
   - Doing 1000 queries from `100,000` vectors in batch mode took `0.3`s (`0.0003`s or `0.3 millisecond` per quiry).
