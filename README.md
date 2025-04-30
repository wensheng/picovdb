# picovdb
-----

An extremely fast, ultra-lightweight local vector database

## Install

```shell
pip install picovdb
```

## Benchmark

> Embedding Dim: 1024. 

Hardware: M3 MacBook Air

1. No FAISS(cpu):
   - Inserting `100,000` vectors took about `0.5`s
   - Doing 100 queries from `100,000` vectors took roughly `0.6`s (`0.006`s per quiry).

2. With FAISS:
   - Inserting `100,000` vectors took `120`s
   - Doing 100 queries from `100,000` vectors took `0.05`s (`0.0005`s or `0.5 millisecond` per quiry).
   - Doing 1000 queries from `100,000` vectors in batch mode took `0.2`s (`0.0002`s or `0.2 millisecond` per quiry).

Hardware: PC with CPU Core i7-12700k and old-gen M2 Nvme SSD

1. No FAISS(cpu):
   - Inserting `100,000` vectors took about `0.7`s
   - Doing 100 queries from `100,000` vectors took roughly `1.3`s (`0.013`s per quiry).

2. With FAISS:
   - Inserting `100,000` vectors took `60`s
   - Doing 100 queries from `100,000` vectors took `0.06`s (`0.0006`s or `0.5 millisecond` per quiry).
   - Doing 1000 queries from `100,000` vectors in batch mode took `0.3`s (`0.0003`s or `0.3 millisecond` per quiry).
