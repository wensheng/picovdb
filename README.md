# picovdb
-----

An extremely fast, ultra-lightweight local vector database

## Install

```shell
pip install picovdb
```

## Benchmark

> Embedding Dim: 1024. Hardware: M3 MacBook Air

No FAISS(cpu):

- Inserting `100,000` vectors took about `0.5`s
- Doing 100 queries from `100,000` vectors took roughly `0.7`s (`0.007`s per quiry).

With FAISS:

- Inserting `100,000` vectors took `120`s
- Doing 100 queries from `100,000` vectors took `0.05`s (`0.0005`s or `0.5 millisecond` per quiry).
- Doing 1000 queries from `100,000` vectors in batch mode took `0.2`s (`0.0002`s or `0.2 millisecond` per quiry).


