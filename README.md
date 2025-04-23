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


