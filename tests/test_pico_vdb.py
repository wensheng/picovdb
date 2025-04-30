import os
from pathlib import Path

import numpy as np

from picovdb import PicoVectorDB, K_METRICS, K_ID, K_VECTOR


def test_init():
    from time import time

    data_len = 1000
    fake_dim = 1024

    start = time()
    a = PicoVectorDB(fake_dim)
    print("Load", time() - start)

    fake_embeds = np.random.rand(data_len, fake_dim)
    fakes_data = [{K_VECTOR: fake_embeds[i], K_ID: str(i)} for i in range(data_len)]
    query_data = fake_embeds[data_len // 2]
    start = time()
    r = a.upsert(fakes_data)
    print("Upsert", time() - start)
    a.save()

    a = PicoVectorDB(fake_dim)

    start = time()
    r = a.query(query_data, 10, better_than=0.01)
    assert r[0][K_ID] == str(data_len // 2)
    print(r)
    assert len(r) <= 10
    for d in r:
        assert d[K_METRICS] >= 0.01
    os.remove("picovdb.meta.json")
    os.remove("picovdb.ids.json")


def test_same_upsert():
    from time import time

    data_len = 1000
    fake_dim = 1024

    start = time()
    a = PicoVectorDB(fake_dim)
    print("Load", time() - start)

    fake_embeds = np.random.rand(data_len, fake_dim)
    fakes_data = [{K_VECTOR: fake_embeds[i]} for i in range(data_len)]
    r1 = a.upsert(fakes_data)
    assert len(r1["insert"]) == len(fakes_data)
    fakes_data = [{K_VECTOR: fake_embeds[i]} for i in range(data_len)]
    r2 = a.upsert(fakes_data)
    assert r2["update"] == r1["insert"]


def test_get():
    a = PicoVectorDB(1024)
    a.upsert(
        [
            {K_VECTOR: np.random.rand(1024), K_ID: str(i), "content": i}
            for i in range(100)
        ]
    )
    r = a.get(["0", "1", "2"])
    assert len(r) == 3
    assert r[0]["content"] == 0
    assert r[1]["content"] == 1
    assert r[2]["content"] == 2


def test_delete():
    Path.unlink('picovdb.vecs.npy', missing_ok=True)
    a = PicoVectorDB(1024)
    a.upsert(
        [
            {K_VECTOR: np.random.rand(1024), K_ID: str(i), "content": i}
            for i in range(100)
        ]
    )
    a.delete(["0", "50", "90"])

    r = a.get(["0", "50", "90"])
    assert len(r) == 0
    assert len(a) == 97


def test_cond_filter():
    data_len = 10
    fake_dim = 1024

    a = PicoVectorDB(fake_dim)
    fake_embeds = np.random.rand(data_len, fake_dim)
    cond_filer = lambda x: x[K_ID] == 1

    fakes_data = [{K_VECTOR: fake_embeds[i], K_ID: i} for i in range(data_len)]
    query_data = fake_embeds[data_len // 2]
    a.upsert(fakes_data)

    assert len(a) == data_len
    r = a.query(query_data, 10, better_than=0.01)
    assert r[0][K_ID] == data_len // 2

    r = a.query(query_data, 10, where=cond_filer)
    assert r[0][K_ID] == 1


def test_additonal_data():
    data_len = 10
    fake_dim = 1024

    a = PicoVectorDB(fake_dim)

    a.store_additional_data(a=1, b=2, c=3)
    a.save()

    a = PicoVectorDB(fake_dim)
    assert a.get_additional_data() == {"a": 1, "b": 2, "c": 3}
    os.remove("picovdb.meta.json")
    os.remove("picovdb.vecs.npy")


def remove_non_empty_dir(dir_path):
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
    os.rmdir(dir_path)

