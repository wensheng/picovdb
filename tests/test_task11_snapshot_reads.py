import threading
import time
import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def test_snapshot_query_stable_under_writes(tmp_path):
    dim = 64
    db = PicoVectorDB(embedding_dim=dim, storage_file=str(tmp_path / "snap"), no_faiss=True)
    items = [
        {K_VECTOR: np.random.rand(dim).astype(np.float32), K_ID: f"id{i}"}
        for i in range(100)
    ]
    db.upsert(items)

    stop = threading.Event()
    errors = []

    def reader():
        q = np.random.rand(dim).astype(np.float32)
        while not stop.is_set():
            res = db.query(q, top_k=5)
            # ensure docs exist and ids are strings
            for r in res:
                if not isinstance(r[K_ID], (str, int)):
                    errors.append("bad id type")
            time.sleep(0.005)

    def writer():
        i = 100
        while not stop.is_set():
            # alternate upsert and delete
            db.upsert([{K_VECTOR: np.random.rand(dim).astype(np.float32), K_ID: f"id{i}"}])
            db.delete([f"id{i-50}"]) if i > 50 else None
            i += 1
            time.sleep(0.002)

    t_r = threading.Thread(target=reader)
    t_w = threading.Thread(target=writer)
    t_r.start(); t_w.start()
    time.sleep(0.1)
    stop.set()
    t_r.join(); t_w.join()

    assert not errors

