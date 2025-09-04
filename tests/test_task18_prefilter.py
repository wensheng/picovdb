import numpy as np
import pytest

from picovdb.pico_vdb import PicoVectorDB, _HAS_FAISS, K_VECTOR, K_ID


def test_where_prefilter_limits_candidates_numpy(tmp_path):
    dim = 5
    db = PicoVectorDB(embedding_dim=dim, storage_file=str(tmp_path / "pf"), no_faiss=True)
    items = []
    for i in range(20):
        vec = np.random.rand(dim).astype(np.float32)
        items.append({K_VECTOR: vec, K_ID: f"id{i}", "keep": (i % 3 == 0)})
    db.upsert(items)

    q = np.random.rand(dim).astype(np.float32)
    res = db.query(q, top_k=10, where=lambda d: d.get("keep", False))
    assert all(int(r[K_ID][2:]) % 3 == 0 for r in res)


@pytest.mark.skipif(not _HAS_FAISS, reason="FAISS not available")
def test_ids_prefilter_limits_candidates_with_faiss(tmp_path):
    dim = 6
    base = str(tmp_path / "pf2")
    db = PicoVectorDB(embedding_dim=dim, storage_file=base)
    vs = np.eye(dim, dtype=np.float32)
    items = [{K_VECTOR: vs[i], K_ID: str(i)} for i in range(dim)]
    db.upsert(items)

    # Restrict to ids {"1","3","5"}
    q = (0.6 * vs[1] + 0.3 * vs[3] + 0.1 * vs[5]).astype(np.float32)
    res = db.query(q, top_k=3, ids=["1", "3", "5"]) 
    got = [r[K_ID] for r in res]
    assert set(got).issubset({"1", "3", "5"})
    # best should be "1"
    assert got[0] == "1"
