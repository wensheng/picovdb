import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def test_adaptive_buffer_fills_topk_with_where(tmp_path):
    dim = 8
    db = PicoVectorDB(embedding_dim=dim, storage_file=str(tmp_path / "ab"), no_faiss=True)
    # Insert 100 items with a flag; about 50 should match
    items = []
    for i in range(100):
        items.append({K_VECTOR: np.random.rand(dim).astype(np.float32), K_ID: str(i), "keep": (i % 2 == 0)})
    db.upsert(items)

    q = np.random.rand(dim).astype(np.float32)
    res = db.query(q, top_k=20, where=lambda d: d.get("keep", False))
    # Should fill top_k since there are many matching candidates
    assert len(res) == 20
    assert all(int(r[K_ID]) % 2 == 0 for r in res)

