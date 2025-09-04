import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def test_query_uses_only_active_rows(tmp_path):
    dim = 8
    db = PicoVectorDB(
        embedding_dim=dim, storage_file=str(tmp_path / "store2"), no_faiss=True
    )

    # Insert 30 random items
    items = [
        {K_VECTOR: np.random.rand(dim).astype(np.float32), K_ID: f"id{i}"}
        for i in range(30)
    ]
    db.upsert(items)

    # Delete 20 of them
    to_delete = [f"id{i}" for i in range(20)]
    db.delete(to_delete)

    # Active set should be the remaining 10 ids
    active_ids = set(db._id2idx.keys())
    assert len(active_ids) == 10

    # Query with a fresh random vector; request top_k larger than actives
    q = np.random.rand(dim).astype(np.float32)
    res = db.query(q, top_k=25)

    # Should return only active items, and up to the number of actives
    assert len(res) == len(active_ids)
    assert all(r[K_ID] in active_ids for r in res)

    # When no deletions, results count should be top_k (bounded by total)
    db2 = PicoVectorDB(
        embedding_dim=dim, storage_file=str(tmp_path / "store3"), no_faiss=True
    )
    db2.upsert(items)
    res2 = db2.query(q, top_k=15)
    assert len(res2) == 15  # 30 available, top_k 15
