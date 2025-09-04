import numpy as np
import pytest

from picovdb.pico_vdb import PicoVectorDB, _HAS_FAISS, K_VECTOR, K_ID


@pytest.mark.skipif(not _HAS_FAISS, reason="FAISS not available")
def test_lazy_rebuild_on_query(tmp_path, monkeypatch):
    base = str(tmp_path / "faiss_lazy")
    db = PicoVectorDB(embedding_dim=8, storage_file=base)

    # spy on rebuild calls
    calls = {"n": 0}
    orig = PicoVectorDB._rebuild_faiss

    def wrapped(self):
        calls["n"] += 1
        return orig(self)

    monkeypatch.setattr(PicoVectorDB, "_rebuild_faiss", wrapped)

    # Upsert does not trigger rebuild
    items = [
        {K_VECTOR: np.random.rand(8).astype(np.float32), K_ID: f"id{i}"}
        for i in range(20)
    ]
    db.upsert(items)
    assert calls["n"] == 0

    # First query triggers exactly one rebuild
    _ = db.query(np.random.rand(8).astype(np.float32), top_k=5)
    assert calls["n"] == 1

    # Subsequent query does not rebuild again
    _ = db.query(np.random.rand(8).astype(np.float32), top_k=5)
    assert calls["n"] == 1

    # Delete marks dirty; next query triggers exactly one rebuild
    db.delete(["id0", "id1"])  # mark dirty
    assert calls["n"] == 1
    _ = db.query(np.random.rand(8).astype(np.float32), top_k=5)
    assert calls["n"] == 2

    # Explicit rebuild_index performs rebuild immediately
    db.rebuild_index()
    assert calls["n"] == 3

    # Upsert marks dirty; save should rebuild once even without querying
    db.upsert([{K_VECTOR: np.random.rand(8).astype(np.float32), K_ID: "extra"}])
    db.save()
    assert calls["n"] == 4
