import numpy as np
import pytest

from picovdb.pico_vdb import PicoVectorDB, _HAS_FAISS, K_VECTOR, K_ID


@pytest.mark.skipif(not _HAS_FAISS, reason="FAISS not available")
def test_faiss_uses_idmap_and_active_only(tmp_path):
    base = str(tmp_path / "faiss_idmap")
    db = PicoVectorDB(embedding_dim=8, storage_file=base)

    # Insert 10, delete 5, then rebuild lazily via query
    items = [{K_VECTOR: np.random.rand(8).astype(np.float32), K_ID: f"id{i}"} for i in range(10)]
    db.upsert(items)
    db.delete([f"id{i}" for i in range(5)])
    # Trigger rebuild on first query
    _ = db.query(np.random.rand(8).astype(np.float32), top_k=10)

    # Verify FAISS is an IndexIDMap2 and ntotal equals active count
    import faiss  # type: ignore

    assert isinstance(db._faiss, faiss.IndexIDMap2)
    assert db._faiss.ntotal == db.count()

    # FAISS search should never return deleted IDs (global row indices map to active docs)
    D, I = db._faiss.search(np.random.rand(1, 8).astype(np.float32), 10)
    for idx in I[0]:
        if idx == -1:
            continue
        # Each returned idx is a global row index; corresponding doc must be active
        assert db._docs[idx] is not None

