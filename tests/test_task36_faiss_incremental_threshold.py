import numpy as np
import pytest

from picovdb.pico_vdb import PicoVectorDB, _HAS_FAISS, K_VECTOR, K_ID


@pytest.mark.skipif(not _HAS_FAISS, reason="FAISS not available")
def test_faiss_incremental_vs_full_threshold(tmp_path):
    base = str(tmp_path / "faiss_thr")
    # Use a moderate threshold for clear split
    db = PicoVectorDB(
        embedding_dim=8, storage_file=base, faiss_incremental_threshold_ratio=0.2
    )

    # Insert 40 vectors
    items = [
        {K_VECTOR: np.random.rand(8).astype(np.float32), K_ID: f"id{i}"}
        for i in range(40)
    ]
    db.upsert(items)

    # First query builds index (full)
    _ = db.query(np.random.rand(8).astype(np.float32), top_k=5)
    assert db._last_faiss_rebuild_mode == "full"

    # Small change: 1 update -> incremental (1/40 = 0.025 <= 0.2)
    db.upsert([{K_VECTOR: np.random.rand(8).astype(np.float32), K_ID: "id0"}])
    _ = db.query(np.random.rand(8).astype(np.float32), top_k=5)
    assert db._last_faiss_rebuild_mode == "incremental"

    # Large change: 12 updates -> full (12/40 = 0.3 > 0.2)
    batch = [
        {K_VECTOR: np.random.rand(8).astype(np.float32), K_ID: f"id{i}"}
        for i in range(12)
    ]
    db.upsert(batch)
    _ = db.query(np.random.rand(8).astype(np.float32), top_k=5)
    assert db._last_faiss_rebuild_mode == "full"

