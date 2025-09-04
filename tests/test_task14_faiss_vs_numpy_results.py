import numpy as np
import pytest

from picovdb.pico_vdb import PicoVectorDB, _HAS_FAISS, K_VECTOR, K_ID


@pytest.mark.skipif(not _HAS_FAISS, reason="FAISS not available")
def test_faiss_and_numpy_paths_match_on_simple_data(tmp_path):
    dim = 6
    base = str(tmp_path / "cmp")
    # Build simple orthonormal-like set for deterministic nearest neighbors
    vs = np.eye(dim, dtype=np.float32)
    items = [{K_VECTOR: vs[i], K_ID: str(i)} for i in range(dim)]

    db_faiss = PicoVectorDB(embedding_dim=dim, storage_file=base)
    db_faiss.upsert(items)
    # Trigger FAISS build lazily
    _ = db_faiss.query(vs[0], top_k=dim)

    db_numpy = PicoVectorDB(
        embedding_dim=dim, storage_file=str(tmp_path / "cmp_np"), no_faiss=True
    )
    db_numpy.upsert(items)

    # Query 1: exact basis; compare top-1 only to avoid tie-ordering differences
    rf1 = db_faiss.query(vs[0], top_k=1)
    rn1 = db_numpy.query(vs[0], top_k=1)
    assert [r[K_ID] for r in rf1] == [r[K_ID] for r in rn1]

    # Query 2: weighted combo with distinct components for deterministic ordering
    q2 = np.array([0.9, 0.08, 0.02, 0, 0, 0], dtype=np.float32)
    rf2 = db_faiss.query(q2, top_k=3)
    rn2 = db_numpy.query(q2, top_k=3)
    assert [r[K_ID] for r in rf2] == [r[K_ID] for r in rn2]
