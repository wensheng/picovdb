import numpy as np
import pytest

from picovdb.pico_vdb import PicoVectorDB, _normalize, K_VECTOR, K_ID, K_METRICS


def test_normalize_zero_is_unit_and_idempotent():
    z = np.zeros(5, dtype=np.float32)
    n1 = _normalize(z)
    assert n1.dtype == np.float32
    assert pytest.approx(np.linalg.norm(n1), rel=1e-6, abs=1e-6) == 1.0
    # idempotent
    n2 = _normalize(n1)
    assert np.allclose(n1, n2)


def test_upsert_zero_vector_becomes_unit(tmp_path):
    db = PicoVectorDB(embedding_dim=3, storage_file=str(tmp_path / "z"), no_faiss=True)
    db.upsert([{K_VECTOR: np.zeros(3, dtype=np.float32), K_ID: "z"}])
    # querying with the same zero vector should return itself with high score
    res = db.query(np.zeros(3, dtype=np.float32), top_k=1)
    assert res[0][K_ID] == "z"
    assert pytest.approx(res[0][K_METRICS], rel=1e-5) == 1.0


def test_query_zero_vector_is_unit_and_deterministic(tmp_path):
    # Build basis vectors e0,e1,e2
    e0 = np.array([1, 0, 0], dtype=np.float32)
    e1 = np.array([0, 1, 0], dtype=np.float32)
    e2 = np.array([0, 0, 1], dtype=np.float32)
    db = PicoVectorDB(embedding_dim=3, storage_file=str(tmp_path / "qz"), no_faiss=True)
    db.upsert(
        [
            {K_VECTOR: e0, K_ID: "0"},
            {K_VECTOR: e1, K_ID: "1"},
            {K_VECTOR: e2, K_ID: "2"},
        ]
    )
    # Zero query should map to e0 deterministically
    res = db.query(np.zeros(3, dtype=np.float32), top_k=3)
    assert res[0][K_ID] == "0"
