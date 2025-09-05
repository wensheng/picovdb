import os
import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def make_db(tmp_path, n=100):
    base = str(tmp_path / "knobs48")
    db = PicoVectorDB(embedding_dim=4, storage_file=base, no_faiss=True)
    items = [
        {K_ID: str(i), K_VECTOR: np.random.rand(4).astype(np.float32)} for i in range(n)
    ]
    db.upsert(items)
    return db


def test_env_var_tuning(tmp_path, monkeypatch):
    monkeypatch.setenv("PICOVDB_ADAPTIVE_BUFFER", "7")
    monkeypatch.setenv("PICOVDB_ARGSORT_THRESHOLD", "0.9")
    db = PicoVectorDB(embedding_dim=4, storage_file=str(tmp_path / "env"), no_faiss=True)
    assert db._adaptive_buffer == 7
    assert abs(db._argsort_threshold - 0.9) < 1e-9


def test_kwargs_override_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PICOVDB_ADAPTIVE_BUFFER", "3")
    monkeypatch.setenv("PICOVDB_ARGSORT_THRESHOLD", "0.9")
    db = PicoVectorDB(
        embedding_dim=4,
        storage_file=str(tmp_path / "kwargs"),
        no_faiss=True,
        adaptive_buffer=11,
        argsort_threshold=0.1,
    )
    assert db._adaptive_buffer == 11
    assert abs(db._argsort_threshold - 0.1) < 1e-9


def test_strategy_selection_knob(tmp_path):
    db = make_db(tmp_path, n=100)
    q = np.random.rand(4).astype(np.float32)

    # Force argsort by making threshold very small (frac ~ 0.1 > 0.0)
    db._argsort_threshold = 0.0
    db._adaptive_buffer = 0
    _ = db.query(q, top_k=10)
    assert db._last_topk_strategy == "argsort"

    # Force argpartition by making threshold very large (frac ~ 0.1 <= 1.0)
    db._argsort_threshold = 1.0
    _ = db.query(q, top_k=10)
    assert db._last_topk_strategy == "argpartition"


def test_adaptive_buffer_affects_k_eff(tmp_path):
    db = make_db(tmp_path, n=50)
    q = np.random.rand(4).astype(np.float32)
    db._adaptive_buffer = 7
    _ = db.query(q, top_k=5, where=lambda d: True)  # ensure buffer applied
    assert db._last_k_eff == 12  # 5 + 7

