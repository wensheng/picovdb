import numpy as np
import pytest

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def test_get_single_and_multiple_ids(tmp_path):
    base = str(tmp_path / "get32")
    db = PicoVectorDB(embedding_dim=4, storage_file=base, no_faiss=True)

    items = [
        {K_ID: "a", K_VECTOR: np.array([1, 0, 0, 0], dtype=np.float32)},
        {K_ID: "b", K_VECTOR: np.array([0, 1, 0, 0], dtype=np.float32)},
    ]
    db.upsert(items)

    # Single ID -> single dict or None
    rec_a = db.get("a")
    assert isinstance(rec_a, dict)
    assert rec_a[K_ID] == "a"

    # Missing single id -> None
    rec_c = db.get("c")
    assert rec_c is None

    # List of IDs -> list of found records (skips missing)
    got = db.get(["a", "c", "b"])  # type: ignore[arg-type]
    assert [r[K_ID] for r in got] == ["a", "b"]

    # include_vector works for both
    rec_bv = db.get("b", include_vector=True)
    assert np.allclose(rec_bv["_vector_"], np.array([0, 1, 0, 0], dtype=np.float32))


def test_get_by_id_deprecated(tmp_path):
    base = str(tmp_path / "get32dep")
    db = PicoVectorDB(embedding_dim=3, storage_file=base, no_faiss=True)
    db.upsert([{K_ID: "x", K_VECTOR: np.array([1, 0, 0], dtype=np.float32)}])

    with pytest.warns(DeprecationWarning):
        rec = db.get_by_id("x")
        assert rec and rec[K_ID] == "x"

