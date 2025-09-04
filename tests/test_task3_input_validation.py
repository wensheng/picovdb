import numpy as np
import pytest

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def test_upsert_rejects_wrong_shape_and_dim(tmp_path):
    db = PicoVectorDB(embedding_dim=4, storage_file=str(tmp_path / "v"), no_faiss=True)

    # Non-1D vector
    with pytest.raises(ValueError) as e:
        db.upsert([{K_VECTOR: np.ones((1, 4), dtype=np.float32), K_ID: "a"}])
    assert "1D" in str(e.value)

    # Wrong length
    with pytest.raises(ValueError) as e:
        db.upsert([{K_VECTOR: np.ones(3, dtype=np.float32), K_ID: "b"}])
    assert "expected 4" in str(e.value) and "got 3" in str(e.value)


def test_query_validates_shapes_and_dims(tmp_path):
    db = PicoVectorDB(embedding_dim=4, storage_file=str(tmp_path / "q"), no_faiss=True)
    # Insert a valid record
    db.upsert([{K_VECTOR: np.array([1, 0, 0, 0], dtype=np.float32), K_ID: "x"}])

    # 1D query ok
    res1 = db.query(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
    assert isinstance(res1, list)

    # 2D query ok
    res2 = db.query(
        np.stack([np.array([1, 0, 0, 0], dtype=np.float32)], axis=0), top_k=1
    )
    assert isinstance(res2, list) and isinstance(res2[0], list)

    # Wrong last dim
    with pytest.raises(ValueError) as e:
        db.query(np.ones(5, dtype=np.float32))
    assert "expected" in str(e.value)

    with pytest.raises(ValueError) as e:
        db.query(np.ones((2, 5), dtype=np.float32))
    assert "expected last dim 4" in str(e.value)

    # 3D array invalid
    with pytest.raises(ValueError) as e:
        db.query(np.ones((1, 1, 4), dtype=np.float32))
    assert "1D or 2D" in str(e.value)
