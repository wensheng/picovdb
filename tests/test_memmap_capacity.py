import os
import numpy as np
import pytest
from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID, Float


def test_memmap_preallocation(tmp_path):
    """Verify that creating a DB with capacity creates a file of the correct size."""
    db_path = str(tmp_path / "test_db")
    capacity = 100
    dim = 4
    db = PicoVectorDB(
        embedding_dim=dim, storage_file=db_path, use_memmap=True, capacity=capacity
    )

    vecs_path = f"{db_path}.vecs.npy"
    assert os.path.exists(vecs_path)
    expected_size = capacity * dim * np.dtype(Float).itemsize
    # Allow for a small header in the npy file
    assert os.path.getsize(vecs_path) >= expected_size

    assert db.capacity() == capacity
    assert db.count() == 0
    assert len(db._free) == capacity


def test_upsert_into_preallocated(tmp_path):
    """Verify that upserting into a pre-allocated DB works correctly."""
    db_path = str(tmp_path / "test_db")
    capacity = 10
    db = PicoVectorDB(
        embedding_dim=2, storage_file=db_path, use_memmap=True, capacity=capacity
    )

    items = [{K_ID: str(i), K_VECTOR: [float(i), float(i)]} for i in range(5)]
    db.upsert(items)

    assert db.count() == 5
    assert db.capacity() == capacity
    assert len(db._free) == capacity - 5

    retrieved = db.get("3", include_vector=True)
    assert retrieved is not None
    # The vector is normalized
    norm_vec = np.array([3.0, 3.0], dtype=Float)
    norm_vec /= np.linalg.norm(norm_vec)
    np.testing.assert_allclose(retrieved[K_VECTOR], norm_vec, rtol=1e-6)


def test_exceed_preallocated_capacity(tmp_path):
    """Verify that upserting beyond the pre-allocated capacity raises an error."""
    db_path = str(tmp_path / "test_db")
    capacity = 5
    db = PicoVectorDB(
        embedding_dim=2, storage_file=db_path, use_memmap=True, capacity=capacity
    )

    items = [{K_ID: str(i), K_VECTOR: [float(i), float(i)]} for i in range(capacity)]
    db.upsert(items)

    assert db.count() == capacity

    with pytest.raises(ValueError, match="Database capacity exceeded"):
        db.upsert([{K_ID: "extra", K_VECTOR: [1.0, 1.0]}])
