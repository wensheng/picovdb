import numpy as np
import pytest

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def make_db_with_deleted(tmp_path):
    """Helper to create a DB with some deleted items."""
    db = PicoVectorDB(embedding_dim=2, storage_file=str(tmp_path / "test_db"))
    db.upsert(
        [
            {K_ID: "a", K_VECTOR: [1.0, 0.0]},
            {K_ID: "b", K_VECTOR: [0.0, 1.0]},
            {K_ID: "c", K_VECTOR: [0.5, 0.5]},
        ]
    )
    db.delete(["b"])
    return db


def test_query_one(tmp_path):
    """Test the query_one convenience method."""
    db = make_db_with_deleted(tmp_path)
    query_vec = np.array([0.8, 0.2], dtype=np.float32)
    results = db.query_one(query_vec, top_k=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0][K_ID] == "a"

    # Test that it returns the same as query with a single vector
    query_results_batch = db.query(query_vec, top_k=1)
    assert results == query_results_batch


def test_stats(tmp_path):
    """Test the stats method."""
    db = make_db_with_deleted(tmp_path)
    db.save()
    stats = db.stats()

    assert stats["active"] == 2
    assert stats["deleted"] == 1
    assert stats["total"] == 3
    assert stats["dim"] == 2
    assert isinstance(stats["faiss"], bool)
    assert not stats["memmap"]
    assert "test_db.ids.json" in stats["file_sizes"]
    assert "test_db.vecs.npy" in stats["file_sizes"]
    assert "test_db.meta.json" in stats["file_sizes"]


def test_vacuum(tmp_path):
    """Test the vacuum method."""
    db = make_db_with_deleted(tmp_path)
    assert db.count() == 2
    assert db.capacity() == 3

    db.vacuum()

    assert db.count() == 2
    assert db.capacity() == 2
    assert len(db._free) == 0

    # Verify that queries still work correctly
    query_vec = np.array([0.4, 0.6], dtype=np.float32)
    results = db.query_one(query_vec, top_k=1)
    assert results[0][K_ID] == "c"

    # Verify that the data is still correct
    item_a = db.get("a", include_vector=True)
    item_c = db.get("c", include_vector=True)
    assert item_a is not None
    assert item_c is not None
    np.testing.assert_allclose(item_a[K_VECTOR], np.array([1.0, 0.0]), rtol=1e-6)
    
    # Item b should be gone
    assert db.get("b") is None
