import os
import logging
import unittest.mock

import numpy as np
import pytest

from picovdb.pico_vdb import (
    PicoVectorDB,
    _normalize,
    _hash_vec,
    _ids_path,
    _meta_path,
    _vecs_path,
    K_VECTOR,
    K_ID,
)


def test_utils_path_helpers():
    base = "mydata/testbase"
    assert _ids_path(base) == "mydata/testbase.ids.json"
    assert _meta_path(base) == "mydata/testbase.meta.json"
    assert _vecs_path(base) == "mydata/testbase.vecs.npy"


def test_hash_and_normalize():
    # zero vector remains zero
    z = np.zeros(5, dtype=np.float32)
    norm_z = _normalize(z)
    assert norm_z.dtype == np.float32
    # assert np.all(norm_z == z)

    # nonzero vector normalized to unit length
    v = np.array([3.0, 4.0], dtype=np.float32)
    norm_v = _normalize(v)
    assert pytest.approx(np.linalg.norm(norm_v), rel=1e-6) == 1.0

    # hash is consistent
    h1 = _hash_vec(norm_v)
    h2 = _hash_vec(norm_v.copy())
    assert isinstance(h1, str)
    assert h1 == h2
    # different vector gives different hash
    norm_v2 = _normalize(np.array([4.0, 3.0], dtype=np.float32))
    assert _hash_vec(norm_v2) != h1


def make_simple_db(tmp_path, use_memmap=False):
    """Helper to create a fresh DB in temporary directory."""
    storage = str(tmp_path / "store")
    return PicoVectorDB(
        embedding_dim=3, storage_file=storage, use_memmap=use_memmap
    )


def test_upsert_and_len_and_get_by_id(tmp_path):
    db = make_simple_db(tmp_path)
    data = [
        {K_VECTOR: np.array([1.0, 0.0, 0.0], dtype=np.float32), K_ID: "a", "x": 1},
        {K_VECTOR: np.array([0.0, 1.0, 0.0], dtype=np.float32), K_ID: "b", "x": 2},
    ]
    report = db.upsert(data)
    # both inserted
    assert set(report["insert"]) == {"a", "b"}
    assert report["update"] == []
    assert len(db) == 2

    # get returns correct metadata
    rec = db.get("a")
    assert rec["x"] == 1 and rec[K_ID] == "a"
    rec = db.get("missing")
    assert rec is None

    # get returns only available ids
    results = db.get(["a", "missing", "b"])
    assert len(results) == 2
    assert {r["x"] for r in results} == {1, 2}


def test_get_all_and_store_additional_and_persistence(tmp_path):
    db = make_simple_db(tmp_path)
    items = [
        {K_VECTOR: np.random.rand(3).astype(np.float32), K_ID: "1"},
        {K_VECTOR: np.random.rand(3).astype(np.float32), K_ID: "2"},
    ]
    db.upsert(items)
    db.store_additional_data(foo="bar", num=123)
    db.save()

    # load into new instance
    db2 = PicoVectorDB(embedding_dim=3, storage_file=str(tmp_path / "store"))
    # additional data persisted
    assert db2.get_additional_data() == {"foo": "bar", "num": 123}

    all_docs = db2.get_all()
    ids = [d[K_ID] for d in all_docs]
    assert set(ids) == {"1", "2"}


def test_update_on_reupsert(tmp_path):
    db = make_simple_db(tmp_path)
    # insert one
    vec = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    db.upsert([{K_VECTOR: vec, K_ID: "u"}])
    # change metadata and vector
    new_vec = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    report = db.upsert([{K_VECTOR: new_vec, K_ID: "u", "tag": "updated"}])
    assert report["insert"] == []
    assert report["update"] == ["u"]
    rec = db.get("u")
    assert rec["tag"] == "updated"


def test_delete_and_reuse_free_slot(tmp_path):
    db = make_simple_db(tmp_path)
    # insert three
    for i in range(3):
        db.upsert([{K_VECTOR: np.eye(3, dtype=np.float32)[i], K_ID: str(i)}])
    assert len(db) == 3
    # delete one
    removed = db.delete(["1"])
    assert removed == ["1"]
    assert len(db) == 2
    # insert new, should reuse slot
    db.upsert([{K_VECTOR: np.array([1, 1, 1], dtype=np.float32), K_ID: "new"}])
    assert len(db) == 3
    # ensure "new" in get_all
    ids = [d[K_ID] for d in db.get_all()]
    assert "new" in ids


def test_query_single_and_batch(tmp_path):
    db = make_simple_db(tmp_path)
    # build orthonormal basis
    vs = [
        np.array([1, 0, 0], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
        np.array([0, 0, 1], dtype=np.float32),
    ]
    items = [{K_VECTOR: v, K_ID: str(i)} for i, v in enumerate(vs)]
    db.upsert(items)

    # single query
    q = np.array([0.9, 0.1, 0], dtype=np.float32)
    res = db.query(q, top_k=2)
    # best match is 0, then 1
    assert [r[K_ID] for r in res] == ["0", "1"]

    # batch query
    batch_q = np.stack([vs[2], vs[1]], axis=0)
    batch_res = db.query(batch_q, top_k=1)
    assert isinstance(batch_res, list) and len(batch_res) == 2
    assert batch_res[0][0][K_ID] == "2"
    assert batch_res[1][0][K_ID] == "1"


def test_query_with_filters(tmp_path):
    db = make_simple_db(tmp_path)
    # insert two
    items = [
        {K_VECTOR: np.array([1, 0, 0], dtype=np.float32), K_ID: "a", "keep": True},
        {K_VECTOR: np.array([1, 0, 0], dtype=np.float32), K_ID: "b", "keep": False},
    ]
    db.upsert(items)
    q = np.array([1, 0, 0], dtype=np.float32)
    # better_than filter that excludes low scores
    res = db.query(q, top_k=2, better_than=0.99)
    assert len(res) == 2

    # where filter
    res2 = db.query(q, top_k=2, where=lambda doc: doc.get("keep", False))
    assert [d[K_ID] for d in res2] == ["a"]


def test_persistence_with_memmap(tmp_path):
    # test using memmap storage
    db = make_simple_db(tmp_path, use_memmap=True)
    items = [{K_VECTOR: np.random.rand(3).astype(np.float32)} for _ in range(5)]
    report = db.upsert(items)
    assert len(report["insert"]) == 5
    db.save()

    # paths exist
    base = str(tmp_path / "store")
    assert os.path.exists(f"{base}.ids.json")
    assert os.path.exists(f"{base}.vecs.npy")

    # reload
    db2 = PicoVectorDB(embedding_dim=3, storage_file=base, use_memmap=True)
    assert len(db2) == 5
    # clean up memmap files
    os.remove(f"{base}.ids.json")
    # os.remove(f"{base}.vecs.npy")
    # meta file may not exist if no additional data
    if os.path.exists(f"{base}.meta.json"):
        os.remove(f"{base}.meta.json")


def test_memmap_append_warning(tmp_path, caplog):
    """Verify that appending to a memmapped DB logs a warning."""
    db_path = str(tmp_path / "test_db")
    db = PicoVectorDB(embedding_dim=2, storage_file=db_path, use_memmap=True)
    db.upsert([{K_ID: "a", K_VECTOR: [1.0, 2.0]}])
    db.save()

    # Re-load to ensure memmap is used
    db2 = PicoVectorDB(embedding_dim=2, storage_file=db_path, use_memmap=True)
    assert isinstance(db2._vectors, np.memmap)

    # With memmap, appending should trigger a warning
    with caplog.at_level(logging.WARNING, logger="picovdb"):
        caplog.clear()
        db2.upsert([{K_ID: "b", K_VECTOR: [3.0, 4.0]}])
        assert "Appending to a memmapped file" in caplog.text
        assert "converts it to an in-memory" in caplog.text

    # Check that it doesn't warn when memmap is not used
    db3_path = str(tmp_path / "test_db3")
    db3 = PicoVectorDB(embedding_dim=2, storage_file=db3_path, use_memmap=False)
    db3.upsert([{K_ID: "a", K_VECTOR: [1.0, 2.0]}])
    db3.save()
    db4 = PicoVectorDB(embedding_dim=2, storage_file=db3_path, use_memmap=False)

    with caplog.at_level(logging.WARNING, logger="picovdb"):
        caplog.clear()
        db4.upsert([{K_ID: "b", K_VECTOR: [3.0, 4.0]}])
        assert "Appending to a memmapped file" not in caplog.text

    # Check that it doesn't warn on update
    with caplog.at_level(logging.WARNING, logger="picovdb"):
        caplog.clear()
        db2.upsert([{K_ID: "a", K_VECTOR: [5.0, 6.0]}])
        assert "Appending to a memmapped file" not in caplog.text

def test_flush_with_memmap(tmp_path):
    """Verify that flush() persists changes for memmapped DBs."""
    db_path = str(tmp_path / "test_db")
    db = PicoVectorDB(embedding_dim=2, storage_file=db_path, use_memmap=True)
    
    # Initial upsert and save to create the memmap file
    db.upsert([{K_ID: "a", K_VECTOR: [1.0, 2.0]}])
    db.save()

    # Re-load to ensure memmap is used
    db2 = PicoVectorDB(embedding_dim=2, storage_file=db_path, use_memmap=True)
    assert isinstance(db2._vectors, np.memmap)
    
    # Update a vector, which should modify the memmap file in place
    db2.upsert([{K_ID: "a", K_VECTOR: [3.0, 4.0]}])
    
    # Load into a third instance - might not see the new data yet depending on OS caching
    # So we flush to be sure.
    db2.flush()
    
    # Load into a new instance to check if the flushed data is there
    db3 = PicoVectorDB(embedding_dim=2, storage_file=db_path, use_memmap=True)
    retrieved = db3.get("a", include_vector=True)
    assert retrieved is not None
    np.testing.assert_allclose(retrieved[K_VECTOR], np.array([0.6, 0.8], dtype=np.float32), rtol=1e-6)

    # Test no-op for non-memmap
    db_non_memmap_path = str(tmp_path / "test_db_non_memmap")
    db_non_memmap = PicoVectorDB(embedding_dim=2, storage_file=db_non_memmap_path, use_memmap=False)
    db_non_memmap.upsert([{K_ID: "a", K_VECTOR: [1.0, 2.0]}])
    db_non_memmap.flush() # Should not raise an error
    # Not saved, so a new instance should be empty
    db_non_memmap2 = PicoVectorDB(embedding_dim=2, storage_file=db_non_memmap_path, use_memmap=False)
    assert len(db_non_memmap2) == 0

def test_atomic_save(tmp_path):
    """Verify that save() is atomic and doesn't corrupt files on failure."""
    db_path = str(tmp_path / "test_db")
    db = PicoVectorDB(embedding_dim=2, storage_file=db_path)
    db.upsert([{K_ID: "a", K_VECTOR: [1.0, 2.0]}])
    db.save()

    # Now, make another change
    db.upsert([{K_ID: "b", K_VECTOR: [3.0, 4.0]}])
    
    # Simulate a failure during save (e.g., disk full)
    with unittest.mock.patch("os.replace", side_effect=OSError("Disk full")):
        with pytest.raises(OSError):
            db.save()

    # Check that temporary files are cleaned up
    assert not any(str(f).endswith(".tmp") for f in tmp_path.iterdir())

    # Load the database again and check that it has the old data, not the corrupted new data
    db2 = PicoVectorDB(embedding_dim=2, storage_file=db_path)
    assert len(db2) == 1
    assert db2.get("a") is not None
    assert db2.get("b") is None
