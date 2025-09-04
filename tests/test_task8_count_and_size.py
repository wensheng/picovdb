import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def test_count_and_size_with_deletions(tmp_path):
    db = PicoVectorDB(
        embedding_dim=3, storage_file=str(tmp_path / "cnt"), no_faiss=True
    )
    items = [
        {K_VECTOR: np.eye(3, dtype=np.float32)[i % 3], K_ID: f"id{i}"} for i in range(5)
    ]
    db.upsert(items)
    assert db.count() == 5
    assert len(db) == 5
    assert db.capacity() == 5

    # delete two
    db.delete(["id1", "id3"])
    assert db.count() == 3
    assert len(db) == 3
    assert db.capacity() == 5  # total slots unchanged
