import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def test_get_all_defaults_exclude_deleted(tmp_path):
    db = PicoVectorDB(embedding_dim=3, storage_file=str(tmp_path / "all"), no_faiss=True)
    items = [
        {K_VECTOR: np.eye(3, dtype=np.float32)[i], K_ID: str(i)}
        for i in range(3)
    ]
    db.upsert(items)
    db.delete(["1"])  # delete middle

    # Default excludes deleted
    res = db.get_all()
    ids = [r[K_ID] for r in res]
    assert set(ids) == {"0", "2"}
    assert "1" not in ids

    # Include deleted shows placeholder for "1"
    res_all = db.get_all(include_deleted=True)
    ids_all = [r[K_ID] for r in res_all]
    assert set(ids_all) == {"0", "1", "2"}
    # Placeholder has only _id_
    placeholder = next(r for r in res_all if r[K_ID] == "1")
    assert set(placeholder.keys()) == {K_ID}

    # Include vectors only for active when requested
    res_vecs = db.get_all(include_vector=True)
    for r in res_vecs:
        assert K_VECTOR in r and r[K_ID] in {"0", "2"}

