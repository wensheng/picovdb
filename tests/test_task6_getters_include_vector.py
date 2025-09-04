import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def test_get_include_vector_flags(tmp_path):
    dim = 4
    db = PicoVectorDB(embedding_dim=dim, storage_file=str(tmp_path / "g"), no_faiss=True)
    v = np.array([1, 0, 0, 0], dtype=np.float32)
    db.upsert([{K_VECTOR: v, K_ID: "a", "meta": 1}])

    res_meta = db.get(["a"])  # default exclude vectors
    assert K_VECTOR not in res_meta[0]

    res_full = db.get(["a"], include_vector=True)
    assert K_VECTOR in res_full[0]
    assert res_full[0][K_VECTOR].shape == (dim,)
    # normalized vector equals e0
    assert np.allclose(res_full[0][K_VECTOR], v)


def test_get_by_id_and_get_all_include_vector(tmp_path):
    dim = 3
    db = PicoVectorDB(embedding_dim=dim, storage_file=str(tmp_path / "ga"), no_faiss=True)
    e0 = np.array([1, 0, 0], dtype=np.float32)
    e1 = np.array([0, 1, 0], dtype=np.float32)
    db.upsert([{K_VECTOR: e0, K_ID: "0"}, {K_VECTOR: e1, K_ID: "1"}])

    r0 = db.get_by_id("0")
    assert r0 is not None and K_VECTOR not in r0
    r0v = db.get_by_id("0", include_vector=True)
    assert K_VECTOR in r0v and np.allclose(r0v[K_VECTOR], e0)

    all_meta = db.get_all()
    assert all(K_VECTOR not in r for r in all_meta)

    all_full = db.get_all(include_vector=True)
    id_map = {r[K_ID]: r for r in all_full}
    assert np.allclose(id_map["0"][K_VECTOR], e0)
    assert np.allclose(id_map["1"][K_VECTOR], e1)

