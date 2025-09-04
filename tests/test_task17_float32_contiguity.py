import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def is_c_f32(arr):
    return arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]


def test_vectors_are_c_contiguous_float32_after_ops_and_reload(tmp_path):
    dim = 10
    base = str(tmp_path / "c32")
    db = PicoVectorDB(embedding_dim=dim, storage_file=base, no_faiss=True)

    # Non-contiguous input vector (slice)
    x = np.arange(2 * dim, dtype=np.float32)[1::2]  # still contiguous, but fine
    db.upsert([{K_VECTOR: x, K_ID: "a"}])
    db.upsert([{K_VECTOR: np.arange(dim, dtype=np.float64), K_ID: "b"}])  # wrong dtype

    assert is_c_f32(db._vectors)

    # delete one, ensure contiguity retained
    db.delete(["a"])  # in-place modifications should not change contiguity
    assert is_c_f32(db._vectors)

    # save and reload (numpy load path)
    db.save()
    db2 = PicoVectorDB(embedding_dim=dim, storage_file=base, no_faiss=True)
    assert is_c_f32(db2._vectors)


def test_memmap_load_is_c_f32(tmp_path):
    dim = 6
    base = str(tmp_path / "mm")
    db = PicoVectorDB(
        embedding_dim=dim, storage_file=base, use_memmap=True, no_faiss=True
    )
    items = [
        {K_VECTOR: np.random.rand(dim).astype(np.float32), K_ID: str(i)}
        for i in range(4)
    ]
    db.upsert(items)
    db.save()

    db2 = PicoVectorDB(
        embedding_dim=dim, storage_file=base, use_memmap=True, no_faiss=True
    )
    arr = db2._vectors
    # memmap should also be C-contiguous float32
    assert arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]


def test_query_accepts_noncontiguous_inputs(tmp_path):
    dim = 8
    db = PicoVectorDB(
        embedding_dim=dim, storage_file=str(tmp_path / "qnc"), no_faiss=True
    )
    db.upsert([{K_VECTOR: np.random.rand(dim).astype(np.float32), K_ID: "x"}])

    # Fortran-ordered 2D input
    qs = np.asfortranarray(np.random.rand(3, dim).astype(np.float32))
    res = db.query(qs, top_k=1)
    assert isinstance(res, list) and len(res) == 3
