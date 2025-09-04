import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def baseline_topk(matrix, q, top_k, ids):
    scores = matrix @ q
    order = np.argsort(-scores)[:top_k]
    return [ids[i] for i in order]


def test_argsort_argpartition_equivalence_small_and_large_k(tmp_path):
    dim = 16
    N = 200
    base = str(tmp_path / "ap")
    db = PicoVectorDB(embedding_dim=dim, storage_file=base, no_faiss=True)

    rng = np.random.default_rng(0)
    vecs = rng.random((N, dim), dtype=np.float32)
    items = [{K_VECTOR: vecs[i], K_ID: str(i)} for i in range(N)]
    db.upsert(items)

    q = rng.random(dim, dtype=np.float32)

    # small k path (argpartition)
    res_small = db.query(q, top_k=5)
    ids_small = [r[K_ID] for r in res_small]
    ids_small_base = baseline_topk(db._vectors, q, 5, [str(i) for i in range(N)])
    assert ids_small == ids_small_base

    # large k path (>20% of N, forces full argsort)
    k_large = int(0.3 * N)
    res_large = db.query(q, top_k=k_large)
    ids_large = [r[K_ID] for r in res_large]
    ids_large_base = baseline_topk(db._vectors, q, k_large, [str(i) for i in range(N)])
    assert ids_large == ids_large_base

