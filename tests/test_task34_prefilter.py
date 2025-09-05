import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def make_db(tmp_path):
    db = PicoVectorDB(embedding_dim=3, storage_file=str(tmp_path / "w34"), no_faiss=True)
    items = [
        {K_ID: "a", K_VECTOR: np.array([1, 0, 0], dtype=np.float32), "keep": True, "cat": "x"},
        {K_ID: "b", K_VECTOR: np.array([1, 0, 0], dtype=np.float32), "keep": False, "cat": "y"},
        {K_ID: "c", K_VECTOR: np.array([1, 0, 0], dtype=np.float32), "keep": True, "cat": "z"},
    ]
    db.upsert(items)
    return db


def test_where_eq_dict_matches_lambda(tmp_path):
    db = make_db(tmp_path)
    q = np.array([1, 0, 0], dtype=np.float32)
    # dict equality
    r_dict = db.query(q, top_k=10, where={"keep": True})
    # equivalent lambda
    r_lambda = db.query(q, top_k=10, where=lambda d: d.get("keep", False))
    assert [r[K_ID] for r in r_dict] == [r[K_ID] for r in r_lambda]


def test_where_in_dict_matches_lambda(tmp_path):
    db = make_db(tmp_path)
    q = np.array([1, 0, 0], dtype=np.float32)
    r_dict = db.query(q, top_k=10, where={"cat": {"$in": ["x", "z"]}})
    r_lambda = db.query(q, top_k=10, where=lambda d: d.get("cat") in {"x", "z"})
    assert [r[K_ID] for r in r_dict] == [r[K_ID] for r in r_lambda]


def test_where_dict_with_ids_intersection(tmp_path):
    db = make_db(tmp_path)
    q = np.array([1, 0, 0], dtype=np.float32)
    # ids subset includes b and c; filter should keep only c
    r = db.query(q, top_k=10, where={"keep": True}, ids=["b", "c"])
    assert [x[K_ID] for x in r] == ["c"]

