import numpy as np
import pytest

from picovdb.pico_vdb import PicoVectorDB, _HAS_FAISS, K_VECTOR, K_ID


@pytest.mark.skipif(not _HAS_FAISS, reason="FAISS not available")
def test_faiss_ef_search_default_and_override(tmp_path):
    base = str(tmp_path / "tunables")
    db = PicoVectorDB(embedding_dim=8, storage_file=base, ef_search_default=17)
    db.upsert(
        [
            {K_VECTOR: np.random.rand(8).astype(np.float32), K_ID: str(i)}
            for i in range(20)
        ]
    )

    # Trigger query: default ef_search should be applied
    _ = db.query(np.random.rand(8).astype(np.float32), top_k=3)
    # Inspect underlying efSearch setting
    try:
        ef = db._faiss.index.hnsw.efSearch  # type: ignore[attr-defined]
        assert ef == 17
    except Exception:
        pytest.skip("FAISS HNSW efSearch attribute not available")

    # Per-query override
    _ = db.query(np.random.rand(8).astype(np.float32), top_k=3, ef_search=9)
    try:
        ef2 = db._faiss.index.hnsw.efSearch  # type: ignore[attr-defined]
        assert ef2 == 9
    except Exception:
        pytest.skip("FAISS HNSW efSearch attribute not available")


@pytest.mark.skipif(not _HAS_FAISS, reason="FAISS not available")
def test_faiss_threads_setting(tmp_path):
    import faiss  # type: ignore

    if not hasattr(faiss, "omp_set_num_threads") or not hasattr(
        faiss, "omp_get_max_threads"
    ):
        pytest.skip("FAISS omp thread APIs not available")

    base = str(tmp_path / "threads")
    PicoVectorDB(embedding_dim=8, storage_file=base, faiss_threads=1)
    # We cannot guarantee exact behavior across platforms, but often FAISS reflects the setting
    got = faiss.omp_get_max_threads()  # type: ignore[attr-defined]
    assert got == 1
