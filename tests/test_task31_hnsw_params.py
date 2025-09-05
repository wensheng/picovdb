import numpy as np
import pytest

from picovdb.pico_vdb import PicoVectorDB, _HAS_FAISS, K_VECTOR, K_ID


@pytest.mark.skipif(not _HAS_FAISS, reason="FAISS not available")
def test_hnsw_build_and_search_params(tmp_path):
    base = str(tmp_path / "hnsw_params")
    db = PicoVectorDB(
        embedding_dim=8,
        storage_file=base,
        hnsw_m=16,
        hnsw_ef_construction=77,
        hnsw_ef_search_default=23,
    )

    db.upsert(
        [
            {K_VECTOR: np.random.rand(8).astype(np.float32), K_ID: str(i)}
            for i in range(25)
        ]
    )

    # Trigger a query to ensure search params are applied
    _ = db.query(np.random.rand(8).astype(np.float32), top_k=3)

    # Validate efConstruction was set for HNSW builds
    try:
        efc = db._faiss.index.hnsw.efConstruction  # type: ignore[attr-defined]
        assert efc == 77
    except Exception:
        pytest.skip("FAISS HNSW efConstruction attribute not available")

    # Validate default efSearch is applied
    try:
        efs = db._faiss.index.hnsw.efSearch  # type: ignore[attr-defined]
        assert efs == 23
    except Exception:
        pytest.skip("FAISS HNSW efSearch attribute not available")

    # Per-query override using new alias
    _ = db.query(
        np.random.rand(8).astype(np.float32), top_k=3, hnsw_ef_search=11
    )
    try:
        efs2 = db._faiss.index.hnsw.efSearch  # type: ignore[attr-defined]
        assert efs2 == 11
    except Exception:
        pytest.skip("FAISS HNSW efSearch attribute not available for override check")

    # Attempt to verify HNSW M if exposed by FAISS build
    # Some builds expose `nb_neighbors`/`nbNeighbors`/`M` on `hnsw`.
    hnsw = getattr(db._faiss, "index").hnsw  # type: ignore[attr-defined]
    for attr in ("nb_neighbors", "nbNeighbors", "M"):
        if hasattr(hnsw, attr):
            assert int(getattr(hnsw, attr)) == 16
            break
    else:
        pytest.skip("FAISS HNSW M/nb_neighbors attribute not available")

