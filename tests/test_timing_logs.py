import logging
import numpy as np
import pytest
from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def test_timing_logs(tmp_path, caplog):
    """Verify that DEBUG-level timing logs are emitted."""
    db_path = str(tmp_path / "test_db")
    
    # Test load timing
    caplog.set_level(logging.DEBUG, logger="picovdb")
    db = PicoVectorDB(embedding_dim=2, storage_file=db_path)
    assert "load took" in caplog.text
    caplog.clear()

    # Test save timing
    db.upsert([{K_ID: "a", K_VECTOR: [1.0, 2.0]}])
    db.save()
    assert "save took" in caplog.text
    caplog.clear()

    # Test query timing
    db.query(np.array([1.0, 2.0]))
    assert "query took" in caplog.text
    caplog.clear()

    # Test rebuild_faiss timing (if faiss is available)
    if db._faiss:
        db.rebuild_index()
        assert "rebuild_faiss took" in caplog.text
        caplog.clear()

    # Verify no logs at INFO level
    caplog.set_level(logging.INFO, logger="picovdb")
    db.save()
    db.query(np.array([1.0, 2.0]))
    if db._faiss:
        db.rebuild_index()
    assert "took" not in caplog.text
