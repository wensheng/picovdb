import logging
import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR


def test_default_logging_quiet(caplog, tmp_path):
    # Ensure INFO logs are not emitted by default
    caplog.set_level(logging.WARNING, logger="picovdb")

    db = PicoVectorDB(
        embedding_dim=4, storage_file=str(tmp_path / "log"), no_faiss=True
    )
    db.upsert([{K_VECTOR: np.ones(4, dtype=np.float32)}])
    db.save()

    # No INFO messages should be captured
    assert all(r.levelno >= logging.WARNING for r in caplog.records)
    assert not any(
        "Loaded existing DB" in r.message or "Saved" in r.message
        for r in caplog.records
    )


def test_info_logging_when_enabled(caplog, tmp_path):
    # Enable INFO for our logger to see messages
    caplog.set_level(logging.INFO, logger="picovdb")

    base = str(tmp_path / "log2")
    db = PicoVectorDB(embedding_dim=4, storage_file=base, no_faiss=True)
    db.upsert([{K_VECTOR: np.ones(4, dtype=np.float32)}])
    db.save()

    # Reload to trigger load message as well
    _ = PicoVectorDB(embedding_dim=4, storage_file=base, no_faiss=True)

    messages = [r.message for r in caplog.records]
    assert any("Saved" in m for m in messages)
    assert any("Loading existing DB" in m for m in messages)
