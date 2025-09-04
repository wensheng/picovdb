import threading
import time
import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def test_writer_blocks_reads(tmp_path):
    db = PicoVectorDB(embedding_dim=4, storage_file=str(tmp_path / "rw"), no_faiss=True)
    db.upsert([{K_VECTOR: np.ones(4, dtype=np.float32), K_ID: "a"}])

    finished = []

    def reader_call():
        # Should block until writer releases
        _ = db.query(np.ones(4, dtype=np.float32), top_k=1)
        finished.append("read")

    # Hold write lock explicitly
    with db._rwlock.write_lock():
        t = threading.Thread(target=reader_call)
        t.start()
        time.sleep(0.05)
        # Reader should not have finished yet
        assert not finished
    t.join(timeout=1)
    assert finished == ["read"]


def test_readers_can_enter_concurrently(tmp_path):
    db = PicoVectorDB(
        embedding_dim=4, storage_file=str(tmp_path / "rw2"), no_faiss=True
    )
    db.upsert([{K_VECTOR: np.ones(4, dtype=np.float32), K_ID: "a"}])

    entered = []
    proceed = threading.Event()

    def reader_hold():
        with db._rwlock.read_lock():
            entered.append("r")
            proceed.wait(0.1)

    t1 = threading.Thread(target=reader_hold)
    t2 = threading.Thread(target=reader_hold)
    t1.start()
    t2.start()
    time.sleep(0.02)
    # Both readers should be in the critical section
    assert len(entered) >= 2
    proceed.set()
    t1.join()
    t2.join()
