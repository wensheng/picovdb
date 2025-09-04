import os
import shutil
import numpy as np
import pytest

from picovdb.pico_vdb import PicoVectorDB, _HAS_FAISS, K_VECTOR, K_ID


@pytest.mark.skipif(not _HAS_FAISS, reason="FAISS not available")
def test_missing_faiss_file_triggers_rebuild(tmp_path):
    base = str(tmp_path / "safe1")
    db = PicoVectorDB(embedding_dim=8, storage_file=base)
    db.upsert(
        [
            {K_VECTOR: np.random.rand(8).astype(np.float32), K_ID: str(i)}
            for i in range(10)
        ]
    )
    db.save()

    faiss_path = f"{base}.vecs.npy.faiss"
    assert os.path.exists(faiss_path)
    os.remove(faiss_path)

    # Reload should rebuild without error
    db2 = PicoVectorDB(embedding_dim=8, storage_file=base)
    assert db2.count() == 10
    # FAISS present and consistent
    assert db2._faiss is not None and db2._faiss.ntotal == 10


@pytest.mark.skipif(not _HAS_FAISS, reason="FAISS not available")
def test_dim_mismatch_faiss_file_triggers_rebuild(tmp_path):
    base8 = str(tmp_path / "safe2_8")
    base16 = str(tmp_path / "safe2_16")
    db8 = PicoVectorDB(embedding_dim=8, storage_file=base8)
    db8.upsert(
        [
            {K_VECTOR: np.random.rand(8).astype(np.float32), K_ID: str(i)}
            for i in range(5)
        ]
    )
    db8.save()

    db16 = PicoVectorDB(embedding_dim=16, storage_file=base16)
    db16.upsert(
        [
            {K_VECTOR: np.random.rand(16).astype(np.float32), K_ID: str(i)}
            for i in range(7)
        ]
    )
    db16.save()

    # Overwrite 8-dim store's FAISS file with 16-dim one
    src = f"{base16}.vecs.npy.faiss"
    dst = f"{base8}.vecs.npy.faiss"
    shutil.copyfile(src, dst)

    # Reload 8-dim store should detect mismatch and rebuild
    db8b = PicoVectorDB(embedding_dim=8, storage_file=base8)
    assert db8b._faiss is not None
    # index ntotal equals active count
    assert db8b._faiss.ntotal == db8b.count() == 5
