import numpy as np

from picovdb.pico_vdb import PicoVectorDB, K_VECTOR, K_ID


def sorted_list(arr_like):
    return sorted(list(arr_like))


def test_active_indices_tracks_mutations(tmp_path):
    db = PicoVectorDB(embedding_dim=4, storage_file=str(tmp_path / "store"))

    # insert 5 items
    items = [
        {K_VECTOR: np.random.rand(4).astype(np.float32), K_ID: f"id{i}"}
        for i in range(5)
    ]
    db.upsert(items)

    # expect active indices 0..4
    expect_positions = [i for i, doc in enumerate(db._docs) if doc is not None]
    expect_from_map = list(db._id2idx.values())
    assert sorted_list(db._active_indices) == sorted_list(expect_positions)
    assert sorted_list(db._active_indices) == sorted_list(expect_from_map)

    # delete two and verify
    db.delete(["id1", "id3"])
    expect_positions = [i for i, doc in enumerate(db._docs) if doc is not None]
    expect_from_map = list(db._id2idx.values())
    assert sorted_list(db._active_indices) == sorted_list(expect_positions)
    assert sorted_list(db._active_indices) == sorted_list(expect_from_map)

    # insert new one; should reuse a free slot (either 1 or 3)
    db.upsert([{K_VECTOR: np.random.rand(4).astype(np.float32), K_ID: "new"}])
    expect_positions = [i for i, doc in enumerate(db._docs) if doc is not None]
    expect_from_map = list(db._id2idx.values())
    assert sorted_list(db._active_indices) == sorted_list(expect_positions)
    assert sorted_list(db._active_indices) == sorted_list(expect_from_map)

    # save and reload; active indices rebuilt correctly
    db.save()
    db2 = PicoVectorDB(embedding_dim=4, storage_file=str(tmp_path / "store"))
    expect_positions2 = [i for i, doc in enumerate(db2._docs) if doc is not None]
    expect_from_map2 = list(db2._id2idx.values())
    assert sorted_list(db2._active_indices) == sorted_list(expect_positions2)
    assert sorted_list(db2._active_indices) == sorted_list(expect_from_map2)
