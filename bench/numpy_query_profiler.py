import time
import numpy as np
import os
import argparse
import set_path
from typing import Callable, Optional
from picovdb import PicoVectorDB, K_ID, K_VECTOR

# Defaults (overridable by CLI)
DIMENSION = 1024
NUM_QUERIES = 50
TOP_K = 10


def get_db_paths(base_path: str):
    return [
        f"{base_path}.ids.json",
        f"{base_path}.meta.json",
        f"{base_path}.vecs.npy",
        f"{base_path}.vecs.npy.faiss",
    ]


def cleanup_db_files(base_path: str, verbose: bool = False):
    for p in get_db_paths(base_path):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                if verbose:
                    print(f"Failed to remove {p}")


def setup_db(db_size: int, storage_file_base: str, verbose: bool = False) -> PicoVectorDB:
    if verbose:
        print(f"Setting up DB: {storage_file_base} with {db_size:,} vectors …")
    cleanup_db_files(storage_file_base, verbose=verbose)

    db = PicoVectorDB(
        embedding_dim=DIMENSION,
        storage_file=storage_file_base,
        no_faiss=True,
    )

    rng = np.random.default_rng(123)
    vectors_to_insert = rng.random((db_size, DIMENSION), dtype=np.float32)

    data_to_upsert = []
    for i in range(db_size):
        data_to_upsert.append(
            {
                K_VECTOR: vectors_to_insert[i],
                K_ID: f"id_{i}",
                "content": f"Document content for id_{i}",
                "category_id": i % 10,  # for where filters (10% buckets)
            }
        )

    t0 = time.perf_counter()
    db.upsert(data_to_upsert)
    t1 = time.perf_counter()
    if verbose:
        print(f"Inserted {db_size:,} vectors in {(t1 - t0):.3f}s")
    return db


def time_query(
    db: PicoVectorDB,
    q: np.ndarray,
    where_fn: Optional[Callable] = None,
    better_than: Optional[float] = None,
    ids: Optional[list[str]] = None,
) -> float:
    t0 = time.perf_counter()
    db.query(q, top_k=TOP_K, where=where_fn, better_than=better_than, ids=ids)
    t1 = time.perf_counter()
    return t1 - t0


def summarize(times: list[float]) -> dict[str, float]:
    arr = np.array(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean() * 1000.0),
        "p50_ms": float(np.percentile(arr, 50) * 1000.0),
        "p95_ms": float(np.percentile(arr, 95) * 1000.0),
        "ops_sec": float(len(arr) / arr.sum()) if arr.sum() > 0 else float("inf"),
    }


def scenario_generators(db: PicoVectorDB, db_size: int):
    # Precompute id lists for ID-restricted scenarios
    ids_all = [f"id_{i}" for i in range(db_size)]
    rng = np.random.default_rng(42)
    idxs_1p = rng.choice(db_size, size=max(1, db_size // 100), replace=False)
    idxs_10p = rng.choice(db_size, size=max(1, db_size // 10), replace=False)
    ids_1p = [ids_all[i] for i in idxs_1p]
    ids_10p = [ids_all[i] for i in idxs_10p]

    def where_even(meta):
        return meta.get("category_id", -1) % 2 == 0  # ~50%

    def where_10p(meta):
        return meta.get("category_id", -1) == 0  # ~10%

    return [
        ("baseline", lambda: dict(where=None, better_than=None, ids=None)),
        ("where_even(50%)", lambda: dict(where=where_even, better_than=None, ids=None)),
        ("where_10%(10%)", lambda: dict(where=where_10p, better_than=None, ids=None)),
        ("ids_10%(10%)", lambda: dict(where=None, better_than=None, ids=ids_10p)),
        ("ids_1%(1%)", lambda: dict(where=None, better_than=None, ids=ids_1p)),
        ("better_than(0.1)", lambda: dict(where=None, better_than=0.1, ids=None)),
        (
            "where_even+bt(0.1)",
            lambda: dict(where=where_even, better_than=0.1, ids=None),
        ),
    ]


def run_suite(
    db_size: int,
    num_queries: int,
    batch_sizes: list[int],
    verbose: bool = False,
):
    base = f"temp_db_{db_size}"
    db = setup_db(db_size, base, verbose=verbose)
    try:
        rng = np.random.default_rng(99)
        # Single-query scenarios
        qs = rng.random((num_queries, DIMENSION), dtype=np.float32)
        scenarios = scenario_generators(db, db_size)
        print(f"\nDB: {db_size:,} vectors | dim={DIMENSION} | top_k={TOP_K}")
        for name, cfg_fn in scenarios:
            cfg = cfg_fn()
            times = [
                time_query(db, qs[i], where_fn=cfg["where"], better_than=cfg["better_than"], ids=cfg["ids"])
                for i in range(num_queries)
            ]
            stats = summarize(times)
            print(
                f"  {name:18s}  mean={stats['mean_ms']:.3f}ms  p50={stats['p50_ms']:.3f}ms  p95={stats['p95_ms']:.3f}ms  ops/s={stats['ops_sec']:.1f}"
            )

        # Batch-query scenarios (2D inputs)
        for b in batch_sizes:
            if b <= 1:
                continue
            q_batch = rng.random((b, DIMENSION), dtype=np.float32)
            t = time_query(db, q_batch)
            print(f"  batch({b:>4d})        mean={t*1000.0:.3f}ms  per_q={t*1000.0/b:.3f}ms")
    finally:
        cleanup_db_files(base, verbose=verbose)


def main(db_sizes_to_test: list[int], num_queries: int, batch_sizes: list[int], verbose: bool):
    print("NumPy Query Path Summary (concise)")
    print("=" * 40)
    for size in db_sizes_to_test:
        run_suite(size, num_queries=num_queries, batch_sizes=batch_sizes, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile PicoVectorDB NumPy query path (concise summary)")
    parser.add_argument(
        "--db_sizes",
        type=str,
        default="10000,50000",
        help="Comma-separated database sizes (e.g., 10000,50000,100000)",
    )
    parser.add_argument("--dim", type=int, default=DIMENSION, help="Embedding dimension")
    parser.add_argument(
        "--num_queries",
        type=int,
        default=NUM_QUERIES,
        help="Queries per scenario (single-vector path)",
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,8,32",
        help="Comma-separated batch sizes for 2D query inputs",
    )
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-k neighbors")
    parser.add_argument("--verbose", action="store_true", help="Verbose setup logs")
    args = parser.parse_args()

    # apply CLI overrides
    DIMENSION = args.dim
    NUM_QUERIES = args.num_queries
    TOP_K = args.top_k

    try:
        db_sizes = [int(s.strip()) for s in args.db_sizes.split(",") if s.strip()]
    except ValueError:
        print("Invalid --db_sizes; use comma-separated integers.")
        raise SystemExit(2)

    try:
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",") if s.strip()]
    except ValueError:
        print("Invalid --batch_sizes; use comma-separated integers.")
        raise SystemExit(2)

    main(db_sizes_to_test=db_sizes, num_queries=NUM_QUERIES, batch_sizes=batch_sizes, verbose=args.verbose)
    print("\nDone.")
