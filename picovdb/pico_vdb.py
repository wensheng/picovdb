"""
PicoVectorDB: A simple and fast vector database
"""

import os
import json
import hashlib
import logging
import warnings
import time
from typing import Any, Callable, Literal, Optional, Union
from threading import RLock
import threading
from contextlib import contextmanager

import numpy as np

# optional FAISS --------------------------------------------------------------
try:
    import faiss  # type: ignore

    # import platform
    # if platform.system() == "Darwin":
    #     faiss.omp_set_num_threads(1)  # without this it crashes when 1.10.0, 1.11.0.  1.9.0 is OK.
    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _HAS_FAISS = False


Float = np.float32
HNSW_M = 32  # number of connections each vertex will have
HNSW_EFC = 40  # depth of layers explored during index construction
HNSW_EFS = 32  # depth of layers explored during search
ADAPTIVE_BUFFER = 32  # extra candidates to fetch under filters
K_ID = "_id_"
K_VECTOR = "_vector_"
K_METRICS = "_metrics_"

logger = logging.getLogger("picovdb")

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _ids_path(base: str) -> str:
    return f"{base}.ids.json"


def _meta_path(base: str) -> str:
    return f"{base}.meta.json"


def _vecs_path(base: str) -> str:
    return f"{base}.vecs.npy"


def _hash_vec(v: np.ndarray) -> str:
    return hashlib.md5(v.tobytes()).hexdigest()


def _normalize(v: np.ndarray) -> np.ndarray:
    # Single-pass normalization with zero-safe handling (no recursion)
    vec = np.asarray(v, dtype=Float)
    n = float(np.linalg.norm(vec))
    if n == 0.0:
        # deterministically map zero to a unit vector on first axis
        out = np.zeros_like(vec, dtype=Float)
        if out.size:
            out.flat[0] = Float(1.0)
        return out
    return (vec / n).astype(Float, copy=False)


def _to_c_f32(a: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float32 view/copy of the array."""
    return np.ascontiguousarray(a, dtype=Float)


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------


def _timed(name: str):
    """Decorator for DEBUG-level timing."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            logger.debug("%s took %.4f ms", name, (end - start) * 1000)
            return result

        return wrapper

    return decorator


class PicoVectorDB:
    """Cosine‑only vector DB with metadata persistence.

    * Saves **both** quick‑load ids file and full metadata file (*my_store.ids.json* + *my_store.meta.json*).
    * Stores contiguous float32 matrix for fast similarity search.
    * Supports optional Faiss acceleration for large datasets.
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        metric: Literal["cosine"] = "cosine",
        storage_file: str = "picovdb",
        use_memmap: bool = False,
        capacity: Optional[int] = None,
        no_faiss: bool = False,
        faiss_threads: Optional[int] = None,
        # FAISS HNSW tunables
        hnsw_m: Optional[int] = None,
        hnsw_ef_construction: Optional[int] = None,
        # Back-compat name kept; prefer `hnsw_ef_search_default` if both provided
        ef_search_default: Optional[int] = None,
        hnsw_ef_search_default: Optional[int] = None,
    ) -> None:
        # Initialize RWLock for thread safety
        self._lock = RLock()
        self._rwlock = _RWLock()
        self.dim = embedding_dim
        self.metric = metric
        self._path = storage_file
        self._use_memmap = use_memmap
        self._capacity = capacity

        # in‑memory parallel lists ------------------------------------------------
        self._vectors: np.ndarray  # (N, dim) float32 & L2‑normalised
        self._ids: list[str]
        self._docs: list[Optional[dict[str, Any]]]
        self._free: list[int] = []
        self._id2idx: dict[str, int] = {}
        self._additional: dict[str, Any] = {}
        # Active (non-deleted) row indices for fast filtering
        self._active_indices: np.ndarray = np.empty(0, dtype=np.int64)

        # HNSW params (store resolved values)
        self._hnsw_m: int = int(hnsw_m) if hnsw_m is not None else HNSW_M
        self._hnsw_efc: int = (
            int(hnsw_ef_construction)
            if hnsw_ef_construction is not None
            else HNSW_EFC
        )

        # faiss index ---------------------------------------------------------
        # self._faiss = faiss.IndexFlatIP(self.dim) if _HAS_FAISS else None
        if _HAS_FAISS and not no_faiss:
            base = faiss.IndexHNSWFlat(
                self.dim, self._hnsw_m, faiss.METRIC_INNER_PRODUCT
            )
            base.hnsw.efConstruction = self._hnsw_efc
            self._faiss = faiss.IndexIDMap2(base)
            # optionally set FAISS threads
            if faiss_threads is not None and hasattr(faiss, "omp_set_num_threads"):
                try:
                    faiss.omp_set_num_threads(int(faiss_threads))  # type: ignore[attr-defined]
                except Exception:
                    pass
        else:
            self._faiss = None
        # dirty flag for lazy FAISS rebuilds
        self._dirty: bool = False
        # default efSearch for FAISS HNSW
        # prefer new kwarg if provided, otherwise fall back to legacy name
        if hnsw_ef_search_default is not None:
            self._faiss_ef_search = int(hnsw_ef_search_default)
        elif ef_search_default is not None:
            self._faiss_ef_search = int(ef_search_default)
        else:
            self._faiss_ef_search = HNSW_EFS

        self._load_or_init()

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------

    @_timed("load")
    def _load_or_init(self) -> None:
        ids_file, vecs_file, meta_file = (
            _ids_path(self._path),
            _vecs_path(self._path),
            _meta_path(self._path),
        )
        if os.path.exists(ids_file) and os.path.exists(vecs_file):
            logger.info("Loading existing DB …")
            # ids ----------------------------------------------------------------
            with open(ids_file, "r", encoding="utf‑8") as f:
                self._ids = json.load(f)
            count = len(self._ids)
            self._vectors = (
                np.memmap(vecs_file, dtype=Float, mode="r+", shape=(count, self.dim))
                if self._use_memmap
                else _to_c_f32(np.load(vecs_file))
            )
            # metadata (optional) ----------------------------------------------
            if os.path.exists(meta_file):
                with open(meta_file, "r", encoding="utf‑8") as f:
                    meta_json = json.load(f)
                self._docs = meta_json.get("data", [None] * count)
                self._additional = meta_json.get("additional_data", {})
            else:
                self._docs = [None] * count
            # build maps & free list ------------------------------------------
            for i, (_id, doc) in enumerate(zip(self._ids, self._docs)):
                if doc is None:
                    self._free.append(i)
                else:
                    if _id is not None:
                        self._id2idx[_id] = i
            # build active indices (non-deleted positions)
            if self._id2idx:
                self._active_indices = np.fromiter(
                    (idx for idx in self._id2idx.values()), dtype=np.int64
                )
            else:
                self._active_indices = np.empty(0, dtype=np.int64)
            if self._faiss is not None:
                faiss_path = vecs_file + ".faiss"
                if os.path.exists(faiss_path):
                    try:
                        idx = faiss.read_index(faiss_path)
                        # validate dimension; for IDMap2, d is available on the wrapper
                        d = getattr(idx, "d", None)
                        if d is None and hasattr(idx, "index"):
                            d = getattr(idx.index, "d", None)  # type: ignore[attr-defined]
                        if d != self.dim:
                            logger.warning(
                                "FAISS index dim %s != expected %s; rebuilding",
                                d,
                                self.dim,
                            )
                            self._rebuild_faiss()
                        else:
                            self._faiss = idx
                    except Exception:
                        logger.warning("Failed to read FAISS index; rebuilding")
                        self._rebuild_faiss()
                else:
                    self._rebuild_faiss()
                self._dirty = False
            logger.info("Loaded %d active / %d total vectors", len(self._id2idx), count)
        else:
            if self._use_memmap and self._capacity is not None:
                # Pre-allocate memmap file
                self._vectors = np.memmap(
                    vecs_file,
                    dtype=Float,
                    mode="w+",
                    shape=(self._capacity, self.dim),
                )
                self._ids = [None] * self._capacity
                self._docs = [None] * self._capacity
                self._free = list(range(self._capacity))
            else:
                self._ids, self._docs = [], []
                self._vectors = np.empty((0, self.dim), dtype=Float)
            self._active_indices = np.empty(0, dtype=np.int64)
            logger.info("No persisted data – fresh DB")
            self._dirty = False

    def size(self) -> int:
        """
        Deprecated: returns total slots (including deleted placeholders).
        Use `count()` for active item count. A `capacity()` method will be provided later.
        """
        warnings.warn(
            "size() is deprecated: use count() for active items; capacity() will be added in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        with self._rwlock.read_lock():
            return len(self._ids)

    def capacity(self) -> int:
        """
        Returns total slots (including deleted placeholders).
        Use `count()` for active item count.
        """
        with self._rwlock.read_lock():
            return len(self._ids)

    def count(self) -> int:
        """Return the number of active (non-deleted) items."""
        with self._rwlock.read_lock():
            return len(self._id2idx)

    @_timed("save")
    def save(self) -> None:
        """
        Persist the current state of the database atomically, overwriting existing files.
        Writes to temporary files and replaces them to avoid corruption.
        """
        with self._rwlock.write_lock():
            ids_file, vecs_file, meta_file = (
                _ids_path(self._path),
                _vecs_path(self._path),
                _meta_path(self._path),
            )
            # Use temporary files for atomic writes
            tmp_ids_file = f"{ids_file}.tmp"
            tmp_vecs_file_base = f"{self._path}.vecs.tmp"  # No .npy extension
            tmp_vecs_file = f"{tmp_vecs_file_base}.npy"
            tmp_meta_file = f"{meta_file}.tmp"
            faiss_file = f"{vecs_file}.faiss"
            tmp_faiss_file = f"{faiss_file}.tmp"

            try:
                # ids quick‑load file --------------------------------------------------
                with open(tmp_ids_file, "w", encoding="utf‑8") as f:
                    json.dump(self._ids, f, ensure_ascii=False)

                # vectors -------------------------------------------------------------
                np.save(tmp_vecs_file_base, self._vectors)  # np.save adds .npy
                if self._faiss:
                    if self._dirty:
                        # Ensure on-disk index reflects current vectors
                        self._rebuild_faiss()
                        self._dirty = False
                    faiss.write_index(self._faiss, tmp_faiss_file)

                # full metadata -------------------------------------------------------
                meta_json = {
                    "embedding_dim": self.dim,
                    "data": self._docs,
                    "additional_data": self._additional,
                }
                with open(tmp_meta_file, "w", encoding="utf‑8") as f:
                    json.dump(meta_json, f, ensure_ascii=False)

                # Atomically replace old files with new ones
                os.replace(tmp_ids_file, ids_file)
                os.replace(tmp_vecs_file, vecs_file)
                os.replace(tmp_meta_file, meta_file)
                if self._faiss and os.path.exists(tmp_faiss_file):
                    os.replace(tmp_faiss_file, faiss_file)

                logger.info("Saved %d vectors", len(self._ids))
            finally:
                # Clean up temporary files in case of an error
                for tmp_file in [
                    tmp_ids_file,
                    tmp_vecs_file,
                    tmp_meta_file,
                    tmp_faiss_file,
                ]:
                    if os.path.exists(tmp_file):
                        try:
                            os.remove(tmp_file)
                        except OSError:
                            pass

    def flush(self) -> None:
        """
        If using memmap, flushes changes to disk. No-op otherwise.
        """
        with self._rwlock.read_lock():
            if self._use_memmap and isinstance(self._vectors, np.memmap):
                self._vectors.flush()

    def upsert(self, items: list[dict[str, Any]]) -> dict[str, list[str]]:
        """---------------------------------------------------------------------
        # Mutators
        """
        with self._rwlock.write_lock():
            report: dict[str, list[str]] = {"update": [], "insert": []}
            new_vecs, new_ids, new_docs = [], [], []
            new_active: list[int] = []
            for item in items:
                # Coerce and validate vector shape
                vec_raw = np.ascontiguousarray(item[K_VECTOR], dtype=Float)
                if vec_raw.ndim != 1:
                    raise ValueError(
                        f"upsert vector must be 1D with length {self.dim}; got shape {tuple(vec_raw.shape)}"
                    )
                if vec_raw.shape[0] != self.dim:
                    raise ValueError(
                        f"upsert vector dim mismatch: expected {self.dim}, got {vec_raw.shape[0]}"
                    )
                vec = _normalize(vec_raw)
                meta = {k: v for k, v in item.items() if k != K_VECTOR}
                item_id = (
                    meta.get(K_ID) if meta.get(K_ID) is not None else _hash_vec(vec)
                )
                meta[K_ID] = item_id
                if item_id in self._id2idx:
                    idx = self._id2idx[item_id]
                    self._vectors[idx] = vec
                    self._docs[idx] = meta
                    report["update"].append(item_id)
                else:
                    if self._free:
                        idx = self._free.pop()
                        self._vectors[idx] = vec
                        self._ids[idx] = item_id
                        self._docs[idx] = meta
                        new_active.append(idx)
                    else:
                        if self._capacity is not None:
                            raise ValueError("Database capacity exceeded")
                        new_vecs.append(vec)
                        new_ids.append(item_id)
                        new_docs.append(meta)
                        idx = len(self._ids) + len(new_ids) - 1
                        new_active.append(idx)
                    self._id2idx[item_id] = idx
                    report["insert"].append(item_id)
            # bulk append ---------------------------------------------------------
            if new_vecs:
                stacked = np.vstack(new_vecs)
                if not self._ids:
                    self._vectors = _to_c_f32(stacked)
                else:
                    if self._use_memmap and isinstance(self._vectors, np.memmap):
                        logger.warning(
                            "Appending to a memmapped file converts it to an in-memory numpy array, "
                            "doubling memory usage. For large datasets, consider pre-allocating "
                            "capacity or using a different growth strategy."
                        )
                    self._vectors = _to_c_f32(np.vstack([self._vectors, stacked]))
                self._ids.extend(new_ids)
                self._docs.extend(new_docs)
            # update active indices
            if new_active:
                if self._active_indices.size:
                    self._active_indices = np.append(
                        self._active_indices, np.asarray(new_active, dtype=np.int64)
                    )
                else:
                    self._active_indices = np.asarray(new_active, dtype=np.int64)
            if self._faiss is not None:
                self._dirty = True
            return report

    def store_additional_data(self, **kwargs) -> None:
        """Store additional data in the metadata file.

        This data is not used for vector search, but can be useful for storing
        other information related to the vectors.
        """
        with self._rwlock.write_lock():
            self._additional.update(kwargs)

    def get_additional_data(self) -> dict[str, Any]:
        """Get additional data stored in the metadata file."""
        with self._rwlock.read_lock():
            return self._additional

    def delete(self, ids: list[str]) -> list[str]:
        """Delete vectors by IDs, return deleted IDs."""
        with self._rwlock.write_lock():
            removed = []
            removed_idxs: list[int] = []
            for _id in ids:
                idx = self._id2idx.pop(_id, None)
                if idx is not None:
                    self._docs[idx] = None
                    self._vectors[idx].fill(0)
                    self._free.append(idx)
                    removed_idxs.append(idx)
                    removed.append(_id)
            # update active indices by removing deleted rows
            if removed_idxs and self._active_indices.size:
                to_remove = np.asarray(removed_idxs, dtype=np.int64)
                mask = ~np.isin(self._active_indices, to_remove)
                self._active_indices = self._active_indices[mask]
            if removed and self._faiss is not None:
                self._dirty = True
            return removed

    @_timed("query")
    def query(
        self,
        query_vecs: np.ndarray,
        top_k: int = 10,
        better_than: Optional[float] = None,
        where: Optional[Callable[[dict[str, Any]], bool]] = None,
        ids: Optional[list[str]] = None,
        # Back-compat: `ef_search`; new alias `hnsw_ef_search`
        ef_search: Optional[int] = None,
        hnsw_ef_search: Optional[int] = None,
    ) -> Union[list[list[dict[str, Any]]], list[dict[str, Any]]]:
        """Query the database.

        For the NumPy path, snapshots read-only references under the read lock and releases
        the lock before heavy math to improve concurrency. For the FAISS path, keeps the
        read lock held to avoid concurrent mutation of the index during search.
        """
        # Prepare and validate input first (no lock needed)
        raw = np.ascontiguousarray(query_vecs, dtype=Float)
        if raw.ndim == 1:
            if raw.shape[0] != self.dim:
                raise ValueError(
                    f"query vector dim mismatch: expected {self.dim}, got {raw.shape[0]}"
                )
            is_single = True
        elif raw.ndim == 2:
            if raw.shape[1] != self.dim:
                raise ValueError(
                    f"query vectors dim mismatch: expected last dim {self.dim}, got {raw.shape[1]}"
                )
            is_single = False
        else:
            raise ValueError(
                f"query expects 1D or 2D array with last dim {self.dim}; got shape {tuple(raw.shape)}"
            )
        vecs = raw[None, :] if is_single else raw
        num_q = vecs.shape[0]
        # Normalize queries (no lock)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        zero_mask = norms.squeeze(-1) == 0
        if np.any(zero_mask):
            vecs = vecs.copy()
            vecs[zero_mask] = 0
            vecs[zero_mask, 0] = 1.0
            norms = np.where(zero_mask[:, None], 1.0, norms)
        vecs = (vecs / norms).astype(Float, copy=False)

        # Snapshot under read lock
        with self._rwlock.read_lock():
            if not self._id2idx:
                return [[] for _ in range(num_q)]
            use_faiss = self._faiss is not None
            needs_rebuild = use_faiss and self._dirty
            # Views under lock (no copies yet)
            vectors_view = self._vectors
            active_idx_view = self._active_indices
            ids_view = self._ids
            docs_view = self._docs
            # Build candidate indices from filters
            candidate_idx: Optional[np.ndarray] = None
            if ids is not None:
                mapped = [self._id2idx.get(s) for s in ids]
                mapped = [m for m in mapped if m is not None]
                if mapped:
                    candidate_idx = np.asarray(sorted(set(mapped)), dtype=np.int64)
                else:
                    candidate_idx = np.empty(0, dtype=np.int64)
            if where is not None:
                mask = [
                    where(docs_view[i]) if docs_view[i] is not None else False
                    for i in active_idx_view
                ]
                filtered = active_idx_view[np.asarray(mask, dtype=bool)]
                if candidate_idx is None:
                    candidate_idx = filtered
                else:
                    candidate_idx = np.intersect1d(
                        candidate_idx, filtered, assume_unique=False
                    )
            if candidate_idx is None:
                candidate_idx = active_idx_view

        if use_faiss and needs_rebuild:
            # Upgrade to write lock to rebuild FAISS lazily
            with self._rwlock.write_lock():
                if self._faiss is not None and self._dirty:
                    self._rebuild_faiss()
                    self._dirty = False

        # If FAISS is available but we have a restricted candidate set, prefer NumPy path
        faiss_ok = use_faiss and candidate_idx.size == active_idx_view.size

        if not faiss_ok:
            # Heavy math outside locks (NumPy path)
            if candidate_idx.size == 0:
                return [[] for _ in range(num_q)]
            # Snapshot arrays and lists before releasing lock
            with self._rwlock.read_lock():
                vectors_ref = vectors_view
                candidate_ref = candidate_idx.copy()
                ids_ref = list(ids_view)
                docs_ref = list(docs_view)
            # Pre-slice candidate vectors to avoid full-matrix matmul
            # Fast path: when no filters and candidates cover all active rows,
            # avoid building a large copy and use the original full-matrix GEMM.
            if (ids is None and where is None) and (
                candidate_ref.size == vectors_ref.shape[0]
            ):
                scores_act = vecs @ vectors_ref.T  # (num_q, N)
            else:
                vectors_cand = vectors_ref[candidate_ref]
                scores_act = vecs @ vectors_cand.T  # (num_q, |candidates|)
            # If filters are present, fetch extra candidates to mitigate underfill after filtering
            base = (
                top_k + ADAPTIVE_BUFFER
                if (ids is not None or where is not None)
                else top_k
            )
            k_eff = min(base, scores_act.shape[1])
            # Heuristic: prefer full argsort when k_eff is a large fraction of candidates
            frac = k_eff / scores_act.shape[1] if scores_act.shape[1] > 0 else 0.0
            if frac > 0.2:
                order_full = np.argsort(-scores_act, axis=1)[:, :k_eff]
                idxs_batch_local = order_full
                scores_batch = np.take_along_axis(scores_act, idxs_batch_local, axis=1)
            else:
                idxs_part_local = np.argpartition(scores_act, -k_eff, axis=1)[
                    :, -k_eff:
                ]
                scores_part = np.take_along_axis(scores_act, idxs_part_local, axis=1)
                order = np.argsort(-scores_part, axis=1)
                idxs_batch_local = np.take_along_axis(idxs_part_local, order, axis=1)
                scores_batch = np.take_along_axis(scores_part, order, axis=1)
            idxs_batch = candidate_ref[idxs_batch_local]

        if faiss_ok:
            # Perform FAISS search and build results under read lock
            with self._rwlock.read_lock():
                try:
                    # prefer per-call hnsw_ef_search, fall back to legacy `ef_search`, then default
                    if hnsw_ef_search is not None:
                        ef = int(hnsw_ef_search)
                    elif ef_search is not None:
                        ef = int(ef_search)
                    else:
                        ef = self._faiss_ef_search
                    self._faiss.index.hnsw.efSearch = ef  # type: ignore[attr-defined]
                except Exception:
                    pass
                scores_batch, idxs_batch = self._faiss.search(vecs, top_k)
                ids_ref = ids_view
                docs_ref = docs_view
                results_batch: list[list[dict[str, Any]]] = []
                for qi in range(num_q):
                    idxs = idxs_batch[qi]
                    scores = scores_batch[qi]
                    results: list[dict[str, Any]] = []
                    for idx, score in zip(idxs, scores):
                        if idx < 0 or idx >= len(ids_ref):
                            continue
                        # Since faiss_ok implies no filter beyond actives, no need to test where/ids here
                        doc = docs_ref[idx]
                        if doc is None:
                            continue
                        if better_than is not None and score < better_than:
                            continue
                        results.append({**doc, K_METRICS: float(score)})
                        if len(results) == top_k:
                            break
                    results_batch.append(results)
                return results_batch[0] if is_single else results_batch
        else:
            # Build results without lock using snapshots (NumPy path)
            results_batch: list[list[dict[str, Any]]] = []
            for qi in range(num_q):
                idxs = idxs_batch[qi]
                scores = scores_batch[qi]
                results: list[dict[str, Any]] = []
                for idx, score in zip(idxs, scores):
                    if idx < 0 or idx >= len(ids_ref):
                        continue
                    doc = docs_ref[idx]
                    if doc is None:
                        continue
                    if better_than is not None and score < better_than:
                        continue
                    meta = doc
                    if where and not where(meta):
                        continue
                    results.append({**meta, K_METRICS: float(score)})
                    if len(results) == top_k:
                        break
                results_batch.append(results)
            return results_batch[0] if is_single else results_batch

    def query_one(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        better_than: Optional[float] = None,
        where: Optional[Callable[[dict[str, Any]], bool]] = None,
        ids: Optional[list[str]] = None,
        ef_search: Optional[int] = None,
        hnsw_ef_search: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Convenience method for single-vector queries."""
        return self.query(  # type: ignore[return-value]
            query_vec,
            top_k=top_k,
            better_than=better_than,
            where=where,
            ids=ids,
            ef_search=ef_search,
            hnsw_ef_search=hnsw_ef_search,
        )

    def stats(self) -> dict[str, Any]:
        """Return a dictionary with database statistics."""
        with self._rwlock.read_lock():
            active = self.count()
            total = self.capacity()
            deleted = total - active
            file_sizes = {}
            for path_fn in [_ids_path, _meta_path, _vecs_path]:
                p = path_fn(self._path)
                try:
                    if os.path.exists(p):
                        file_sizes[os.path.basename(p)] = os.path.getsize(p)
                except OSError:
                    pass
            faiss_path = _vecs_path(self._path) + ".faiss"
            if os.path.exists(faiss_path):
                try:
                    file_sizes[os.path.basename(faiss_path)] = os.path.getsize(
                        faiss_path
                    )
                except OSError:
                    pass

            return {
                "active": active,
                "deleted": deleted,
                "total": total,
                "dim": self.dim,
                "faiss": self._faiss is not None,
                "memmap": self._use_memmap,
                "file_sizes": file_sizes,
            }

    def vacuum(self) -> None:
        """
        Compacts the database by removing deleted entries and rebuilding indices.
        """
        with self._rwlock.write_lock():
            if not self._free:
                return  # No deleted items to vacuum

            # Filter out deleted entries
            active_indices = sorted(self._id2idx.values())
            self._vectors = self._vectors[active_indices]
            self._ids = [self._ids[i] for i in active_indices]
            self._docs = [self._docs[i] for i in active_indices]

            # Rebuild mappings
            self._id2idx = {id: i for i, id in enumerate(self._ids)}
            self._active_indices = np.arange(len(self._ids), dtype=np.int64)
            self._free = []

            # Rebuild FAISS index if it exists
            if self._faiss is not None:
                self._rebuild_faiss()
                self._dirty = False

    def rebuild_index(self) -> None:
        """Rebuild the FAISS index immediately if present."""
        with self._rwlock.write_lock():
            if self._faiss is not None:
                self._rebuild_faiss()
                self._dirty = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @_timed("rebuild_faiss")
    def _rebuild_faiss(self) -> None:
        if self._faiss is None:
            return
        # Reset and add only active vectors with global row indices as FAISS IDs
        self._faiss.reset()
        if self._vectors.size:
            active_idx = self._active_indices
            if active_idx.size:
                vecs = self._vectors[active_idx]
                ids = active_idx.astype(np.int64)
                # ensure inner HNSW search params are set
                if hasattr(self._faiss, "index") and hasattr(self._faiss.index, "hnsw"):
                    # set construction depth on inner HNSW before rebuilding
                    try:
                        self._faiss.index.hnsw.efConstruction = self._hnsw_efc
                    except Exception:
                        pass
                self._faiss.add_with_ids(vecs, ids)

    def __len__(self) -> int:
        with self._rwlock.read_lock():
            return len(self._id2idx)

    def get(self, ids: list[str], include_vector: bool = False) -> list[dict[str, Any]]:
        """
        Get records by IDs.

        Returns metadata by default; when `include_vector=True`, also include `_vector_`.
        """
        with self._rwlock.read_lock():
            out = []
            for _id in ids:
                idx = self._id2idx.get(_id)
                if idx is not None:
                    meta = self._docs[idx] or {K_ID: _id}
                    rec = dict(meta)
                    if include_vector:
                        rec[K_VECTOR] = self._vectors[idx].copy()
                    out.append(rec)
            return out

    def get_by_id(
        self, sid: str, include_vector: bool = False
    ) -> Optional[dict[str, Any]]:
        """
        Get a single record by ID.

        Returns metadata by default; when `include_vector=True`, also include `_vector_`.
        """
        with self._rwlock.read_lock():
            idx = self._id2idx.get(sid)
            if idx is not None:
                meta = self._docs[idx] or {K_ID: sid}
                rec = dict(meta)
                if include_vector:
                    rec[K_VECTOR] = self._vectors[idx].copy()
                return rec
            return None

    def get_all(
        self, include_vector: bool = False, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """
        Get all records.

        Returns metadata by default; when `include_vector=True`, also include `_vector_` for active records.
        By default returns only active (non-deleted) records; set `include_deleted=True` to include deleted placeholders.
        """
        with self._rwlock.read_lock():
            docs = []
            if include_deleted:
                # include all slots, with placeholders for deleted
                for _id, doc in zip(self._ids, self._docs):
                    if doc is not None:
                        rec = dict(doc)
                        rec[K_ID] = _id
                        if include_vector:
                            idx = self._id2idx[_id]
                            rec[K_VECTOR] = self._vectors[idx].copy()
                        docs.append(rec)
                    else:
                        docs.append({K_ID: _id})
            else:
                # only active rows, using active indices order
                for idx in self._active_indices.tolist():
                    _id = self._ids[idx]
                    doc = self._docs[idx]
                    if _id is None or doc is None:
                        continue
                    rec = dict(doc)
                    rec[K_ID] = _id
                    if include_vector:
                        rec[K_VECTOR] = self._vectors[idx].copy()
                    docs.append(rec)
            return docs


# -----------------------------------------------------------------------------
# Concurrency
# -----------------------------------------------------------------------------


class _RWLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._readers = 0
        self._writer = False

    @contextmanager
    def read_lock(self):
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_lock(self):
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writer = True

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()
