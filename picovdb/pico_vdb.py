"""
PicoVectorDB: A simple and fast vector database
"""
import os
import json
import hashlib
import logging
from typing import Any, Callable, Literal, Optional, Union
from threading import RLock

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

# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

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
        no_faiss: bool = False,
    ) -> None:
        # Initialize RWLock for thread safety
        self._lock = RLock()
        self.dim = embedding_dim
        self.metric = metric
        self._path = storage_file
        self._use_memmap = use_memmap

        # in‑memory parallel lists ------------------------------------------------
        self._vectors: np.ndarray  # (N, dim) float32 & L2‑normalised
        self._ids: list[str]
        self._docs: list[Optional[dict[str, Any]]]
        self._free: list[int] = []
        self._id2idx: dict[str, int] = {}
        self._additional: dict[str, Any] = {}
        # Active (non-deleted) row indices for fast filtering
        self._active_indices: np.ndarray = np.empty(0, dtype=np.int64)

        # faiss index ---------------------------------------------------------
        # self._faiss = faiss.IndexFlatIP(self.dim) if _HAS_FAISS else None
        if _HAS_FAISS and not no_faiss:
            self._faiss = faiss.IndexHNSWFlat(self.dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            self._faiss.hnsw.efConstruction = HNSW_EFC
        else:
            self._faiss = None

        self._load_or_init()

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------

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
                else np.load(vecs_file)
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
                if os.path.exists(vecs_file + ".faiss"):
                    self._faiss = faiss.read_index(vecs_file + ".faiss")
                else:
                    self._rebuild_faiss()
            logger.info("Loaded %d active / %d total vectors", len(self._id2idx), count)
        else:
            self._ids, self._docs = [], []
            self._vectors = np.empty((0, self.dim), dtype=Float)
            self._active_indices = np.empty(0, dtype=np.int64)
            logger.info("No persisted data – fresh DB")

    def size(self) -> int:
        return len(self._ids)

    def save(self) -> None:
        """
        Persist the current state of the database, overwrite existing files.
        """
        with self._lock:
            ids_file, vecs_file, meta_file = (
                _ids_path(self._path),
                _vecs_path(self._path),
                _meta_path(self._path),
            )
            # ids quick‑load file --------------------------------------------------
            with open(ids_file, "w", encoding="utf‑8") as f:
                json.dump(self._ids, f, ensure_ascii=False)
            # vectors -------------------------------------------------------------
            np.save(vecs_file, self._vectors)
            if self._faiss:
                faiss.write_index(self._faiss, vecs_file + ".faiss")
            # full metadata -------------------------------------------------------
            meta_json = {
                "embedding_dim": self.dim,
                "data": self._docs,
                "additional_data": self._additional,
            }
            with open(meta_file, "w", encoding="utf‑8") as f:
                json.dump(meta_json, f, ensure_ascii=False)
            logger.info("Saved %d vectors", len(self._ids))

    def upsert(self, items: list[dict[str, Any]]) -> dict[str, list[str]]:
        """---------------------------------------------------------------------
        # Mutators
        """
        with self._lock:
            report : dict[str, list[str]] = {"update": [], "insert": []}
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
                item_id = meta.get(K_ID) if meta.get(K_ID) is not None else _hash_vec(vec)
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
                self._vectors = (
                    stacked if not self._ids else np.vstack([self._vectors, stacked])
                )
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
                self._rebuild_faiss()
            return report

    def store_additional_data(self, **kwargs) -> None:
        """Store additional data in the metadata file.

        This data is not used for vector search, but can be useful for storing
        other information related to the vectors.
        """
        with self._lock:
            self._additional.update(kwargs)

    def get_additional_data(self) -> dict[str, Any]:
        """Get additional data stored in the metadata file."""
        return self._additional

    def delete(self, ids: list[str]) -> list[str]:
        """ Delete vectors by IDs, return deleted IDs."""
        with self._lock:
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
                self._rebuild_faiss()
            return removed

    def query(
        self,
        query_vecs: np.ndarray,
        top_k: int = 10,
        better_than: Optional[float] = None,
        where: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> Union[list[list[dict[str, Any]]], list[dict[str, Any]]]:
        with self._lock:
            """---------------------------------------------------------------------
            # Query
            """
            # prepare empty batch result if no vectors
            raw = np.ascontiguousarray(query_vecs, dtype=Float)
            # Validate input shape: 1D (dim,) or 2D (batch, dim)
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
            if not self._id2idx:
                return [[] for _ in range(num_q)]
            # normalize each query vector
            # batch normalize without Python loop
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            # For zero vectors, set to a deterministic unit vector on first axis
            zero_mask = norms.squeeze(-1) == 0
            if np.any(zero_mask):
                vecs = vecs.copy()
                vecs[zero_mask] = 0
                vecs[zero_mask, 0] = 1.0
                norms = np.where(zero_mask[:, None], 1.0, norms)
            vecs = (vecs / norms).astype(Float, copy=False)
            # compute scores and indices batch-wise
            if self._faiss is not None:
                self._faiss.hnsw.efSearch = HNSW_EFS
                scores_batch, idxs_batch = self._faiss.search(vecs, top_k)
            else:
                # Use only active (non-deleted) rows for scoring without copying (avoid large fancy-indexed candidate matrix)
                active_idx = self._active_indices
                if active_idx.size == 0:
                    return [[] for _ in range(num_q)]
                # Compute full scores then slice active columns; cheaper than slicing vectors first
                scores_full = vecs @ self._vectors.T  # shape (num_q, N)
                scores_act = scores_full[:, active_idx]  # shape (num_q, M_active)
                k_eff = min(top_k, scores_act.shape[1])
                # partial top-k indices per query along axis=1 (local to active set)
                idxs_part_local = np.argpartition(scores_act, -k_eff, axis=1)[:, -k_eff:]
                # gather scores for those local indices
                scores_part = np.take_along_axis(scores_act, idxs_part_local, axis=1)
                # sort within top-k
                order = np.argsort(-scores_part, axis=1)
                # local indices ordered by score
                idxs_batch_local = np.take_along_axis(idxs_part_local, order, axis=1)
                scores_batch = np.take_along_axis(scores_part, order, axis=1)
                # map local active positions back to global row indices
                idxs_batch = active_idx[idxs_batch_local]
            # build results for each query
            results_batch: list[list[dict[str, Any]]] = []
            for qi in range(num_q):
                idxs = idxs_batch[qi]
                scores = scores_batch[qi]
                results: list[dict[str, Any]] = []
                for idx, score in zip(idxs, scores):
                    if idx < 0 or idx >= len(self._ids):
                        continue
                    # Skip deleted entries (doc is None)
                    doc = self._docs[idx]
                    if doc is None:
                        continue
                    doc_id = self._ids[idx]
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild_faiss(self) -> None:
        if self._faiss is None:
            return
        self._faiss.reset()
        if self._vectors.size:
            self._faiss.add(self._vectors)

    def __len__(self) -> int:
        return len(self._id2idx)

    def get(self, ids: list[str], include_vector: bool = False) -> list[dict[str, Any]]:
        """
        Get records by IDs.

        Returns metadata by default; when `include_vector=True`, also include `_vector_`.
        """
        with self._lock:
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

    def get_by_id(self, sid: str, include_vector: bool = False) -> Optional[dict[str, Any]]:
        """
        Get a single record by ID.

        Returns metadata by default; when `include_vector=True`, also include `_vector_`.
        """
        with self._lock:
            idx = self._id2idx.get(sid)
            if idx is not None:
                meta = self._docs[idx] or {K_ID: sid}
                rec = dict(meta)
                if include_vector:
                    rec[K_VECTOR] = self._vectors[idx].copy()
                return rec
            return None

    def get_all(self, include_vector: bool = False, include_deleted: bool = False) -> list[dict[str, Any]]:
        """
        Get all records.

        Returns metadata by default; when `include_vector=True`, also include `_vector_` for active records.
        By default returns only active (non-deleted) records; set `include_deleted=True` to include deleted placeholders.
        """
        with self._lock:
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
