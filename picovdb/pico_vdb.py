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
logging.basicConfig(level=logging.INFO)

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
    n = np.linalg.norm(v)
    if n == 0:
        # Replace 0-vector with small random noise
        noise = np.random.normal(0, 0.01, size=v.shape).astype(Float)
        return _normalize(noise)  # Recursively normalize the noise vector
    return (v / n).astype(Float, copy=False)

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
            for i, _id in enumerate(self._ids):
                if _id is None:
                    self._free.append(i)
                else:
                    self._id2idx[_id] = i
            if self._faiss is not None:
                if os.path.exists(vecs_file + ".faiss"):
                    self._faiss = faiss.read_index(vecs_file + ".faiss")
                else:
                    self._rebuild_faiss()
            logger.info("Loaded %d active / %d total vectors", len(self._id2idx), count)
        else:
            self._ids, self._docs = [], []
            self._vectors = np.empty((0, self.dim), dtype=Float)
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
            for item in items:
                vec = _normalize(np.asarray(item[K_VECTOR], dtype=Float))
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
                    else:
                        new_vecs.append(vec)
                        new_ids.append(item_id)
                        new_docs.append(meta)
                        idx = len(self._ids) + len(new_ids) - 1
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
            for _id in ids:
                idx = self._id2idx.pop(_id, None)
                if idx is not None:
                    self._ids[idx] = None
                    self._docs[idx] = None
                    self._vectors[idx].fill(0)
                    self._free.append(idx)
                    removed.append(_id)
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
            raw = np.asarray(query_vecs, dtype=Float)
            is_single = raw.ndim == 1
            vecs = raw[None, :] if is_single else raw
            num_q = vecs.shape[0]
            if not self._id2idx:
                return [[] for _ in range(num_q)]
            # normalize each query vector
            # batch normalize without Python loop
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs = (vecs / norms).astype(Float, copy=False)
            # compute scores and indices batch-wise
            if self._faiss is not None:
                self._faiss.hnsw.efSearch = HNSW_EFS
                scores_batch, idxs_batch = self._faiss.search(vecs, top_k)
            else:
                # vectorized top-k selection
                scores = self._vectors @ vecs.T  # shape (N, num_q)
                k_eff = min(top_k, self._vectors.shape[0])
                # partial top-k indices per query
                idxs_part = np.argpartition(scores, -k_eff, axis=0)[-k_eff:, :]  # shape (k_eff, num_q)
                # gather scores for those indices
                scores_part = scores[idxs_part, np.arange(num_q)[None, :]]  # shape (k_eff, num_q)
                # sort within top-k
                order = np.argsort(-scores_part, axis=0)  # shape (k_eff, num_q)
                # build final batch indices and scores
                idxs_batch = np.take_along_axis(idxs_part, order, axis=0).T  # shape (num_q, k_eff)
                scores_batch = np.take_along_axis(scores_part, order, axis=0).T
            # build results for each query
            results_batch: list[list[dict[str, Any]]] = []
            for qi in range(num_q):
                idxs = idxs_batch[qi]
                scores = scores_batch[qi]
                results: list[dict[str, Any]] = []
                for idx, score in zip(idxs, scores):
                    if idx < 0 or idx >= len(self._ids):
                        continue
                    doc_id = self._ids[idx]
                    if doc_id is None:
                        continue
                    if better_than is not None and score < better_than:
                        continue
                    meta = self._docs[idx] or {K_ID: doc_id}
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

    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """ Get vectors by IDs, return list of dicts with metadata and vector. """
        with self._lock:
            out = []
            for _id in ids:
                idx = self._id2idx.get(_id)
                if idx is not None:
                    out.append(self._docs[idx] or {K_ID: _id})
            return out

    def get_by_id(self, sid: str) -> Optional[dict[str, Any]]:
        """ Get vector by ID, return dict with metadata and vector. """
        with self._lock:
            idx = self._id2idx.get(sid)
            if idx is not None:
                return self._docs[idx] or {K_ID: sid}
            return None

    def get_all(self) -> list[dict[str, Any]]:
        """ Get all vectors, return list of dicts with metadata and vector. """
        with self._lock:
            docs = []
            for _id, doc in zip(self._ids, self._docs):
                if doc is not None:
                    docs.append(doc | {K_ID: _id})
                else:
                    docs.append({K_ID: _id})
            return docs
