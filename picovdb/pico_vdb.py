import os
import json
import hashlib
import logging
from typing import Callable, Optional, Literal, Any

import numpy as np


Float = np.float32
f_ID = "_id_"
f_VECTOR = "_vector_"
f_METRICS = "_metrics_"

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
        return v.astype(Float, copy=False)
    return (v / n).astype(Float, copy=False)

# optional FAISS --------------------------------------------------------------
try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _HAS_FAISS = False

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
    ) -> None:
        self.dim = embedding_dim
        self.metric = metric
        self._path = storage_file
        self._use_memmap = use_memmap

        # in‑memory parallel lists ------------------------------------------------
        self._vectors: np.ndarray  # (N, dim) float32 & L2‑normalised
        self._ids: list[Optional[str]]
        self._docs: list[Optional[dict[str, Any]]]
        self._free: list[int] = []
        self._id2idx: dict[str, int] = {}
        self._additional: dict[str, Any] = {}

        # faiss index ---------------------------------------------------------
        self._faiss = faiss.IndexFlatIP(self.dim) if _HAS_FAISS else None

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
            mode = "r+" if self._use_memmap else "r"
            self._vectors = (
                np.memmap(vecs_file, dtype=Float, mode=mode, shape=(count, self.dim))
                if self._use_memmap
                else np.load(vecs_file, mmap_mode="r+")
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
            if _HAS_FAISS:
                self._rebuild_faiss()
            logger.info("Loaded %d active / %d total vectors", len(self._id2idx), count)
        else:
            self._ids, self._docs = [], []
            self._vectors = np.empty((0, self.dim), dtype=Float)
            logger.info("No persisted data – fresh DB")

    def save(self) -> None:
        ids_file, vecs_file, meta_file = (
            _ids_path(self._path),
            _vecs_path(self._path),
            _meta_path(self._path),
        )
        logger.info("Saving DB …")
        # ids quick‑load file --------------------------------------------------
        with open(ids_file, "w", encoding="utf‑8") as f:
            json.dump(self._ids, f, ensure_ascii=False)
        # vectors -------------------------------------------------------------
        np.save(vecs_file, self._vectors)
        # full metadata -------------------------------------------------------
        meta_json = {
            "embedding_dim": self.dim,
            "data": self._docs,
            "additional_data": self._additional,
        }
        with open(meta_file, "w", encoding="utf‑8") as f:
            json.dump(meta_json, f, ensure_ascii=False)
        logger.info("Saved %d vectors", len(self._ids))

    # ---------------------------------------------------------------------
    # Mutators
    # ---------------------------------------------------------------------

    def upsert(self, items: list[dict[str, Any]]) -> dict[str, list[str]]:
        report = {"update": [], "insert": []}
        new_vecs, new_ids, new_docs = [], [], []
        for item in items:
            vec = _normalize(np.asarray(item[f_VECTOR], dtype=Float))
            meta = {k: v for k, v in item.items() if k != f_VECTOR}
            item_id = meta.get(f_ID) or _hash_vec(vec)
            meta[f_ID] = item_id
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
                stacked if not len(self._ids) else np.vstack([self._vectors, stacked])
            )
            self._ids.extend(new_ids)
            self._docs.extend(new_docs)
        if _HAS_FAISS:
            self._rebuild_faiss()
        return report

    def store_additional_data(self, **kwargs) -> None:
        """Store additional data in the metadata file.

        This data is not used for vector search, but can be useful for storing
        other information related to the vectors.
        """
        self._additional.update(kwargs)

    def get_additional_data(self) -> dict[str, Any]:
        """Get additional data stored in the metadata file."""
        return self._additional

    def delete(self, ids: list[str]) -> list[str]:
        removed = []
        for _id in ids:
            idx = self._id2idx.pop(_id, None)
            if idx is not None:
                self._ids[idx] = None
                self._docs[idx] = None
                self._vectors[idx].fill(0)
                self._free.append(idx)
                removed.append(_id)
        if removed and _HAS_FAISS:
            self._rebuild_faiss()
        return removed

    def compact(self) -> None:
        keep = [i for i, _id in enumerate(self._ids) if _id is not None]
        self._vectors = self._vectors[keep].copy()
        self._ids = [self._ids[i] for i in keep]
        self._docs = [self._docs[i] for i in keep]
        self._free.clear()
        self._id2idx = {_id: i for i, _id in enumerate(self._ids)}
        if _HAS_FAISS:
            self._rebuild_faiss()
        logger.info("Compacted – %d vectors", len(self))

    # ---------------------------------------------------------------------
    # Query
    # ---------------------------------------------------------------------

    def query(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        better_than: Optional[float] = None,
        where: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> list[dict[str, Any]]:
        if not self._id2idx:
            return []
        q = _normalize(np.asarray(query_vec, dtype=Float))
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q[None, :], top_k)
            idxs, scores = idxs[0], scores[0]
        else:
            scores = self._vectors @ q
            k = min(top_k, len(scores))
            idxs = np.argpartition(scores, -k)[-k:]
            ord_desc = np.argsort(scores[idxs])[::-1]
            idxs, scores = idxs[ord_desc], scores[idxs][ord_desc]
        results = []
        for idx, score in zip(idxs, scores):
            if idx < 0 or idx >= len(self._ids):
                continue
            doc_id = self._ids[idx]
            if doc_id is None:
                continue
            if better_than is not None and score < better_than:
                continue
            meta = self._docs[idx] or {f_ID: doc_id}
            if where and not where(meta):
                continue
            results.append({**meta, f_METRICS: float(score)})
            if len(results) == top_k:
                break
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild_faiss(self) -> None:
        if self._faiss is None:
            return
        self._faiss.reset()
        if self._vectors.size:
            self._faiss.add(self._vectors)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._id2idx)

    # convenience ------------------------------------------------------------
    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        out = []
        for _id in ids:
            idx = self._id2idx.get(_id)
            if idx is not None:
                out.append(self._docs[idx] or {f_ID: _id})
        return out
