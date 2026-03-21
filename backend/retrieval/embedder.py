# backend/retrieval/embedder.py
"""
FinThesisGuard AI — Text Embedding Engine
ArkAngel Financial Solutions

Converts raw text into 384-dimensional dense vectors using
sentence-transformers/all-MiniLM-L6-v2. Used by both the
ingestion pipeline (document chunking) and retrieval pipeline
(query encoding).

Key design decisions:
- Lazy loading: model is NOT loaded at import time, only on first call
- GPU auto-detection: uses CUDA if available, falls back to CPU silently
- Redis caching: query vectors are cached to avoid re-encoding duplicates
- L2 normalization: all output vectors are unit-normalized for cosine similarity
- Batch size: 32 texts per forward pass (optimal for MiniLM on CPU/GPU)
"""

import time
import numpy as np
from typing import Optional
from functools import lru_cache

from backend.config import (
    settings,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    CACHE_TTL_EMBEDDINGS,
)
from backend.utils.logger import logger, log_metric
from backend.utils.cache import cache, CacheNamespace


# ─────────────────────────────────────────────
# DEVICE DETECTION (module-level, runs once)
# ─────────────────────────────────────────────

def _detect_device() -> str:
    """
    Detects the best available compute device.
    Priority: CUDA GPU → MPS (Apple Silicon) → CPU

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(
                f"[EMBEDDER] CUDA GPU detected: {gpu_name} | "
                f"VRAM: {vram_gb:.1f}GB | using device=cuda"
            )
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("[EMBEDDER] Apple MPS detected | using device=mps")
            return "mps"
    except ImportError:
        pass

    logger.info("[EMBEDDER] No GPU detected | using device=cpu")
    return "cpu"


DEVICE: str = _detect_device()


# ─────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────

class EmbedderError(Exception):
    """Base exception for all embedding failures."""
    pass


class EmbedderModelLoadError(EmbedderError):
    """Raised when the sentence-transformer model fails to load."""
    pass


class EmbedderEncodingError(EmbedderError):
    """Raised when encoding a text or batch fails."""
    pass


# ─────────────────────────────────────────────
# EMBEDDER CLASS
# ─────────────────────────────────────────────

class Embedder:
    """
    Lazy-loading sentence-transformer embedding engine.

    The model is NOT initialized at class instantiation.
    It loads on the first call to encode(), encode_batch(),
    or encode_query() — this keeps startup time fast.

    Usage:
        from backend.retrieval.embedder import embedder

        # Single text
        vector = await embedder.encode_query("What is HDFC NIM?")

        # Batch encoding (for ingestion)
        vectors = embedder.encode_batch(["chunk1", "chunk2", "chunk3"])

        # Direct sync encoding
        vector = embedder.encode("Net Interest Margin definition")
    """

    def __init__(self):
        self._model = None
        self._model_name: str = EMBEDDING_MODEL
        self._device: str = DEVICE
        self._is_loaded: bool = False
        self._load_time_ms: float = 0.0
        self._total_encoded: int = 0
        self._total_batches: int = 0
        self._total_cache_hits: int = 0
        self._total_encode_time_ms: float = 0.0

    # ─────────────────────────────────────────
    # MODEL LOADING
    # ─────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Loads the sentence-transformer model. Called only once on first use.
        Thread-safe: if called concurrently, the model loads once.

        Raises:
            EmbedderModelLoadError if model cannot be loaded
        """
        if self._is_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"[EMBEDDER] Loading model: {self._model_name} | "
                f"device={self._device}"
            )
            load_start = time.perf_counter()

            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )

            # Warm up with a dummy encode to trigger JIT compilation
            _ = self._model.encode(
                ["warmup"],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            self._load_time_ms = (time.perf_counter() - load_start) * 1000
            self._is_loaded = True

            logger.info(
                f"[EMBEDDER] Model loaded successfully | "
                f"model={self._model_name} | "
                f"device={self._device} | "
                f"dim={EMBEDDING_DIMENSION} | "
                f"load_time={self._load_time_ms:.0f}ms"
            )

        except ImportError:
            raise EmbedderModelLoadError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbedderModelLoadError(
                f"Failed to load model '{self._model_name}': {str(e)}"
            ) from e

    def _ensure_loaded(self) -> None:
        """Ensures model is loaded before any encode call."""
        if not self._is_loaded:
            self._load_model()

    # ─────────────────────────────────────────
    # CORE: encode (sync, single text)
    # ─────────────────────────────────────────

    def encode(self, text: str) -> list[float]:
        """
        Encodes a single text string into a 384-dim L2-normalized vector.
        Synchronous — use encode_query() for async+cached version.

        Args:
            text: Input text string (will be truncated at 512 tokens by model)

        Returns:
            List of 384 floats (L2-normalized)

        Raises:
            EmbedderEncodingError on failure

        Examples:
            vector = embedder.encode("What is Net Interest Margin?")
            len(vector)  # → 384
        """
        self._ensure_loaded()

        if not text or not text.strip():
            raise EmbedderEncodingError(
                "Cannot encode empty string. Provide non-empty text."
            )

        try:
            encode_start = time.perf_counter()

            vector = self._model.encode(
                [text.strip()],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]

            duration_ms = (time.perf_counter() - encode_start) * 1000
            self._total_encoded += 1
            self._total_encode_time_ms += duration_ms

            result = _to_python_list(vector)

            logger.debug(
                f"[EMBEDDER] encode | "
                f"text_len={len(text)} | "
                f"dim={len(result)} | "
                f"{duration_ms:.1f}ms"
            )
            return result

        except EmbedderEncodingError:
            raise
        except Exception as e:
            raise EmbedderEncodingError(
                f"Failed to encode text: {str(e)}"
            ) from e

    # ─────────────────────────────────────────
    # CORE: encode_batch (sync, bulk ingestion)
    # ─────────────────────────────────────────

    def encode_batch(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Encodes a list of texts into vectors in batches.
        Designed for document ingestion — processes 32 texts at a time.

        Args:
            texts:          List of text strings to encode
            batch_size:     Override batch size (default: EMBEDDING_BATCH_SIZE=32)
            show_progress:  Show tqdm progress bar (useful for large ingestion jobs)

        Returns:
            List of 384-dim L2-normalized vectors, same order as input

        Raises:
            EmbedderEncodingError if any batch fails
            ValueError if texts list is empty

        Examples:
            chunks = ["NIM definition...", "CASA ratio explained...", ...]
            vectors = embedder.encode_batch(chunks)
            len(vectors)        # → same as len(chunks)
            len(vectors[0])     # → 384
        """
        if not texts:
            raise ValueError("encode_batch requires at least one text.")

        self._ensure_loaded()
        _batch_size = batch_size or EMBEDDING_BATCH_SIZE

        # Sanitize: strip and replace blanks with placeholder
        cleaned: list[str] = []
        blank_indices: list[int] = []
        for i, t in enumerate(texts):
            stripped = t.strip() if isinstance(t, str) else ""
            if not stripped:
                blank_indices.append(i)
                cleaned.append("empty document")     # placeholder, filtered after
            else:
                cleaned.append(stripped)

        if blank_indices:
            logger.warning(
                f"[EMBEDDER] encode_batch: {len(blank_indices)} blank text(s) "
                f"at indices {blank_indices[:5]} replaced with placeholder"
            )

        try:
            batch_start = time.perf_counter()

            vectors_np = self._model.encode(
                cleaned,
                batch_size=_batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

            duration_ms = (time.perf_counter() - batch_start) * 1000
            n = len(texts)
            self._total_encoded += n
            self._total_batches += 1
            self._total_encode_time_ms += duration_ms

            result = [_to_python_list(v) for v in vectors_np]

            # Validate output dimension
            if result and len(result[0]) != EMBEDDING_DIMENSION:
                raise EmbedderEncodingError(
                    f"Unexpected embedding dimension: "
                    f"got {len(result[0])}, expected {EMBEDDING_DIMENSION}"
                )

            throughput = n / (duration_ms / 1000) if duration_ms > 0 else 0
            logger.info(
                f"[EMBEDDER] encode_batch | "
                f"texts={n} | "
                f"batch_size={_batch_size} | "
                f"device={self._device} | "
                f"dim={EMBEDDING_DIMENSION} | "
                f"{duration_ms:.0f}ms | "
                f"{throughput:.0f} texts/sec"
            )
            log_metric("embedder_batch_duration_ms", duration_ms, unit="ms")
            log_metric("embedder_batch_size", n, unit="count")

            return result

        except EmbedderEncodingError:
            raise
        except Exception as e:
            raise EmbedderEncodingError(
                f"Batch encoding failed for {len(texts)} texts: {str(e)}"
            ) from e

    # ─────────────────────────────────────────
    # CORE: encode_query (async, cached)
    # ─────────────────────────────────────────

    async def encode_query(self, query: str) -> list[float]:
        """
        Async query encoder with Redis caching.
        Identical output to encode() but:
        1. Checks Redis cache first (avoids re-encoding identical queries)
        2. Stores result in Redis with CACHE_TTL_EMBEDDINGS TTL
        3. Logs separately from batch encode for monitoring

        Used exclusively by the retrieval pipeline (not ingestion).

        Args:
            query: User's financial query string

        Returns:
            384-dim L2-normalized vector

        Raises:
            EmbedderEncodingError on encoding failure

        Examples:
            vector = await embedder.encode_query("HDFC NIM Q3 FY26")
        """
        if not query or not query.strip():
            raise EmbedderEncodingError(
                "encode_query requires a non-empty query string."
            )

        query_clean = query.strip()

        # Step 1: Redis cache lookup
        cached_vector = await cache.get_embedding(query_clean)
        if cached_vector is not None:
            self._total_cache_hits += 1
            logger.debug(
                f"[EMBEDDER] encode_query CACHE HIT | "
                f"query='{query_clean[:50]}'"
            )
            return cached_vector

        # Step 2: Encode fresh
        encode_start = time.perf_counter()

        try:
            self._ensure_loaded()

            vector_np = self._model.encode(
                [query_clean],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]

            duration_ms = (time.perf_counter() - encode_start) * 1000
            self._total_encoded += 1
            self._total_encode_time_ms += duration_ms

            vector = _to_python_list(vector_np)

            logger.debug(
                f"[EMBEDDER] encode_query | "
                f"query='{query_clean[:60]}' | "
                f"dim={len(vector)} | "
                f"{duration_ms:.1f}ms"
            )
            log_metric("embedder_query_duration_ms", duration_ms, unit="ms")

        except EmbedderEncodingError:
            raise
        except Exception as e:
            raise EmbedderEncodingError(
                f"encode_query failed for '{query_clean[:80]}': {str(e)}"
            ) from e

        # Step 3: Cache result for future identical queries
        await cache.set_embedding(
            text=query_clean,
            vector=vector,
            ttl=CACHE_TTL_EMBEDDINGS,
        )

        return vector

    # ─────────────────────────────────────────
    # UTILITY: normalize (manual override)
    # ─────────────────────────────────────────

    @staticmethod
    def normalize(vector: list[float]) -> list[float]:
        """
        Applies L2 normalization to a vector.
        Model outputs are already normalized — use this only
        if you have external vectors that need normalization.

        Args:
            vector: Raw embedding vector

        Returns:
            L2-normalized vector (unit length)

        Examples:
            normalized = Embedder.normalize([0.5, 0.3, 0.8, ...])
        """
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            return vector
        return (arr / norm).tolist()

    # ─────────────────────────────────────────
    # UTILITY: similarity
    # ─────────────────────────────────────────

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """
        Computes cosine similarity between two L2-normalized vectors.
        For already-normalized vectors, this is simply the dot product.

        Args:
            vec_a: First embedding vector
            vec_b: Second embedding vector

        Returns:
            Similarity score from -1.0 to 1.0 (higher = more similar)
        """
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        return float(np.dot(a, b))

    # ─────────────────────────────────────────
    # UTILITY: warmup
    # ─────────────────────────────────────────

    def warmup(self) -> dict:
        """
        Explicitly loads model and runs a test encode.
        Call this at app startup to pre-load the model
        so the first real request isn't slow.

        Returns:
            Dict with load status and timing

        Examples:
            result = embedder.warmup()
            # {'status': 'ok', 'model': '...', 'device': 'cpu', 'load_ms': 1230}
        """
        try:
            start = time.perf_counter()
            self._load_model()
            test_vector = self.encode("Net Interest Margin financial metric warmup")
            total_ms = (time.perf_counter() - start) * 1000

            if len(test_vector) != EMBEDDING_DIMENSION:
                raise EmbedderError(
                    f"Warmup produced wrong dimension: {len(test_vector)}"
                )

            logger.info(
                f"[EMBEDDER] Warmup complete | "
                f"model={self._model_name} | "
                f"device={self._device} | "
                f"dim={EMBEDDING_DIMENSION} | "
                f"{total_ms:.0f}ms"
            )
            return {
                "status": "ok",
                "model": self._model_name,
                "device": self._device,
                "dimension": EMBEDDING_DIMENSION,
                "total_ms": round(total_ms, 1),
                "load_ms": round(self._load_time_ms, 1),
            }
        except Exception as e:
            logger.error(f"[EMBEDDER] Warmup failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model": self._model_name,
                "device": self._device,
            }

    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns embedding engine usage statistics."""
        avg_time = (
            round(self._total_encode_time_ms / self._total_encoded, 2)
            if self._total_encoded > 0 else 0.0
        )
        return {
            "model": self._model_name,
            "device": self._device,
            "is_loaded": self._is_loaded,
            "load_time_ms": round(self._load_time_ms, 1),
            "total_texts_encoded": self._total_encoded,
            "total_batches": self._total_batches,
            "total_cache_hits": self._total_cache_hits,
            "avg_encode_time_ms": avg_time,
            "total_encode_time_ms": round(self._total_encode_time_ms, 1),
            "dimension": EMBEDDING_DIMENSION,
        }

    def reset_stats(self) -> None:
        """Resets all counters. Used in testing."""
        self._total_encoded = 0
        self._total_batches = 0
        self._total_cache_hits = 0
        self._total_encode_time_ms = 0.0


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _to_python_list(arr: np.ndarray) -> list[float]:
    """
    Converts a numpy array to a native Python list of floats.
    Pinecone and Redis require plain Python types, not numpy types.

    Args:
        arr: numpy float array

    Returns:
        Python list of floats (cast from float32 → float64 for JSON safety)
    """
    return arr.astype(np.float64).tolist()


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# All pipeline modules import this directly
# ─────────────────────────────────────────────

embedder = Embedder()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "embedder",
    "Embedder",
    "EmbedderError",
    "EmbedderModelLoadError",
    "EmbedderEncodingError",
    "DEVICE",
]
