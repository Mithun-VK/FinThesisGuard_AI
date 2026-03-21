# backend/retrieval/pinecone_client.py
"""
FinThesisGuard AI — Pinecone Vector Database Client
ArkAngel Financial Solutions

All Pinecone vector DB operations: index management, upsert,
query, delete, and namespace management. Used by the ingestion
pipeline (upsert) and retrieval pipeline (query).

Key design:
- Lazy initialization: client connects on first use
- Batch upsert: 100 vectors per request (Pinecone limit)
- Namespace isolation: separate namespaces per source type
- Retry: 3 attempts with exponential backoff on all operations
- Metadata filters: full support for source_type, company, date etc.
- Graceful error: raises PineconeClientError with full context
"""

import asyncio
import time
import uuid
from typing import Any, Optional
from dataclasses import dataclass, field

from pinecone import Pinecone, ServerlessSpec, PodSpec
from pinecone.exceptions import PineconeException

from backend.config import (
    settings,
    EMBEDDING_DIMENSION,
    LLM_RETRY_ATTEMPTS,
    LLM_RETRY_BACKOFF_BASE,
)
from backend.utils.logger import logger, log_metric


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

PINECONE_UPSERT_BATCH_SIZE: int  = 100     # Pinecone max per upsert call
PINECONE_QUERY_TIMEOUT: int      = 15      # seconds
PINECONE_METRIC: str             = "cosine"
PINECONE_CLOUD: str              = "aws"
PINECONE_REGION: str             = "us-east-1"

# Valid metadata filter keys — guards against injection
VALID_FILTER_KEYS: frozenset[str] = frozenset({
    "source_type",
    "company",
    "sector",
    "authority",
    "date",
    "date_year",
    "document_id",
    "namespace",
    "chunk_index",
    "doc_type",
})

# Default namespace when none specified
DEFAULT_NAMESPACE: str = "default"

# Namespace map: source_type → pinecone namespace
SOURCE_TYPE_NAMESPACE: dict[str, str] = {
    "rbi":                  "regulatory",
    "sebi":                 "regulatory",
    "mca":                  "regulatory",
    "tax_law":              "regulatory",
    "income_tax":           "regulatory",  # ← ADD THIS
    "irdai":                "regulatory",  # ← ADD THIS  
    "annual_report":        "corporate",
    "earnings_transcript":  "corporate",
    "broker_research":      "research",
    "news":                 "news",
    "blog":                 "research",
    "unknown":              "default",
}


# ─────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────

class PineconeClientError(Exception):
    """
    Base exception for all Pinecone operations.
    Carries structured context for debugging.
    """
    def __init__(
        self,
        message: str,
        operation: str = "unknown",
        attempt: int = 0,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.operation   = operation
        self.attempt     = attempt
        self.original_error = original_error

    def __str__(self):
        return (
            f"PineconeClientError(op={self.operation}, "
            f"attempt={self.attempt}): {super().__str__()}"
        )


class PineconeIndexNotFoundError(PineconeClientError):
    """Raised when querying a non-existent index."""
    pass


class PineconeUpsertError(PineconeClientError):
    """Raised when vector upsert fails."""
    pass


class PineconeQueryError(PineconeClientError):
    """Raised when vector query fails."""
    pass


class PineconeConnectionError(PineconeClientError):
    """Raised when initial connection/initialization fails."""
    pass


# ─────────────────────────────────────────────
# VECTOR DATACLASS
# ─────────────────────────────────────────────

@dataclass
class PineconeVector:
    """
    Structured representation of a single vector record.
    Use this when constructing vectors for upsert.

    Fields:
        id:        Unique string ID (auto-generated if not provided)
        values:    384-dim float list from embedder
        metadata:  Filterable metadata dict
        namespace: Target namespace (auto-derived from source_type if not set)
    """
    values: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    namespace: str = DEFAULT_NAMESPACE

    def __post_init__(self):
        if len(self.values) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Vector must have {EMBEDDING_DIMENSION} dimensions, "
                f"got {len(self.values)}"
            )

    def to_pinecone_dict(self) -> dict:
        """Converts to Pinecone upsert-ready format."""
        return {
            "id":       self.id,
            "values":   self.values,
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────
# QUERY RESULT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class QueryResult:
    """
    A single retrieved vector match from Pinecone.
    Used by the retrieval pipeline and re-ranker.
    """
    id: str
    score: float
    text: str
    source_type: str
    source: str
    date: str
    authority: float
    company: Optional[str]
    metadata: dict[str, Any]

    @classmethod
    def from_pinecone_match(cls, match: Any) -> "QueryResult":
        """Constructs QueryResult from raw Pinecone match object."""
        meta = match.metadata or {}
        return cls(
            id          = match.id,
            score       = round(float(match.score), 6),
            text        = meta.get("text", ""),
            source_type = meta.get("source_type", "unknown"),
            source      = meta.get("source", "Unknown Source"),
            date        = meta.get("date", "unknown"),
            authority   = float(meta.get("authority", 1.0)),
            company     = meta.get("company"),
            metadata    = meta,
        )

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "score":       self.score,
            "text":        self.text,
            "source_type": self.source_type,
            "source":      self.source,
            "date":        self.date,
            "authority":   self.authority,
            "company":     self.company,
            "metadata":    self.metadata,
        }


# ─────────────────────────────────────────────
# PINECONE CLIENT CLASS
# ─────────────────────────────────────────────

class PineconeClient:
    """
    Production Pinecone vector DB client for FinThesisGuard AI.

    Usage:
        from backend.retrieval.pinecone_client import pinecone_client

        # Initialize at startup
        await pinecone_client.initialize_index()

        # Upsert vectors (ingestion pipeline)
        await pinecone_client.upsert_vectors(vectors, namespace="regulatory")

        # Query vectors (retrieval pipeline)
        results = await pinecone_client.query_vectors(
            vector=query_embedding,
            top_k=10,
            filters={"source_type": "rbi"},
        )

        # Stats
        stats = await pinecone_client.get_index_stats()
    """

    def __init__(self):
        self._pc: Optional[Pinecone] = None
        self._index = None
        self._index_name: str = settings.pinecone_index
        self._is_initialized: bool = False
        self._total_upserted: int  = 0
        self._total_queried: int   = 0
        self._total_deleted: int   = 0
        self._failed_ops: int      = 0

    # ─────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────

    async def initialize_index(self) -> bool:
        """
        Connects to Pinecone and creates the index if it doesn't exist.
        Must be called once at application startup.

        Returns:
            True if index is ready, raises PineconeConnectionError on failure

        Raises:
            PineconeConnectionError
        """
        if self._is_initialized:
            return True

        return await self._retry(
            operation="initialize_index",
            fn=self._do_initialize,
        )

    async def _do_initialize(self) -> bool:
        """Inner logic for index initialization."""
        logger.info(
            f"[PINECONE] Initializing | "
            f"index={self._index_name} | "
            f"dim={EMBEDDING_DIMENSION}"
        )

        # Connect Pinecone client
        self._pc = Pinecone(api_key=settings.pinecone_api_key)

        # Check if index exists
        existing_indexes = [idx.name for idx in self._pc.list_indexes()]
        logger.info(f"[PINECONE] Existing indexes: {existing_indexes}")

        if self._index_name not in existing_indexes:
            logger.info(
                f"[PINECONE] Index '{self._index_name}' not found — creating..."
            )
            await self._create_index()
        else:
            logger.info(
                f"[PINECONE] Index '{self._index_name}' exists — connecting"
            )

        # Connect to the index
        self._index = self._pc.Index(self._index_name)

        # Verify connectivity with stats call
        stats = self._index.describe_index_stats()
        total_vectors = stats.total_vector_count
        logger.info(
            f"[PINECONE] Connected | "
            f"index={self._index_name} | "
            f"total_vectors={total_vectors} | "
            f"namespaces={list(stats.namespaces.keys())}"
        )
        self._is_initialized = True
        return True

    async def _create_index(self) -> None:
        """Creates Pinecone serverless index with cosine metric."""
        create_start = time.perf_counter()
        self._pc.create_index(
            name=self._index_name,
            dimension=EMBEDDING_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
        # Wait for index to become ready (up to 60 seconds)
        waited = 0
        while waited < 60:
            status = self._pc.describe_index(self._index_name).status
            if status.get("ready", False):
                break
            await asyncio.sleep(2)
            waited += 2

        duration_ms = (time.perf_counter() - create_start) * 1000
        logger.info(
            f"[PINECONE] Index created | "
            f"name={self._index_name} | "
            f"dim={EMBEDDING_DIMENSION} | "
            f"metric={PINECONE_METRIC} | "
            f"{duration_ms:.0f}ms"
        )

    def _ensure_initialized(self) -> None:
        """Guards all operations against uninitialized state."""
        if not self._is_initialized or self._index is None:
            raise PineconeClientError(
                message=(
                    "PineconeClient is not initialized. "
                    "Call await pinecone_client.initialize_index() at startup."
                ),
                operation="guard_check",
            )

    # ─────────────────────────────────────────
    # UPSERT VECTORS
    # ─────────────────────────────────────────

    async def upsert_vectors(
        self,
        vectors: list[dict],
        namespace: str = DEFAULT_NAMESPACE,
    ) -> int:
        """
        Upserts vectors into Pinecone in batches of 100.
        Each vector dict must have: id (str), values (list[float]), metadata (dict).

        Args:
            vectors:   List of vector dicts [{id, values, metadata}, ...]
            namespace: Pinecone namespace to write to

        Returns:
            Total number of vectors successfully upserted

        Raises:
            PineconeUpsertError on failure

        Examples:
            count = await pinecone_client.upsert_vectors(
                vectors=[
                    {
                        "id": "doc_001_chunk_0",
                        "values": [0.1, 0.2, ...],   # 384-dim
                        "metadata": {
                            "text":        "NIM is the net interest margin...",
                            "source":      "HDFC Bank Annual Report 2026",
                            "source_type": "annual_report",
                            "company":     "HDFC Bank",
                            "date":        "2026-01-15",
                            "authority":   1.3,
                        }
                    }
                ],
                namespace="corporate",
            )
        """
        self._ensure_initialized()

        if not vectors:
            logger.warning("[PINECONE] upsert_vectors called with empty list — skipped")
            return 0

        # Validate all vectors before sending
        _validate_vectors(vectors)

        total_upserted = await self._retry(
            operation="upsert_vectors",
            fn=lambda: self._do_upsert_batched(vectors, namespace),
        )
        return total_upserted

    async def _do_upsert_batched(
        self,
        vectors: list[dict],
        namespace: str,
    ) -> int:
        """Splits vectors into batches and upserts sequentially."""
        total = len(vectors)
        upserted = 0
        batches = _chunk_list(vectors, PINECONE_UPSERT_BATCH_SIZE)
        n_batches = len(batches)

        upsert_start = time.perf_counter()

        for batch_num, batch in enumerate(batches, 1):
            batch_start = time.perf_counter()

            # Convert to Pinecone format
            pinecone_records = [
                {
                    "id":       v["id"],
                    "values":   v["values"],
                    "metadata": v.get("metadata", {}),
                }
                for v in batch
            ]

            self._index.upsert(
                vectors=pinecone_records,
                namespace=namespace,
            )

            batch_ms = (time.perf_counter() - batch_start) * 1000
            upserted += len(batch)
            logger.debug(
                f"[PINECONE] Upsert batch {batch_num}/{n_batches} | "
                f"vectors={len(batch)} | "
                f"namespace={namespace} | "
                f"{batch_ms:.0f}ms"
            )

        total_ms = (time.perf_counter() - upsert_start) * 1000
        self._total_upserted += upserted

        logger.info(
            f"[PINECONE] Upsert complete | "
            f"total={upserted} | "
            f"batches={n_batches} | "
            f"namespace={namespace} | "
            f"{total_ms:.0f}ms"
        )
        log_metric("pinecone_upsert_count", upserted, unit="count")
        log_metric("pinecone_upsert_duration_ms", total_ms, unit="ms")

        return upserted

    # ─────────────────────────────────────────
    # UPSERT WITH NAMESPACE AUTO-ROUTING
    # ─────────────────────────────────────────

    async def upsert_namespace(
        self,
        vectors: list[dict],
        source_type: str,
    ) -> int:
        """
        Auto-routes vectors to the correct namespace based on source_type.
        Wrapper around upsert_vectors with automatic namespace derivation.

        Args:
            vectors:     List of vector dicts
            source_type: Document type (rbi, sebi, annual_report, etc.)

        Returns:
            Count of vectors upserted

        Examples:
            count = await pinecone_client.upsert_namespace(
                vectors=rbi_vectors,
                source_type="rbi",
            )
            # Automatically routes to "regulatory" namespace
        """
        namespace = SOURCE_TYPE_NAMESPACE.get(
            source_type.lower(), DEFAULT_NAMESPACE
        )
        logger.info(
            f"[PINECONE] upsert_namespace | "
            f"source_type={source_type} → namespace={namespace} | "
            f"vectors={len(vectors)}"
        )
        return await self.upsert_vectors(vectors, namespace=namespace)

    # ─────────────────────────────────────────
    # QUERY VECTORS
    # ─────────────────────────────────────────

    async def query_vectors(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Queries Pinecone for the most similar vectors.

        Args:
            vector:           Query embedding (384-dim)
            top_k:            Number of results to return (max 20)
            filters:          Metadata filters e.g. {"source_type": "rbi"}
            namespace:        Query a specific namespace (None = query all)
            include_metadata: Include metadata in results (default True)
            min_score:        Minimum similarity score threshold (0.0–1.0)

        Returns:
            List of result dicts sorted by descending relevance score:
            [
                {
                    "id": "doc_001_chunk_0",
                    "score": 0.921,
                    "text": "NIM is...",
                    "source_type": "annual_report",
                    "source": "HDFC Bank Annual Report 2026",
                    "date": "2026-01-15",
                    "authority": 1.3,
                    "company": "HDFC Bank",
                    "metadata": {...}
                },
                ...
            ]

        Raises:
            PineconeQueryError on failure

        Examples:
            results = await pinecone_client.query_vectors(
                vector=query_embedding,
                top_k=10,
                filters={"source_type": "rbi"},
            )
        """
        self._ensure_initialized()

        if len(vector) != EMBEDDING_DIMENSION:
            raise PineconeQueryError(
                message=(
                    f"Query vector must be {EMBEDDING_DIMENSION}-dim, "
                    f"got {len(vector)}"
                ),
                operation="query_vectors",
            )

        top_k = max(1, min(top_k, 20))

        # Validate filters
        validated_filters = _validate_filters(filters) if filters else None

        results = await self._retry(
            operation="query_vectors",
            fn=lambda: self._do_query(
                vector=vector,
                top_k=top_k,
                filters=validated_filters,
                namespace=namespace,
                include_metadata=include_metadata,
                min_score=min_score,
            ),
        )
        return results

    async def _do_query(
        self,
        vector: list[float],
        top_k: int,
        filters: Optional[dict],
        namespace: Optional[str],
        include_metadata: bool,
        min_score: float,
    ) -> list[dict]:
        """Executes the Pinecone query and parses results."""
        query_start = time.perf_counter()

        kwargs: dict[str, Any] = {
            "vector":           vector,
            "top_k":            top_k,
            "include_metadata": include_metadata,
            "include_values":   False,       # Don't return raw vectors (saves bandwidth)
        }
        if filters:
            kwargs["filter"] = filters
        if namespace:
            kwargs["namespace"] = namespace

        response = self._index.query(**kwargs)
        duration_ms = (time.perf_counter() - query_start) * 1000

        matches = response.matches or []

        # Parse and filter by min_score
        results = []
        for match in matches:
            if float(match.score) >= min_score:
                result = QueryResult.from_pinecone_match(match)
                results.append(result.to_dict())

        self._total_queried += 1
        logger.info(
            f"[PINECONE] Query | "
            f"top_k={top_k} | "
            f"matched={len(matches)} | "
            f"returned={len(results)} | "
            f"namespace={namespace or 'all'} | "
            f"filters={filters} | "
            f"{duration_ms:.0f}ms"
        )
        log_metric("pinecone_query_duration_ms", duration_ms, unit="ms")
        log_metric("pinecone_query_results", len(results), unit="count")

        return results

    # ─────────────────────────────────────────
    # MULTI-NAMESPACE QUERY
    # ─────────────────────────────────────────

    async def query_all_namespaces(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Queries across all known namespaces and merges results,
        re-sorted by score descending.

        Useful when source type is unknown or cross-namespace retrieval
        is needed (e.g., comparing regulatory vs corporate sources).

        Args:
            vector:    Query embedding
            top_k:     Total results to return across all namespaces
            filters:   Metadata filters
            min_score: Minimum score threshold

        Returns:
            Merged, sorted list of results
        """
        self._ensure_initialized()

        namespaces = list(set(SOURCE_TYPE_NAMESPACE.values())) + [DEFAULT_NAMESPACE]
        namespaces = list(set(namespaces))

        # Query each namespace concurrently
        tasks = [
            self.query_vectors(
                vector=vector,
                top_k=top_k,
                filters=filters,
                namespace=ns,
                min_score=min_score,
            )
            for ns in namespaces
        ]

        all_results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        merged: list[dict] = []
        seen_ids: set[str] = set()
        for result_list in all_results_nested:
            if isinstance(result_list, Exception):
                logger.warning(f"[PINECONE] Namespace query failed: {result_list}")
                continue
            for r in result_list:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    merged.append(r)

        # Sort by score descending and return top_k
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    # ─────────────────────────────────────────
    # DELETE VECTORS
    # ─────────────────────────────────────────

    async def delete_vectors(
        self,
        ids: list[str],
        namespace: str = DEFAULT_NAMESPACE,
    ) -> int:
        """
        Deletes vectors by ID from a namespace.

        Args:
            ids:       List of vector IDs to delete
            namespace: Namespace to delete from

        Returns:
            Count of IDs submitted for deletion

        Raises:
            PineconeClientError on failure

        Examples:
            deleted = await pinecone_client.delete_vectors(
                ids=["doc_001_chunk_0", "doc_001_chunk_1"],
                namespace="corporate",
            )
        """
        self._ensure_initialized()

        if not ids:
            logger.warning("[PINECONE] delete_vectors called with empty list — skipped")
            return 0

        await self._retry(
            operation="delete_vectors",
            fn=lambda: self._do_delete(ids, namespace),
        )
        return len(ids)

    async def _do_delete(self, ids: list[str], namespace: str) -> None:
        """Executes deletion in batches of 1000 (Pinecone limit)."""
        delete_start = time.perf_counter()
        batches = _chunk_list(ids, 1000)

        for batch in batches:
            self._index.delete(ids=batch, namespace=namespace)

        duration_ms = (time.perf_counter() - delete_start) * 1000
        self._total_deleted += len(ids)

        logger.info(
            f"[PINECONE] Delete | "
            f"ids={len(ids)} | "
            f"namespace={namespace} | "
            f"{duration_ms:.0f}ms"
        )

    async def delete_namespace(self, namespace: str) -> bool:
        """
        Deletes ALL vectors in a namespace (use with caution).

        Args:
            namespace: Namespace to flush

        Returns:
            True on success

        Examples:
            await pinecone_client.delete_namespace("regulatory")
        """
        self._ensure_initialized()

        try:
            delete_start = time.perf_counter()
            self._index.delete(delete_all=True, namespace=namespace)
            duration_ms = (time.perf_counter() - delete_start) * 1000
            logger.warning(
                f"[PINECONE] Namespace DELETED | "
                f"namespace={namespace} | "
                f"{duration_ms:.0f}ms"
            )
            return True
        except Exception as e:
            logger.error(
                f"[PINECONE] delete_namespace failed | "
                f"namespace={namespace} | {e}"
            )
            return False

    # ─────────────────────────────────────────
    # INDEX STATS
    # ─────────────────────────────────────────

    async def get_index_stats(self) -> dict:
        """
        Returns index statistics including total vector count
        and per-namespace breakdown.

        Returns:
            Dict with:
            - total_vector_count: int
            - namespaces: {namespace: {vector_count: int}}
            - dimension: int
            - index_fullness: float

        Raises:
            PineconeClientError on failure
        """
        self._ensure_initialized()

        return await self._retry(
            operation="get_index_stats",
            fn=self._do_get_stats,
        )

    async def _do_get_stats(self) -> dict:
        stats_start = time.perf_counter()
        stats = self._index.describe_index_stats()
        duration_ms = (time.perf_counter() - stats_start) * 1000

        namespaces_parsed = {
            ns: {"vector_count": data.vector_count}
            for ns, data in (stats.namespaces or {}).items()
        }

        result = {
            "total_vector_count": stats.total_vector_count,
            "dimension":          stats.dimension,
            "index_fullness":     round(float(stats.index_fullness or 0.0), 6),
            "namespaces":         namespaces_parsed,
            "index_name":         self._index_name,
        }

        logger.info(
            f"[PINECONE] Stats | "
            f"total_vectors={result['total_vector_count']} | "
            f"namespaces={list(namespaces_parsed.keys())} | "
            f"{duration_ms:.0f}ms"
        )
        return result

    # ─────────────────────────────────────────
    # FETCH VECTOR BY ID
    # ─────────────────────────────────────────

    async def fetch_vector(
        self,
        ids: list[str],
        namespace: str = DEFAULT_NAMESPACE,
    ) -> dict[str, dict]:
        """
        Fetches specific vectors by ID (including values and metadata).
        Useful for deduplication checks during ingestion.

        Args:
            ids:       List of vector IDs to fetch
            namespace: Namespace to fetch from

        Returns:
            Dict of {id: {values, metadata}} for found vectors
        """
        self._ensure_initialized()

        if not ids:
            return {}

        try:
            fetch_start = time.perf_counter()
            response = self._index.fetch(ids=ids, namespace=namespace)
            duration_ms = (time.perf_counter() - fetch_start) * 1000

            vectors = response.vectors or {}
            result = {
                vid: {
                    "values":   list(vdata.values or []),
                    "metadata": vdata.metadata or {},
                }
                for vid, vdata in vectors.items()
            }

            logger.debug(
                f"[PINECONE] Fetch | "
                f"requested={len(ids)} | "
                f"found={len(result)} | "
                f"namespace={namespace} | "
                f"{duration_ms:.0f}ms"
            )
            return result

        except Exception as e:
            logger.error(f"[PINECONE] fetch_vector failed: {e}")
            return {}

    # ─────────────────────────────────────────
    # STATS & MONITORING
    # ─────────────────────────────────────────

    def get_client_stats(self) -> dict:
        """Returns operational statistics for monitoring."""
        return {
            "is_initialized":    self._is_initialized,
            "index_name":        self._index_name,
            "total_upserted":    self._total_upserted,
            "total_queried":     self._total_queried,
            "total_deleted":     self._total_deleted,
            "failed_operations": self._failed_ops,
        }

    async def health_check(self) -> dict:
        """
        Validates Pinecone connectivity.
        Used by /health endpoint.
        """
        start = time.perf_counter()
        try:
            if not self._is_initialized:
                await self.initialize_index()
            stats = await self.get_index_stats()
            duration_ms = (time.perf_counter() - start) * 1000
            return {
                "status":       "ok",
                "index":        self._index_name,
                "total_vectors": stats["total_vector_count"],
                "latency_ms":   round(duration_ms, 1),
            }
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return {
                "status":     "error",
                "error":      str(e),
                "latency_ms": round(duration_ms, 1),
            }

    # ─────────────────────────────────────────
    # INTERNAL: RETRY LOGIC
    # ─────────────────────────────────────────

    async def _retry(
        self,
        operation: str,
        fn,
        max_attempts: int = LLM_RETRY_ATTEMPTS,
    ) -> Any:
        """
        Executes fn() with exponential backoff retry.
    Handles sync callables, async callables, AND lambdas
    wrapping async functions (iscoroutinefunction fails on lambdas).
        """
        last_exception: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                result = fn()
                # iscoroutinefunction() misses lambdas → check the object instead
                if asyncio.iscoroutine(result):
                    return await result
                return result

            except PineconeClientError:
                raise

            except PineconeException as e:
                last_exception = e
                wait = LLM_RETRY_BACKOFF_BASE ** (attempt - 1)
                self._failed_ops += 1
                logger.warning(
                    f"[PINECONE] {operation} failed | "
                    f"attempt={attempt}/{max_attempts} | "
                    f"error={type(e).__name__}: {e} | "
                    f"retry in {wait}s"
                )
                if attempt < max_attempts:
                    await asyncio.sleep(wait)

            except Exception as e:
                last_exception = e
                wait = LLM_RETRY_BACKOFF_BASE ** (attempt - 1)
                self._failed_ops += 1
                logger.error(
                    f"[PINECONE] {operation} unexpected error | "
                    f"attempt={attempt}/{max_attempts} | "
                    f"error={type(e).__name__}: {e} | "
                    f"retry in {wait}s"
                )
                if attempt < max_attempts:
                    await asyncio.sleep(wait)

        raise PineconeClientError(
            message=(
                f"Operation '{operation}' failed after "
                f"{max_attempts} attempts: {last_exception}"
            ),
            operation=operation,
            attempt=max_attempts,
            original_error=last_exception,
        )

# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _validate_vectors(vectors: list[dict]) -> None:
    """
    Validates vector list structure before upsert.
    Catches common ingestion pipeline errors early.
    """
    for i, v in enumerate(vectors):
        if "id" not in v or not v["id"]:
            raise PineconeUpsertError(
                f"Vector at index {i} missing 'id' field.",
                operation="validate_vectors",
            )
        if "values" not in v or not v["values"]:
            raise PineconeUpsertError(
                f"Vector at index {i} missing 'values' field.",
                operation="validate_vectors",
            )
        if len(v["values"]) != EMBEDDING_DIMENSION:
            raise PineconeUpsertError(
                f"Vector '{v['id']}' has wrong dimension: "
                f"expected {EMBEDDING_DIMENSION}, got {len(v['values'])}",
                operation="validate_vectors",
            )
        if "metadata" not in v:
            v["metadata"] = {}


def _validate_filters(filters: dict) -> dict:
    """
    Validates metadata filter keys against allowed set.
    Pinecone filter values can be strings, numbers, or lists.
    """
    validated = {}
    for key, value in filters.items():
        if key not in VALID_FILTER_KEYS:
            logger.warning(
                f"[PINECONE] Unknown filter key '{key}' — skipped. "
                f"Allowed: {sorted(VALID_FILTER_KEYS)}"
            )
            continue
        validated[key] = value
    return validated


def _chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Splits a list into chunks of given size."""
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

pinecone_client = PineconeClient()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "pinecone_client",
    "PineconeClient",
    "PineconeVector",
    "QueryResult",
    "PineconeClientError",
    "PineconeIndexNotFoundError",
    "PineconeUpsertError",
    "PineconeQueryError",
    "PineconeConnectionError",
    "SOURCE_TYPE_NAMESPACE",
    "DEFAULT_NAMESPACE",
]
