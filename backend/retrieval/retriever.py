"""
FinThesisGuard AI — Retrieval Orchestrator
ArkAngel Financial Solutions

Orchestrates the full retrieval pipeline:
    query → embed → Pinecone search → re-rank → return chunks

Used by every agent that needs document retrieval (Agents 3, 4, 5, 6, 7).
Applies a multi-signal re-ranking formula combining vector similarity,
source authority, document recency, and financial term overlap.

Re-ranking formula:
    final_score = (
        W_SIM       × vector_similarity_score  +
        W_AUTHORITY × authority_weight          +
        W_RECENCY   × recency_boost             +
        W_OVERLAP   × term_overlap_score
    )
"""

import asyncio
import re
import time
import math
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Tuple

from backend.retrieval.embedder import embedder, EmbedderEncodingError
from backend.retrieval.pinecone_client import (
    pinecone_client,
    PineconeClientError,
    SOURCE_TYPE_NAMESPACE,
    DEFAULT_NAMESPACE,
)
from backend.utils.financial_terms import FINANCIAL_TERMS
from backend.utils.logger import logger, log_metric
from backend.config import settings


# ─────────────────────────────────────────────
# RE-RANKING WEIGHTS
# Must sum to 1.0
# ─────────────────────────────────────────────

W_SIM:       float = 0.55    # Vector cosine similarity (primary signal)
W_AUTHORITY: float = 0.25    # Source authority weight
W_RECENCY:   float = 0.12    # Document freshness
W_OVERLAP:   float = 0.08    # Financial term overlap with query

assert abs(W_SIM + W_AUTHORITY + W_RECENCY + W_OVERLAP - 1.0) < 1e-9, \
    "Re-ranking weights must sum to 1.0"


# ─────────────────────────────────────────────
# AUTHORITY WEIGHTS PER SOURCE TYPE
# ─────────────────────────────────────────────

AUTHORITY_WEIGHTS: Dict[str, float] = {
    "rbi":                  1.00,
    "sebi":                 1.00,
    "mca":                  0.95,
    "tax_law":              0.90,
    "annual_report":        0.75,
    "earnings_transcript":  0.65,
    "broker_research":      0.40,
    "news":                 0.25,
    "blog":                 0.15,
    "unknown":              0.20,
}

_AUTHORITY_MAX: float = max(AUTHORITY_WEIGHTS.values())   # 1.0
_AUTHORITY_MIN: float = min(AUTHORITY_WEIGHTS.values())   # 0.15


# ─────────────────────────────────────────────
# RECENCY CONSTANTS
# ─────────────────────────────────────────────

RECENCY_FULL_DAYS:  int   = 180
RECENCY_DECAY_DAYS: int   = 730
RECENCY_FLOOR:      float = 0.05


# ─────────────────────────────────────────────
# RETRIEVAL DEFAULTS
# ─────────────────────────────────────────────

DEFAULT_TOP_K:         int   = 10
DEFAULT_MIN_SCORE:     float = 0.30
DEFAULT_MIN_AUTHORITY: float = 0.0
MAX_TOP_K:             int   = 20
PINECONE_OVERSAMPLE:   int   = 3


# ─────────────────────────────────────────────
# CUSTOM EXCEPTION
# ─────────────────────────────────────────────

class RetrieverError(Exception):
    """Base exception for retrieval pipeline failures."""

    def __init__(
        self,
        message: str,
        query: str = "",
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.query          = query
        self.original_error = original_error

    def __str__(self):
        return f"RetrieverError(query='{self.query[:60]}'): {super().__str__()}"


# ─────────────────────────────────────────────
# RETRIEVER CLASS
# ─────────────────────────────────────────────

class Retriever:
    """
    Full retrieval pipeline: embed → search → re-rank → return.

    Every agent calls retrieve() to get ranked document chunks.

    Public API
    ----------
    retrieve()                → single-query retrieval
    retrieve_multi()          → parallel multi-query retrieval (Agent 2 output)
    retrieve_thesis_context() → 4-angle thesis stress-test retrieval (Agent 6 input)
    retrieve_regulatory()     → regulatory-namespace fast-path
    retrieve_corporate()      → corporate-namespace with optional company filter
    retrieve_research()       → broker/news namespace
    retrieve_multi_source()   → all-namespaces parallel (Agent 4 conflict detection)

    Usage:
        from backend.retrieval.retriever import retriever

        # Single query
        chunks = await retriever.retrieve("HDFC Bank NIM Q3 FY26", top_k=10)

        # Multi sub-query (Agent 2 decomposition output)
        chunks = await retriever.retrieve_multi(
            queries=["HDFC NIM Q3 FY26", "ICICI NIM Q3 FY26"],
            top_k=6,
        )

        # Thesis stress-test context
        chunks = await retriever.retrieve_thesis_context(
            thesis="HDFC will outperform because NIM expansion...",
            top_k=12,
        )
    """

    def __init__(self):
        self._total_retrievals:       int   = 0
        self._total_chunks_returned:  int   = 0
        self._total_retrieve_time_ms: float = 0.0
        self._cache_hits:             int   = 0


    # ─────────────────────────────────────────
    # MAIN: retrieve (single query)
    # ─────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        min_score: float = DEFAULT_MIN_SCORE,
        min_authority: float = DEFAULT_MIN_AUTHORITY,
        search_all_namespaces: bool = False,
    ) -> List[dict]:
        """
        Full retrieval pipeline: embed → Pinecone search → re-rank → filter.

        Args:
            query:                 Financial query string (already acronym-expanded)
            top_k:                 Number of final ranked chunks to return (max 20)
            filters:               Pinecone metadata filters e.g. {"source_type": "rbi"}
            namespace:             Specific Pinecone namespace (None = auto fan-out to all)
            min_score:             Minimum final_score threshold (0.0–1.0)
            min_authority:         Minimum raw authority weight filter
            search_all_namespaces: If True, forces fan-out regardless of namespace arg

        Returns:
            List of chunk dicts sorted by final_score descending.
            Each chunk contains: id, text, source, source_type, date, company,
            vector_score, authority_weight, authority_score, recency_score,
            overlap_score, final_score.

        Raises:
            RetrieverError: On embed failure or Pinecone failure.

        Latency target: 400ms (embedding ~50ms + Pinecone ~200ms + rerank ~150ms)

        Namespace routing:
            namespace=None  → query_all_namespaces() [parallel fan-out, ~300ms]
            namespace="rbi" → query_vectors(namespace="rbi") [single ns, ~200ms]
        """
        if not query or not query.strip():
            raise RetrieverError("retrieve() requires a non-empty query.", query=query)

        top_k          = max(1, min(top_k, MAX_TOP_K))
        retrieve_start = time.perf_counter()
        query_clean    = query.strip()

        logger.info(
            f"[RETRIEVER] retrieve | "
            f"query='{query_clean[:80]}' | "
            f"top_k={top_k} | filters={filters} | namespace={namespace}"
        )

        # ── Step 1: Embed ─────────────────────────────────────────────────────
        try:
            query_vector = await embedder.encode_query(query_clean)
        except EmbedderEncodingError as e:
            raise RetrieverError(
                f"Query embedding failed: {e}",
                query=query_clean,
                original_error=e,
            )

        # ── Step 2: Pinecone search (oversample for re-ranking) ───────────────
        oversample_k = min(top_k * PINECONE_OVERSAMPLE, MAX_TOP_K * 2)

        # FIX: when no specific namespace is requested, always fan out to all
        # namespaces in parallel via query_all_namespaces(). Previously this only
        # triggered when search_all_namespaces=True was explicitly passed, causing
        # Agent6's internal retrieve() calls to fall into query_vectors() which
        # iterates namespaces sequentially (5 × ~270ms = ~1,400ms per call).
        # Now namespace=None → parallel fan-out → wall time = slowest single ns (~300ms).
        _fan_out = search_all_namespaces or (namespace is None)

        try:
            if _fan_out:
                raw_chunks = await pinecone_client.query_all_namespaces(
                    vector=query_vector,
                    top_k=oversample_k,
                    filters=filters,
                    min_score=0.0,
                )
            else:
                raw_chunks = await pinecone_client.query_vectors(
                    vector=query_vector,
                    top_k=oversample_k,
                    filters=filters,
                    namespace=namespace,
                    min_score=0.0,
                )
        except PineconeClientError as e:
            raise RetrieverError(
                f"Pinecone query failed: {e}",
                query=query_clean,
                original_error=e,
            )

        if not raw_chunks:
            duration_ms = (time.perf_counter() - retrieve_start) * 1000
            logger.warning(
                f"[RETRIEVER] No results from Pinecone | "
                f"query='{query_clean[:60]}' | {duration_ms:.0f}ms"
            )
            return []

        # ── Step 3: Re-rank ───────────────────────────────────────────────────
        ranked_chunks = self.apply_rerank_score(chunks=raw_chunks, query=query_clean)

        # ── Step 4: Authority filter ──────────────────────────────────────────
        if min_authority > 0.0:
            ranked_chunks = self.filter_by_authority(ranked_chunks, min_authority)

        # ── Step 5: Score threshold filter ───────────────────────────────────
        ranked_chunks = [
            c for c in ranked_chunks
            if c.get("final_score", 0.0) >= min_score
        ]

        # ── Step 6: Trim to top_k ─────────────────────────────────────────────
        final_chunks = ranked_chunks[:top_k]

        duration_ms = (time.perf_counter() - retrieve_start) * 1000
        self._total_retrievals       += 1
        self._total_chunks_returned  += len(final_chunks)
        self._total_retrieve_time_ms += duration_ms

        logger.info(
            f"[RETRIEVER] retrieve complete | "
            f"raw={len(raw_chunks)} | ranked={len(ranked_chunks)} | "
            f"returned={len(final_chunks)} | {duration_ms:.0f}ms"
        )
        log_metric("retriever_duration_ms",     duration_ms,       unit="ms")
        log_metric("retriever_chunks_returned", len(final_chunks), unit="count")

        return final_chunks


    # ─────────────────────────────────────────
    # retrieve_multi
    # ─────────────────────────────────────────

    async def retrieve_multi(
        self,
        queries: List[str],
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """
        Parallel retrieval for multiple sub-queries (Agent 2 decomposition output).

        Fires one retrieve() call per sub-query concurrently via asyncio.gather.
        Merges all results, deduplicates by chunk ID keeping the copy with the
        highest final_score, then returns a single flat list sorted descending.

        Args:
            queries:  List of sub-query strings (from Agent 2 decompose()).
                      Empty list returns [] immediately.
            top_k:    Chunks to retrieve per sub-query. Final merged list
                      may have up to len(queries) × top_k unique chunks
                      before global sort — callers typically slice downstream.
            filters:  Pinecone metadata filter dict applied to every sub-query
                      e.g. {"source_type": {"$in": ["rbi", "sebi"]}}

        Returns:
            Deduplicated, merged list[dict] sorted by final_score descending.
            Failed sub-query calls are logged and skipped — never crash the batch.

        Latency:
            ~400ms (parallel — dominated by slowest single retrieve())
        """
        if not queries:
            return []

        filters = filters or {}

        logger.info(
            f"[RETRIEVER] retrieve_multi | "
            f"sub_queries={len(queries)} | top_k_each={top_k} | filters={filters}"
        )

        t0 = time.perf_counter()

        tasks   = [self.retrieve(q, top_k=top_k, filters=filters) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen:   Dict[str, dict] = {}
        failed = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                logger.warning(
                    f"[RETRIEVER] retrieve_multi sub-query [{i}] failed: "
                    f"{type(result).__name__}: {result}"
                )
                continue

            for chunk in result:
                cid      = chunk.get("id") or chunk.get("text", "")[:120]
                existing = seen.get(cid)
                if (existing is None
                        or chunk.get("final_score", 0.0)
                           > existing.get("final_score", 0.0)):
                    seen[cid] = chunk

        merged = sorted(seen.values(), key=lambda c: -c.get("final_score", 0.0))

        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[RETRIEVER] retrieve_multi complete | "
            f"sub_queries={len(queries)} | failed={failed} | "
            f"unique_chunks={len(merged)} | {duration_ms:.0f}ms"
        )
        log_metric("retriever_multi_duration_ms",   duration_ms,  unit="ms")
        log_metric("retriever_multi_unique_chunks", len(merged),  unit="count")

        return merged


    # ─────────────────────────────────────────
    # retrieve_thesis_context
    # ─────────────────────────────────────────

    async def retrieve_thesis_context(
        self,
        thesis: str,
        top_k: int = 12,
    ) -> List[dict]:
        """
        Specialized 4-angle retrieval for investment thesis stress-testing.

        Generates four targeted sub-queries covering different evidence angles:
            1. Core claim       — direct semantic match to the thesis statement
            2. Historical analog — past market events analogous to the thesis
            3. Regulatory/macro — RBI/SEBI backdrop, macro policy evidence
            4. Valuation comps  — peer multiples and valuation benchmarks

        All 4 angles fire concurrently via asyncio.gather. Each angle calls
        retrieve() with namespace=None, which now fans out to query_all_namespaces()
        (parallel across all Pinecone namespaces). Wall time is dominated by the
        slowest single angle (~300ms), not the sum (~1,200ms).

        Args:
            thesis: Full investment thesis text (already acronym-expanded
                    by Agent 1 before this node is called).
                    Truncated to 400 chars for sub-query safety.
            top_k:  Total chunks to return across all 4 angles.
                    Minimum 4 (at least 1 per angle).

        Returns:
            Deduplicated list[dict] sorted by final_score descending,
            capped at top_k.

        Latency target: ~350ms (4 parallel × ~300ms fan-out each)
        """
        if not thesis or not thesis.strip():
            logger.warning("[RETRIEVER] retrieve_thesis_context called with empty thesis")
            return []

        top_k       = max(4, min(top_k, MAX_TOP_K))
        per_angle_k = max(1, top_k // 4)

        thesis_core  = thesis.strip()[:400]
        thesis_short = thesis.strip()[:180]

        sub_queries: List[Tuple[str, Optional[Dict[str, Any]]]] = [
            (
                thesis_core,
                None,
            ),
            (
                f"historical market analog case study: {thesis_short}",
                None,
            ),
            (
                f"RBI SEBI regulatory macro policy risk: {thesis_short}",
                {"source_type": {"$in": ["rbi", "sebi", "mca", "tax_law"]}},
            ),
            (
                f"valuation peer comparison multiple book value: {thesis_short}",
                None,
            ),
        ]

        logger.info(
            f"[RETRIEVER] retrieve_thesis_context | "
            f"thesis_len={len(thesis.strip())} | "
            f"top_k={top_k} | per_angle_k={per_angle_k}"
        )

        t0 = time.perf_counter()

        # All 4 angles fire concurrently.
        # Each retrieve() with namespace=None now calls query_all_namespaces()
        # (parallel namespace fan-out) instead of query_vectors() (sequential).
        tasks = [
            self.retrieve(
                query=q,
                top_k=per_angle_k,
                filters=f,
            )
            for q, f in sub_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen:   Dict[str, dict] = {}
        failed = 0
        angle_labels = ["core_claim", "historical_analog", "regulatory_macro", "valuation_comps"]

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                logger.warning(
                    f"[RETRIEVER] retrieve_thesis_context angle[{angle_labels[i]}] "
                    f"failed: {type(result).__name__}: {result}"
                )
                continue

            for chunk in result:
                cid      = chunk.get("id") or chunk.get("text", "")[:120]
                existing = seen.get(cid)
                if (existing is None
                        or chunk.get("final_score", 0.0)
                           > existing.get("final_score", 0.0)):
                    seen[cid] = chunk

        merged = sorted(seen.values(), key=lambda c: -c.get("final_score", 0.0))[:top_k]

        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[RETRIEVER] retrieve_thesis_context complete | "
            f"angles_failed={failed} | unique_chunks={len(merged)} | "
            f"{duration_ms:.0f}ms"
        )
        log_metric("retriever_thesis_duration_ms",   duration_ms,  unit="ms")
        log_metric("retriever_thesis_unique_chunks", len(merged),  unit="count")

        return merged


    # ─────────────────────────────────────────
    # RE-RANKING
    # ─────────────────────────────────────────

    def apply_rerank_score(
        self,
        chunks: List[dict],
        query: str,
    ) -> List[dict]:
        """
        Applies multi-signal re-ranking formula to raw Pinecone results.

        Formula:
            final_score = (
                W_SIM       × vector_similarity  +
                W_AUTHORITY × normalized_authority +
                W_RECENCY   × recency_boost       +
                W_OVERLAP   × term_overlap
            )

        Args:
            chunks: Raw Pinecone result dicts (must have 'score', 'source_type', 'date')
            query:  Original user query string

        Returns:
            Chunks enriched with scoring fields, sorted by final_score descending.
        """
        if not chunks:
            return []

        scored: List[dict] = []

        for chunk in chunks:
            vector_score         = float(chunk.get("score", 0.0))
            vector_score         = max(0.0, min(1.0, vector_score))

            source_type          = chunk.get("source_type", "unknown").lower()
            raw_authority        = get_authority_weight(source_type)
            normalized_authority = _normalize_authority(raw_authority)

            date_str = chunk.get("date", "")
            recency  = calculate_recency_boost(date_str)

            chunk_text = chunk.get("text", "")
            overlap    = calculate_term_overlap(query, chunk_text)

            final_score = (
                W_SIM       * vector_score         +
                W_AUTHORITY * normalized_authority  +
                W_RECENCY   * recency               +
                W_OVERLAP   * overlap
            )
            final_score = round(final_score, 6)

            enriched = {
                **chunk,
                "vector_score":     round(vector_score, 6),
                "authority_weight": raw_authority,
                "authority_score":  round(normalized_authority, 6),
                "recency_score":    round(recency, 6),
                "overlap_score":    round(overlap, 6),
                "final_score":      final_score,
            }
            scored.append(enriched)

        scored.sort(key=lambda x: x["final_score"], reverse=True)

        logger.debug(
            f"[RETRIEVER] reranked {len(scored)} chunks | "
            f"top={scored[0]['final_score']:.4f} | "
            f"bottom={scored[-1]['final_score']:.4f}"
        )
        return scored


    # ─────────────────────────────────────────
    # FILTER BY AUTHORITY
    # ─────────────────────────────────────────

    def filter_by_authority(
        self,
        chunks: List[dict],
        min_weight: float,
    ) -> List[dict]:
        """
        Filters out chunks from sources with authority weight below threshold.

        Args:
            chunks:     Re-ranked chunk list
            min_weight: Minimum raw authority weight (0.0–1.0)

        Returns:
            Filtered chunk list
        """
        if min_weight <= 0.0:
            return chunks

        filtered = [
            c for c in chunks
            if get_authority_weight(c.get("source_type", "unknown")) >= min_weight
        ]

        removed = len(chunks) - len(filtered)
        if removed > 0:
            logger.debug(
                f"[RETRIEVER] authority filter | "
                f"min_weight={min_weight} | removed={removed} | kept={len(filtered)}"
            )
        return filtered


    # ─────────────────────────────────────────
    # RETRIEVE FOR SPECIFIC SOURCE TYPES
    # ─────────────────────────────────────────

    async def retrieve_regulatory(
        self,
        query: str,
        top_k: int = 8,
    ) -> List[dict]:
        """
        Retrieves only from regulatory sources (RBI, SEBI, MCA, tax_law).
        Passes explicit namespace to bypass the fan-out path for speed.
        """
        return await self.retrieve(
            query=query,
            top_k=top_k,
            filters=None,
            namespace=SOURCE_TYPE_NAMESPACE["rbi"],   # explicit ns → single query
            min_authority=AUTHORITY_WEIGHTS["mca"],
        )

    async def retrieve_corporate(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = 8,
    ) -> List[dict]:
        """
        Retrieves from corporate sources (annual reports, earnings transcripts).
        Passes explicit namespace to bypass the fan-out path for speed.
        """
        filters = {}
        if company:
            filters["company"] = company
        return await self.retrieve(
            query=query,
            top_k=top_k,
            filters=filters or None,
            namespace=SOURCE_TYPE_NAMESPACE["annual_report"],  # explicit ns
        )

    async def retrieve_research(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[dict]:
        """
        Retrieves from broker research and news sources.
        Passes explicit namespace to bypass the fan-out path for speed.
        """
        return await self.retrieve(
            query=query,
            top_k=top_k,
            namespace=SOURCE_TYPE_NAMESPACE["broker_research"],  # explicit ns
            min_score=0.20,
        )

    async def retrieve_multi_source(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[dict]]:
        """
        Retrieves from all namespaces simultaneously and returns
        results grouped by namespace. Used by Agent 4 (Conflict Detector)
        to surface cross-source contradictions.

        Args:
            query:   Query string
            top_k:   Top-k per namespace
            filters: Optional metadata filters

        Returns:
            Dict of {namespace: [chunks]} for all namespaces
        """
        namespaces = list(set(SOURCE_TYPE_NAMESPACE.values()))

        async def _retrieve_ns(ns: str) -> Tuple[str, List[dict]]:
            try:
                chunks = await self.retrieve(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    namespace=ns,       # explicit ns → single query, no fan-out
                    min_score=0.15,
                )
                return ns, chunks
            except Exception as e:
                logger.warning(
                    f"[RETRIEVER] retrieve_multi_source ns={ns} failed: {e}"
                )
                return ns, []

        results = await asyncio.gather(*[_retrieve_ns(ns) for ns in namespaces])
        return {ns: chunks for ns, chunks in results}


    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns retrieval pipeline statistics."""
        avg_time = (
            round(self._total_retrieve_time_ms / self._total_retrievals, 1)
            if self._total_retrievals > 0 else 0.0
        )
        avg_chunks = (
            round(self._total_chunks_returned / self._total_retrievals, 1)
            if self._total_retrievals > 0 else 0.0
        )
        return {
            "total_retrievals":      self._total_retrievals,
            "total_chunks_returned": self._total_chunks_returned,
            "total_time_ms":         round(self._total_retrieve_time_ms, 1),
            "avg_retrieval_ms":      avg_time,
            "avg_chunks_per_query":  avg_chunks,
            "cache_hits":            self._cache_hits,
        }


# ─────────────────────────────────────────────
# STANDALONE SCORING FUNCTIONS
# ─────────────────────────────────────────────

def get_authority_weight(source_type: str) -> float:
    """
    Returns the raw authority weight for a source type.

    Args:
        source_type: Source type string (case-insensitive)

    Returns:
        Float between 0.15 (blog) and 1.0 (rbi/sebi)
    """
    return AUTHORITY_WEIGHTS.get(
        source_type.lower().strip(),
        AUTHORITY_WEIGHTS["unknown"],
    )


def calculate_recency_boost(date_str: str) -> float:
    """
    Calculates a recency boost score based on document publication date.
    Uses cosine decay for a smooth, natural falloff.

    Decay curve:
        0–180 days old → 1.00
        365 days old   → ~0.65
        730 days old   → 0.05 (RECENCY_FLOOR)
        unknown date   → 0.40

    Args:
        date_str: Date string in YYYY-MM-DD, YYYY-MM, or YYYY format.

    Returns:
        Float between RECENCY_FLOOR (0.05) and 1.0
    """
    if not date_str or date_str.lower() in {"unknown", "n/a", "none", ""}:
        return 0.40

    parsed_date: Optional[date] = _parse_date_flexible(date_str)
    if parsed_date is None:
        return 0.40

    today    = datetime.now(timezone.utc).date()
    days_old = (today - parsed_date).days

    if days_old < 0:
        return 1.0
    if days_old <= RECENCY_FULL_DAYS:
        return 1.0
    if days_old >= RECENCY_DECAY_DAYS:
        return RECENCY_FLOOR

    decay_range     = RECENCY_DECAY_DAYS - RECENCY_FULL_DAYS
    days_into_decay = days_old - RECENCY_FULL_DAYS
    decay_ratio     = days_into_decay / decay_range
    cosine_decay    = (1.0 + math.cos(math.pi * decay_ratio)) / 2.0
    recency         = RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * cosine_decay
    return round(recency, 6)


def calculate_term_overlap(query: str, chunk: str) -> float:
    """
    Calculates the fraction of recognized financial terms in the query
    that also appear in the chunk text.

    Args:
        query: User's financial query string
        chunk: Retrieved document chunk text

    Returns:
        Float from 0.0 (no overlap) to 1.0 (all query terms found in chunk)
    """
    if not query or not chunk:
        return 0.0

    query_upper = query.upper()
    chunk_upper = chunk.upper()
    query_terms = _extract_financial_terms(query_upper)

    if query_terms:
        chunk_terms = _extract_financial_terms(chunk_upper)
        matched     = query_terms & chunk_terms
        return round(min(len(matched) / len(query_terms), 1.0), 6)

    # Fallback: normalized word overlap
    query_words   = set(_tokenize(query.lower())) - _STOP_WORDS
    chunk_words   = set(_tokenize(chunk.lower()))
    if not query_words:
        return 0.0
    matched_words = query_words & chunk_words
    return round(min(len(matched_words) / len(query_words), 1.0), 6)


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _normalize_authority(raw_weight: float) -> float:
    """Min-max normalizes raw authority weight to 0.0–1.0."""
    if _AUTHORITY_MAX == _AUTHORITY_MIN:
        return 1.0
    normalized = (raw_weight - _AUTHORITY_MIN) / (_AUTHORITY_MAX - _AUTHORITY_MIN)
    return round(max(0.0, min(1.0, normalized)), 6)


def _parse_date_flexible(date_str: str) -> Optional[date]:
    """
    Parses date strings in multiple common formats.
    Supported: YYYY-MM-DD, YYYY-MM, YYYY, DD-MM-YYYY,
               MM/DD/YYYY, DD/MM/YYYY, Month YYYY, DD Month YYYY
    """
    if not date_str:
        return None

    date_str = date_str.strip()

    formats = [
        "%Y-%m-%d", "%Y-%m", "%Y",
        "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y",
        "%B %Y", "%b %Y", "%B %d, %Y", "%d %B %Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue

    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        try:
            return date(int(year_match.group()), 1, 1)
        except ValueError:
            pass

    logger.debug(f"[RETRIEVER] Could not parse date: '{date_str}'")
    return None


def _extract_financial_terms(text: str) -> set:
    """Extracts recognized financial acronyms from uppercase text."""
    tokens = re.findall(r'\b[A-Z][A-Z0-9/&_]{1,9}\b', text)
    return {t for t in tokens if t in FINANCIAL_TERMS}


def _tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase words, strips punctuation."""
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


_STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "is", "in", "on", "at", "to", "of", "and",
    "or", "for", "by", "with", "this", "that", "it", "its", "be",
    "are", "was", "were", "has", "have", "had", "will", "would",
    "could", "should", "may", "might", "shall", "do", "does", "did",
    "as", "from", "into", "through", "during", "before", "after",
    "above", "below", "between", "each", "more", "than", "such",
    "when", "where", "which", "who", "what", "how", "why", "if",
    "then", "than", "so", "but", "not", "no", "nor", "yet", "both",
    "also", "any", "all", "most", "other", "some", "these", "those",
})


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

retriever = Retriever()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "retriever",
    "Retriever",
    "get_authority_weight",
    "calculate_recency_boost",
    "calculate_term_overlap",
    "AUTHORITY_WEIGHTS",
    "W_SIM",
    "W_AUTHORITY",
    "W_RECENCY",
    "W_OVERLAP",
    "RetrieverError",
]
