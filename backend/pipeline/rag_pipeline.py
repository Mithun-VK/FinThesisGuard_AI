# backend/pipeline/rag_pipeline.py
"""
FinThesisGuard AI — RAG Pipeline
ArkAngel Financial Solutions

Orchestrates the full financial Q&A pipeline across Agents 1–5 +
Groq synthesis. Accepts a raw financial query, coordinates every
processing stage, and returns a fully populated QueryResponse.

Pipeline stages and latency budget (target < 2.5 seconds):
    ┌────────────────────────────────────────────────────────────────────┐
    │ Stage 1  Agent 1 — Acronym expansion           sequential  ~50ms  │
    │ Stage 2  Agent 2 — Query decomposition          sequential ~200ms  │
    │ Stage 3  Retriever — Parallel sub-query fetch   parallel   ~400ms  │
    │ Stage 4  Agent 3 — Semantic re-ranking          sequential ~100ms  │
    │ Stage 5  Agent 4 + Agent 5 — Parallel analysis  parallel   ~800ms  │
    │           ├── Agent 4: Conflict detection                          │
    │           └── Agent 5: Quantitative validation                     │
    │ Stage 6  Groq — Answer synthesis                sequential ~600ms  │
    └────────────────────────────────────────────────────────────────────┘
    Total:  ~2,150ms  (well within 2,500ms hard target)

Graceful degradation:
    Every stage is individually try/except-wrapped. Any agent failure
    produces a warning log + default empty output. The pipeline never
    raises to the caller — it always returns a valid QueryResponse.

Debug mode:
    When settings.debug_mode=True, AgentTrace objects are attached to
    the response with per-stage latency, input/output summaries, and
    error details.

Depends on:
    backend/agents/agent1_acronym_resolver.py
    backend/agents/agent2_query_decomposer.py
    backend/agents/agent3_reranker.py
    backend/agents/agent4_conflict_detector.py
    backend/agents/agent5_quant_validator.py
    backend/retrieval/retriever.py
    backend/utils/llm_client.py
    backend/models/response_models.py
    backend/models/thesis_models.py
    backend/utils/logger.py
    backend/config.py
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.agents.agent1_acronym_resolver import acronym_resolver
from backend.agents.agent2_query_decomposer import query_decomposer
from backend.agents.agent3_reranker import reranker
from backend.agents.agent4_conflict_detector import conflict_detector
from backend.agents.agent5_quant_validator import quant_validator
from backend.config import (
    AGENT_LATENCY_TARGETS,
    AUTHORITY_WEIGHTS,
    LLM_MAX_TOKENS_SYNTHESIS,
    LLM_TEMPERATURE_FACTUAL,
    MAX_LATENCY_MS,
    MIN_SIMILARITY_THRESHOLD,
    TOP_K_RERANKED,
    TOP_K_RETRIEVAL,
)
from backend.models.response_models import (
    AgentTrace,
    Citation,
    Conflict,
    ConfidenceLevel,
    DataGap,
    QueryResponse,
)
from backend.models.thesis_models import QuantitativeFlag
from backend.retrieval.retriever import RetrieverError, retriever
from backend.utils.llm_client import (
    LLMError,
    build_rag_prompt,
    llm_client,
)
from backend.utils.logger import log_metric, logger
from backend.config import settings


# ── Pipeline identity ──────────────────────────────────────────────────────────
PIPELINE_NAME      = "rag_pipeline"
LATENCY_TARGET_MS  = MAX_LATENCY_MS  # 2,500ms from config

# ── Stage latency budgets (ms) ─────────────────────────────────────────────────
_STAGE_BUDGETS: dict[str, int] = {
    "agent1_acronym":    AGENT_LATENCY_TARGETS.get("agent1_acronym_resolver",   50),
    "agent2_decompose":  AGENT_LATENCY_TARGETS.get("agent2_query_decomposer",  200),
    "retrieval":         AGENT_LATENCY_TARGETS.get("retriever",                400),
    "agent3_rerank":     AGENT_LATENCY_TARGETS.get("agent3_reranker",          100),
    "agent4_conflicts":  AGENT_LATENCY_TARGETS.get("agent4_conflict_detector", 800),
    "agent5_quant":      AGENT_LATENCY_TARGETS.get("agent5_quant_validator",   300),
    "synthesis":         600,
}

# ── Minimum chunks required before attempting synthesis ───────────────────────
MIN_CHUNKS_FOR_SYNTHESIS = 2

# ── Maximum sub-queries from Agent 2 (cap to avoid excessive retrieval) ────────
# backend/pipeline/rag_pipeline.py
# Add this comment above MAX_SUB_QUERIES = 4
# NOTE: Agent 2 may generate up to 6 sub-queries (agent2_query_decomposer).
# The pipeline deliberately caps consumption at 4 to control retrieval
# latency. Remaining sub-queries are silently discarded after dedup.
MAX_SUB_QUERIES = 4  # Hard retrieval cap — Agent 2 cap is 6 (see agent2)

# ── Confidence thresholds ──────────────────────────────────────────────────────
_CONF_HIGH_MIN_CHUNKS      = 4
_CONF_HIGH_MIN_AVG_SCORE   = 0.60
_CONF_HIGH_MAX_CONFLICTS   = 0
_CONF_MEDIUM_MIN_CHUNKS    = 2
_CONF_MEDIUM_MIN_AVG_SCORE = 0.35
_CONF_LOW_MAX_CONFLICTS    = 3   # Major-conflict count that forces Low confidence

# ── Query type classification keywords ────────────────────────────────────────
_QUERY_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("REGULATORY", [
        "sebi", "rbi", "regulation", "circular", "guideline", "compliance",
        "norms", "lcr", "crr", "slr", "licence", "approved", "penalty",
    ]),
    ("THESIS",     [
        "buy", "sell", "hold", "thesis", "target price", "recommendation",
        "valuation", "re-rate", "upside", "downside", "pe", "p/b",
    ]),
    ("COMPARATIVE", [
        "compare", "vs", "versus", "better", "worse", "difference",
        "between", "both", "which", "contrast", "outperform",
    ]),
    ("FACTUAL",    []),  # Default fallback
]


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _PipelineState:
    """
    Carries mutable state through each pipeline stage.
    Populated incrementally as stages complete.
    """
    original_query:     str                    = ""
    expanded_query:     str                    = ""
    sub_queries:        list[str]              = field(default_factory=list)
    raw_chunks:         list[dict]             = field(default_factory=list)
    ranked_chunks:      list[dict]             = field(default_factory=list)
    conflicts:          list[Conflict]         = field(default_factory=list)
    quant_flags:        list[QuantitativeFlag] = field(default_factory=list)
    data_gaps:          list[str]              = field(default_factory=list)
    answer:             str                    = ""
    reasoning:          str                    = ""
    confidence:         ConfidenceLevel        = "Low"
    query_type:         str                    = "FACTUAL"
    agents_used:        list[str]              = field(default_factory=list)
    agent_traces:       list[AgentTrace]       = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Orchestrates the full 6-stage financial Q&A pipeline.

    Usage:
        from backend.pipeline.rag_pipeline import rag_pipeline

        response: QueryResponse = await rag_pipeline.run(
            query      = "What is HDFC Bank's NIM for Q3 FY26?",
            top_k      = 10,
            request_id = "req_abc123",
        )
    """

    def __init__(self) -> None:
        self._total_runs:      int   = 0
        self._total_ms:        float = 0.0
        self._cache_hits:      int   = 0
        self._agent_failures:  dict[str, int] = {
            "agent1": 0, "agent2": 0, "agent3": 0,
            "agent4": 0, "agent5": 0, "synthesis": 0,
        }
        logger.info(
            f"[RAG PIPELINE] Initialized | "
            f"latency_target={LATENCY_TARGET_MS}ms | "
            f"max_sub_queries={MAX_SUB_QUERIES} | "
            f"min_chunks_synthesis={MIN_CHUNKS_FOR_SYNTHESIS}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: main entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def run(
        self,
        query:      str,
        top_k:      int           = TOP_K_RETRIEVAL,
        filters:    Optional[dict] = None,
        request_id: Optional[str]  = None,
        cache_hit:  bool           = False,
    ) -> QueryResponse:
        """
        Runs the full 6-stage RAG pipeline for a financial query.

        Pipeline:
            Stage 1 → Agent 1: Expand acronyms (e.g. NIM → Net Interest Margin)
            Stage 2 → Agent 2: Decompose into focused sub-queries
            Stage 3 → Retriever: Parallel corpus fetch for all sub-queries
            Stage 4 → Agent 3: Semantic re-ranking of merged chunks
            Stage 5 → Parallel:
                        Agent 4: Detect cross-source conflicts
                        Agent 5: Validate quantitative claims
            Stage 6 → Groq: Synthesize cited answer from ranked evidence

        Args:
            query:      Raw financial question from the user.
            top_k:      Number of chunks to retrieve per sub-query.
            filters:    Optional Pinecone metadata filters
                        (e.g., {"source_type": "annual_report"}).
            request_id: Optional trace ID from the API layer.
            cache_hit:  True if caller resolved this from Redis cache.

        Returns:
            QueryResponse — fully assembled, always valid, never raises.
        """
        start_ms = time.perf_counter()
        self._total_runs += 1
        top_k = max(1, min(top_k, 20))

        query_stripped = query.strip() if query else ""

        logger.info(
            f"[RAG PIPELINE] Run start | "
            f"query='{query_stripped[:80]}' | top_k={top_k} | "
            f"request_id={request_id}"
        )

        # Guard: empty query
        if not query_stripped:
            return QueryResponse.error_response(
                error_message = "Query is empty.",
                request_id    = request_id,
                latency_ms    = 0,
            )

        state = _PipelineState(
            original_query = query_stripped,
            expanded_query = query_stripped,
        )
        state.query_type = self._classify_query(query_stripped)

        try:
            # ── Stage 1: Acronym expansion ────────────────────────────────────
            await self._stage_expand_acronyms(state)

            # ── Stage 2: Query decomposition ──────────────────────────────────
            await self._stage_decompose_query(state)

            # ── Stage 3: Parallel retrieval ───────────────────────────────────
            await self._stage_retrieve(state, top_k=top_k, filters=filters)

            # Early exit: no chunks retrieved
            if not state.raw_chunks:
                elapsed_ms = (time.perf_counter() - start_ms) * 1000
                self._total_ms += elapsed_ms
                logger.warning(
                    f"[RAG PIPELINE] No chunks retrieved for "
                    f"'{query_stripped[:60]}'. Returning insufficient-data response."
                )
                return QueryResponse.insufficient_data_response(
                    query      = query_stripped,
                    request_id = request_id,
                    latency_ms = int(elapsed_ms),
                )

            # ── Stage 4: Semantic re-ranking ──────────────────────────────────
            await self._stage_rerank(state, top_k=top_k)

            # ── Stage 5: Parallel — conflicts + quant validation ──────────────
            await self._stage_analyze_parallel(state)

            # ── Stage 6: Answer synthesis ─────────────────────────────────────
            await self._stage_synthesize(state)

            # ── Assemble final response ────────────────────────────────────────
            citations    = self._build_citations(state.ranked_chunks)
            elapsed_ms   = (time.perf_counter() - start_ms) * 1000
            self._total_ms += elapsed_ms

            response = QueryResponse(
                answer          = state.answer,
                citations       = citations,
                confidence      = state.confidence,
                reasoning       = state.reasoning,
                conflicts       = state.conflicts,
                data_gaps       = state.data_gaps,
                latency_ms      = int(elapsed_ms),
                agents_used     = state.agents_used,
                query_type      = state.query_type,
                cache_hit       = cache_hit,
                sub_queries     = state.sub_queries,
                agent_traces    = state.agent_traces if settings.debug_mode else [],
                request_id      = request_id,
            )

            logger.info(
                f"[RAG PIPELINE] Complete | "
                f"answer_len={len(state.answer)} | "
                f"chunks={len(state.ranked_chunks)} | "
                f"citations={len(citations)} | "
                f"conflicts={len(state.conflicts)} | "
                f"confidence={state.confidence} | "
                f"{elapsed_ms:.0f}ms"
            )
            log_metric("rag_pipeline_duration_ms",  elapsed_ms,              unit="ms")
            log_metric("rag_pipeline_chunks_used",  len(state.ranked_chunks), unit="count")
            log_metric("rag_pipeline_conflicts",    len(state.conflicts),     unit="count")

            if elapsed_ms > LATENCY_TARGET_MS:
                logger.warning(
                    f"[RAG PIPELINE] Latency target breached | "
                    f"{elapsed_ms:.0f}ms > {LATENCY_TARGET_MS}ms"
                )

            return response

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_ms) * 1000
            self._total_ms += elapsed_ms
            logger.error(
                f"[RAG PIPELINE] Unexpected error: {type(e).__name__}: {e} | "
                f"{elapsed_ms:.0f}ms"
            )
            return QueryResponse.error_response(
                error_message = f"{type(e).__name__}: {e}",
                request_id    = request_id,
                latency_ms    = int(elapsed_ms),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: synthesize_answer (also callable independently)
    # ──────────────────────────────────────────────────────────────────────────

    async def synthesize_answer(
        self,
        query:       str,
        chunks:      list[dict],
        conflicts:   list[Conflict]         = None,
        quant_flags: list[QuantitativeFlag] = None,
        query_type:  str                    = "FACTUAL",
    ) -> tuple[str, ConfidenceLevel, str]:
        """
        Synthesizes a cited answer from retrieved evidence using Groq.

        Strategy:
            1. Build a RAG prompt injecting top chunks as context.
            2. Append conflict notes so the model addresses contradictions.
            3. Append quantitative discrepancy flags.
            4. Call llm_client.call_factual() (T=0.1, strict factual mode).
            5. Calculate confidence from chunk quality + conflict severity.

        Citation format:
            The model is instructed to cite inline as [1], [2], [3]...
            matching the Citation.id in the returned QueryResponse.

        Args:
            query:       Expanded + original query string.
            chunks:      Semantically ranked corpus chunks.
            conflicts:   Detected cross-source conflicts (may be empty).
            quant_flags: Quantitative discrepancies detected by Agent 5.
            query_type:  Classified query type for prompt customisation.

        Returns:
            Tuple of (answer_text, confidence_level, reasoning_text).
            Returns a fallback triple on Groq failure (non-fatal).
        """
        conflicts   = conflicts   or []
        quant_flags = quant_flags or []

        if not chunks:
            return (
                "Insufficient corpus data to answer this query. "
                "Please ingest relevant financial documents first.",
                "Low",
                "No chunks retrieved from Pinecone above similarity threshold.",
            )

        # Build context-injected RAG prompt
        rag_prompt = build_rag_prompt(
            query         = query,
            context_chunks = chunks,
            max_context_chars = 7000,
        )

        # Extend with conflict context
        if conflicts:
            conflict_block = self._format_conflicts_for_prompt(conflicts)
            rag_prompt += (
                f"\n\nSOURCE CONFLICTS DETECTED — ADDRESS THESE IN YOUR ANSWER:\n"
                f"{conflict_block}\n"
                f"When conflicts exist, prefer the more authoritative or more recent source. "
                f"Explicitly note the discrepancy in your answer."
            )

        # Extend with quantitative flags
        if quant_flags:
            flag_block = "\n".join(
                f"  • {f.field}: thesis claims {f.thesis_claim}, "
                f"corpus shows {f.corpus_data} [{f.severity} discrepancy]"
                for f in quant_flags[:3]  # Cap at 3 to stay within token budget
            )
            rag_prompt += (
                f"\n\nQUANTITATIVE DISCREPANCIES — NOTE IN ANSWER IF RELEVANT:\n"
                f"{flag_block}"
            )

        # Query-type-specific instruction suffix
        type_suffix = self._query_type_suffix(query_type)
        if type_suffix:
            rag_prompt += f"\n\n{type_suffix}"

        try:
            answer = await llm_client.call_factual(
                prompt      = rag_prompt,
                agent_name  = PIPELINE_NAME,
                max_tokens  = LLM_MAX_TOKENS_SYNTHESIS,
            )
            answer = answer.strip()
        except Exception as e:
            self._agent_failures["synthesis"] += 1
            logger.warning(
                f"[RAG PIPELINE] synthesize_answer Groq failed: "
                f"{type(e).__name__}: {e}. Using chunk-based fallback."
            )
            answer    = self._fallback_answer(query, chunks)
            reasoning = f"Groq synthesis failed: {type(e).__name__}. Fallback from top chunk."
            confidence = self._calculate_confidence(chunks, conflicts, fallback=True)
            return answer, confidence, reasoning

        confidence = self._calculate_confidence(chunks, conflicts)
        reasoning  = self._build_reasoning(
            query        = query,
            chunks       = chunks,
            conflicts    = conflicts,
            quant_flags  = quant_flags,
            confidence   = confidence,
        )

        logger.debug(
            f"[RAG PIPELINE] synthesize_answer complete | "
            f"answer_len={len(answer)} | confidence={confidence} | "
            f"chunks={len(chunks)} | conflicts={len(conflicts)}"
        )
        return answer, confidence, reasoning

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Pipeline stages
    # ──────────────────────────────────────────────────────────────────────────

    async def _stage_expand_acronyms(self, state: _PipelineState) -> None:
        """
        Stage 1 — Agent 1: Expands financial acronyms in the query.

        Examples:
            "HDFC NIM Q3" → "HDFC Net Interest Margin Q3"
            "RBI MPC repo rate" → "Reserve Bank of India Monetary Policy Committee..."

        Failure is non-fatal: original query passes through unchanged.
        Target: 50ms.
        """
        stage_start = time.perf_counter()
        try:
            expanded = await acronym_resolver.expand(state.original_query)
            state.expanded_query = expanded if expanded else state.original_query
            state.agents_used.append("agent1_acronym_resolver")

            elapsed = (time.perf_counter() - stage_start) * 1000
            logger.debug(
                f"[RAG PIPELINE] Stage 1 (acronym) | "
                f"original='{state.original_query[:60]}' | "
                f"expanded='{state.expanded_query[:80]}' | "
                f"{elapsed:.0f}ms"
            )
            self._record_trace(
                state       = state,
                agent_name  = "agent1_acronym_resolver",
                elapsed_ms  = elapsed,
                success     = True,
                input_summary  = state.original_query[:100],
                output_summary = state.expanded_query[:100],
            )
            if elapsed > _STAGE_BUDGETS["agent1_acronym"]:
                logger.warning(
                    f"[RAG PIPELINE] Stage 1 breached budget: "
                    f"{elapsed:.0f}ms > {_STAGE_BUDGETS['agent1_acronym']}ms"
                )

        except Exception as e:
            elapsed = (time.perf_counter() - stage_start) * 1000
            self._agent_failures["agent1"] += 1
            logger.warning(
                f"[RAG PIPELINE] Stage 1 (acronym) failed: "
                f"{type(e).__name__}: {e}. Using original query."
            )
            self._record_trace(
                state       = state,
                agent_name  = "agent1_acronym_resolver",
                elapsed_ms  = elapsed,
                success     = False,
                error       = str(e),
            )

    async def _stage_decompose_query(self, state: _PipelineState) -> None:
        """
        Stage 2 — Agent 2: Decomposes complex query into focused sub-queries.

        Example:
            "Compare HDFC and ICICI NIM and asset quality for FY26"
            → ["HDFC Bank NIM Q3 FY26",
               "ICICI Bank NIM Q3 FY26",
               "HDFC Bank GNPA NPA FY26",
               "ICICI Bank GNPA NPA FY26"]

        Caps at MAX_SUB_QUERIES to prevent retrieval overload.
        Always includes the expanded query itself as the first sub-query.
        Failure: falls back to single-query [expanded_query].
        Target: 200ms.
        """
        stage_start = time.perf_counter()
        try:
            sub_queries = await query_decomposer.decompose(state.expanded_query)

            # Validate: must be non-empty list of strings
            if not sub_queries or not isinstance(sub_queries, list):
                raise ValueError(
                    f"decompose() returned invalid type: {type(sub_queries)}"
                )

            # Deduplicate + cap + always lead with full expanded query
            seen    = {state.expanded_query.lower()}
            cleaned = [state.expanded_query]
            for sq in sub_queries:
                sq_clean = sq.strip() if isinstance(sq, str) else ""
                if sq_clean and sq_clean.lower() not in seen and len(sq_clean) >= 5:
                    seen.add(sq_clean.lower())
                    cleaned.append(sq_clean)
                    if len(cleaned) >= MAX_SUB_QUERIES:
                        break

            state.sub_queries = cleaned
            state.agents_used.append("agent2_query_decomposer")

            elapsed = (time.perf_counter() - stage_start) * 1000
            logger.debug(
                f"[RAG PIPELINE] Stage 2 (decompose) | "
                f"sub_queries={len(state.sub_queries)} | "
                f"{state.sub_queries} | {elapsed:.0f}ms"
            )
            self._record_trace(
                state          = state,
                agent_name     = "agent2_query_decomposer",
                elapsed_ms     = elapsed,
                success        = True,
                input_summary  = state.expanded_query[:100],
                output_summary = f"{len(state.sub_queries)} sub-queries: "
                                 f"{state.sub_queries[:2]}",
            )

        except Exception as e:
            elapsed = (time.perf_counter() - stage_start) * 1000
            self._agent_failures["agent2"] += 1
            state.sub_queries = [state.expanded_query]  # Fallback: single query
            logger.warning(
                f"[RAG PIPELINE] Stage 2 (decompose) failed: "
                f"{type(e).__name__}: {e}. Using single query."
            )
            self._record_trace(
                state      = state,
                agent_name = "agent2_query_decomposer",
                elapsed_ms = elapsed,
                success    = False,
                error      = str(e),
            )

    async def _stage_retrieve(
        self,
        state:   _PipelineState,
        top_k:   int,
        filters: Optional[dict],
    ) -> None:
        """
        Stage 3 — Parallel retrieval for all sub-queries.

        Retrieves top-K chunks for each sub-query concurrently using
        asyncio.gather(). Merges and deduplicates all results by chunk ID.
        Re-ranks the merged set using the retriever's multi-signal formula.

        Deduplication key: chunk["id"] (Pinecone vector ID).
        On collision, keeps the chunk with the higher final_score.

        Failure: non-fatal — state.raw_chunks remains empty,
        pipeline returns insufficient_data_response.
        Target: 400ms.
        """
        stage_start = time.perf_counter()

        async def _retrieve_single(sub_query: str) -> list[dict]:
            try:
                return await retriever.retrieve(
                    query                = sub_query,
                    top_k                = top_k,
                    filters              = filters,
                    search_all_namespaces = True,
                    min_score            = MIN_SIMILARITY_THRESHOLD,
                )
            except RetrieverError as e:
                logger.warning(
                    f"[RAG PIPELINE] Retrieval failed for sub-query "
                    f"'{sub_query[:60]}': {e}"
                )
                return []

        try:
            results_per_query = await asyncio.gather(
                *[_retrieve_single(sq) for sq in state.sub_queries],
                return_exceptions=False,
            )

            # Merge + deduplicate by chunk id (keep highest score)
            merged: dict[str, dict] = {}
            for result_list in results_per_query:
                for chunk in result_list:
                    cid   = chunk.get("id") or chunk.get("chunk_id") or ""
                    score = float(chunk.get("final_score", 0.0))
                    if cid not in merged or score > float(
                        merged[cid].get("final_score", 0.0)
                    ):
                        merged[cid] = chunk

            # Sort merged set by final_score descending
            state.raw_chunks = sorted(
                merged.values(),
                key=lambda c: float(c.get("final_score", 0.0)),
                reverse=True,
            )

            elapsed = (time.perf_counter() - stage_start) * 1000
            logger.debug(
                f"[RAG PIPELINE] Stage 3 (retrieve) | "
                f"sub_queries={len(state.sub_queries)} | "
                f"total_raw={sum(len(r) for r in results_per_query)} | "
                f"after_dedup={len(state.raw_chunks)} | {elapsed:.0f}ms"
            )
            self._record_trace(
                state          = state,
                agent_name     = "retriever",
                elapsed_ms     = elapsed,
                success        = True,
                input_summary  = f"{len(state.sub_queries)} sub-queries",
                output_summary = f"{len(state.raw_chunks)} unique chunks",
            )

            # Register data gaps when retrieval is weak
            if len(state.raw_chunks) < MIN_CHUNKS_FOR_SYNTHESIS:
                state.data_gaps.append(
                    f"Only {len(state.raw_chunks)} chunk(s) found above "
                    f"similarity threshold {MIN_SIMILARITY_THRESHOLD}. "
                    f"Consider ingesting more relevant documents."
                )

        except Exception as e:
            elapsed = (time.perf_counter() - stage_start) * 1000
            logger.error(
                f"[RAG PIPELINE] Stage 3 (retrieve) failed: "
                f"{type(e).__name__}: {e}"
            )
            self._record_trace(
                state      = state,
                agent_name = "retriever",
                elapsed_ms = elapsed,
                success    = False,
                error      = str(e),
            )

    async def _stage_rerank(
        self,
        state: _PipelineState,
        top_k: int,
    ) -> None:
        """
        Stage 4 — Agent 3: Semantic re-ranking of the merged chunk set.

        Agent 3 performs cross-encoder scoring to move the most semantically
        relevant chunks to the top. This second-pass re-ranking is layered
        on top of the retriever's formula-based first pass.

        Caps output at top_k chunks to keep synthesis within token budget.
        Failure: falls back to retriever-ranked order (raw_chunks[:top_k]).
        Target: 100ms.
        """
        stage_start = time.perf_counter()
        try:
            reranked = await reranker.rerank(
                query   = state.expanded_query,
                chunks  = state.raw_chunks,
                top_k   = top_k,
            )
            state.ranked_chunks = reranked if reranked else state.raw_chunks[:top_k]
            state.agents_used.append("agent3_reranker")

            elapsed = (time.perf_counter() - stage_start) * 1000
            logger.debug(
                f"[RAG PIPELINE] Stage 4 (rerank) | "
                f"input={len(state.raw_chunks)} → output={len(state.ranked_chunks)} | "
                f"top_score={state.ranked_chunks[0].get('final_score', 0):.3f} | "
                f"{elapsed:.0f}ms"
            )
            self._record_trace(
                state          = state,
                agent_name     = "agent3_reranker",
                elapsed_ms     = elapsed,
                success        = True,
                input_summary  = f"{len(state.raw_chunks)} chunks",
                output_summary = f"{len(state.ranked_chunks)} ranked chunks",
            )

        except Exception as e:
            elapsed = (time.perf_counter() - stage_start) * 1000
            self._agent_failures["agent3"] += 1
            state.ranked_chunks = state.raw_chunks[:top_k]  # Fallback to formula ranking
            logger.warning(
                f"[RAG PIPELINE] Stage 4 (rerank) failed: "
                f"{type(e).__name__}: {e}. Using retriever-ranked order."
            )
            self._record_trace(
                state      = state,
                agent_name = "agent3_reranker",
                elapsed_ms = elapsed,
                success    = False,
                error      = str(e),
            )

    async def _stage_analyze_parallel(self, state: _PipelineState) -> None:
        """
        Stage 5 — Parallel: Agent 4 (conflict detection) + Agent 5 (quant validation).

        Both agents are completely independent of each other — run concurrently
        via asyncio.gather(). This is the most latency-sensitive parallel stage
        (Agent 4 can take up to 800ms for complex multi-source conflict analysis).

        Agent 4 — detect_conflicts():
            Identifies metric-level contradictions across sources.
            Example: Source A says NIM=4.2%, Source B says NIM=4.5%.

        Agent 5 — validate_quantities():
            Flags numerical inconsistencies between the query's claims
            and what the retrieved corpus actually shows.
            Example: Query implies "15% credit growth" but corpus shows 12%.

        Both failures are non-fatal — pipeline continues with empty lists.
        Target: max(800ms, 300ms) = 800ms (because they run in parallel).
        """
        stage_start = time.perf_counter()

        async def _run_conflict_detection() -> list[Conflict]:
            t = time.perf_counter()
            try:
                result = await conflict_detector.detect_conflicts(
                    chunks = state.ranked_chunks,
                    query  = state.expanded_query,
                )
                elapsed = (time.perf_counter() - t) * 1000
                state.agents_used.append("agent4_conflict_detector")
                self._record_trace(
                    state          = state,
                    agent_name     = "agent4_conflict_detector",
                    elapsed_ms     = elapsed,
                    success        = True,
                    input_summary  = f"{len(state.ranked_chunks)} chunks",
                    output_summary = f"{len(result)} conflicts",
                )
                return result or []
            except Exception as e:
                elapsed = (time.perf_counter() - t) * 1000
                self._agent_failures["agent4"] += 1
                logger.warning(
                    f"[RAG PIPELINE] Agent 4 (conflict detection) failed: "
                    f"{type(e).__name__}: {e}"
                )
                self._record_trace(
                    state      = state,
                    agent_name = "agent4_conflict_detector",
                    elapsed_ms = elapsed,
                    success    = False,
                    error      = str(e),
                )
                return []

        async def _run_quant_validation() -> list[QuantitativeFlag]:
            t = time.perf_counter()
            try:
                result = await quant_validator.validate(
                    query  = state.expanded_query,
                    chunks = state.ranked_chunks,
                )
                elapsed = (time.perf_counter() - t) * 1000
                state.agents_used.append("agent5_quant_validator")
                self._record_trace(
                    state          = state,
                    agent_name     = "agent5_quant_validator",
                    elapsed_ms     = elapsed,
                    success        = True,
                    input_summary  = f"{len(state.ranked_chunks)} chunks",
                    output_summary = f"{len(result)} flags",
                )
                return result or []
            except Exception as e:
                elapsed = (time.perf_counter() - t) * 1000
                self._agent_failures["agent5"] += 1
                logger.warning(
                    f"[RAG PIPELINE] Agent 5 (quant validation) failed: "
                    f"{type(e).__name__}: {e}"
                )
                self._record_trace(
                    state      = state,
                    agent_name = "agent5_quant_validator",
                    elapsed_ms = elapsed,
                    success    = False,
                    error      = str(e),
                )
                return []

        # Run both in parallel
        conflicts, quant_flags = await asyncio.gather(
            _run_conflict_detection(),
            _run_quant_validation(),
            return_exceptions=False,
        )

        state.conflicts   = conflicts
        state.quant_flags = quant_flags

        # Promote high-severity conflicts to data_gaps
        major_conflicts = [
            c for c in conflicts if c.severity == "Major"
        ]
        for mc in major_conflicts[:2]:
            state.data_gaps.append(
                f"Major conflict on '{mc.field}': "
                f"{mc.source_a} reports {mc.value_a}, "
                f"{mc.source_b} reports {mc.value_b}. "
                f"Recommended: {mc.recommended_source}."
            )

        elapsed = (time.perf_counter() - stage_start) * 1000
        logger.debug(
            f"[RAG PIPELINE] Stage 5 (parallel analysis) | "
            f"conflicts={len(state.conflicts)} | "
            f"quant_flags={len(state.quant_flags)} | "
            f"major_conflicts={len(major_conflicts)} | "
            f"{elapsed:.0f}ms"
        )

    async def _stage_synthesize(self, state: _PipelineState) -> None:
        """
        Stage 6 — Groq answer synthesis.

        Delegates to synthesize_answer() which:
            1. Injects ranked chunks as context (RAG prompt).
            2. Appends conflict + quant-flag context.
            3. Calls Groq (T=0.1, factual mode).
            4. Computes confidence from source quality + conflicts.

        Result populates state.answer, state.confidence, state.reasoning.
        Failure: state.answer is set to a fallback message.
        Target: 600ms.
        """
        stage_start = time.perf_counter()
        try:
            answer, confidence, reasoning = await self.synthesize_answer(
                query       = state.expanded_query,
                chunks      = state.ranked_chunks,
                conflicts   = state.conflicts,
                quant_flags = state.quant_flags,
                query_type  = state.query_type,
            )
            state.answer     = answer
            state.confidence = confidence
            state.reasoning  = reasoning
            state.agents_used.append(PIPELINE_NAME)

            elapsed = (time.perf_counter() - stage_start) * 1000
            self._record_trace(
                state          = state,
                agent_name     = "rag_synthesizer",
                elapsed_ms     = elapsed,
                success        = True,
                input_summary  = f"{len(state.ranked_chunks)} chunks, "
                                 f"{len(state.conflicts)} conflicts",
                output_summary = f"answer_len={len(answer)}, "
                                 f"confidence={confidence}",
            )
            if elapsed > _STAGE_BUDGETS["synthesis"]:
                logger.warning(
                    f"[RAG PIPELINE] Stage 6 (synthesis) breached budget: "
                    f"{elapsed:.0f}ms > {_STAGE_BUDGETS['synthesis']}ms"
                )

        except Exception as e:
            elapsed = (time.perf_counter() - stage_start) * 1000
            self._agent_failures["synthesis"] += 1
            state.answer     = self._fallback_answer(state.expanded_query, state.ranked_chunks)
            state.confidence = "Low"
            state.reasoning  = f"Synthesis failed: {type(e).__name__}: {e}"
            logger.error(
                f"[RAG PIPELINE] Stage 6 (synthesis) failed: "
                f"{type(e).__name__}: {e}"
            )
            self._record_trace(
                state      = state,
                agent_name = "rag_synthesizer",
                elapsed_ms = elapsed,
                success    = False,
                error      = str(e),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Confidence calculation
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_confidence(
        self,
        chunks:       list[dict],
        conflicts:    list[Conflict],
        fallback:     bool = False,
    ) -> ConfidenceLevel:
        """
        Derives ConfidenceLevel from chunk quality and conflict severity.

        Algorithm:
            1. Start at High.
            2. Compute avg_score and count high-authority chunks.
            3. Downgrade to Medium / Low based on thresholds.
            4. Major conflicts force Low regardless of chunk quality.

        Scoring rules:
            High:   chunks ≥ 4, avg_score ≥ 0.60, no major conflicts
            Low:    chunks < 2, avg_score < 0.35,
                    or ≥ 3 major conflicts, or fallback=True
            Medium: all other cases
        """
        if fallback or not chunks:
            return "Low"

        avg_score = sum(
            float(c.get("final_score", 0.0)) for c in chunks
        ) / len(chunks)

        high_authority_count = sum(
            1 for c in chunks
            if AUTHORITY_WEIGHTS.get(
                c.get("source_type", "unknown"), 0.2
            ) >= 0.65   # annual_report, rbi, sebi, mca, earnings_transcript
        )

        major_conflict_count = sum(
            1 for c in conflicts if c.severity == "Major"
        )

        # Hard Low conditions
        if (
            len(chunks) < _CONF_MEDIUM_MIN_CHUNKS
            or avg_score < _CONF_MEDIUM_MIN_AVG_SCORE
            or major_conflict_count >= _CONF_LOW_MAX_CONFLICTS
        ):
            return "Low"

        # High conditions
        if (
            len(chunks) >= _CONF_HIGH_MIN_CHUNKS
            and avg_score >= _CONF_HIGH_MIN_AVG_SCORE
            and major_conflict_count == _CONF_HIGH_MAX_CONFLICTS
            and high_authority_count >= 2
        ):
            return "High"

        return "Medium"

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Reasoning builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_reasoning(
        self,
        query:       str,
        chunks:      list[dict],
        conflicts:   list[Conflict],
        quant_flags: list[QuantitativeFlag],
        confidence:  ConfidenceLevel,
    ) -> str:
        """
        Builds a chain-of-thought reasoning string for the UI reasoning panel.

        Format:
            Step 1: Query expansion / classification
            Step 2: Retrieval summary
            Step 3: Source authority summary
            Step 4: Conflicts summary (if any)
            Step 5: Confidence derivation
        """
        avg_score = (
            round(sum(float(c.get("final_score", 0)) for c in chunks) / len(chunks), 3)
            if chunks else 0.0
        )
        source_summary = ", ".join(
            f"{c.get('source_type', 'unknown')} "
            f"({c.get('source', 'Unknown')[:40]})"
            for c in chunks[:3]
        )
        conflict_note = (
            f"{len(conflicts)} conflict(s) detected "
            f"({sum(1 for c in conflicts if c.severity == 'Major')} major). "
            if conflicts
            else "No cross-source conflicts detected. "
        )
        flag_note = (
            f"{len(quant_flags)} quantitative discrepancy flag(s). "
            if quant_flags else ""
        )

        return (
            f"Step 1 — Query processed: '{query[:80]}'. "
            f"Step 2 — Retrieved {len(chunks)} chunk(s) "
            f"(avg_score={avg_score}). "
            f"Step 3 — Top sources: {source_summary}. "
            f"Step 4 — {conflict_note}"
            f"{flag_note}"
            f"Step 5 — Confidence={confidence} based on "
            f"source authority, coverage, and conflict severity."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Citation builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_citations(self, chunks: list[dict]) -> list[Citation]:
        """
        Converts ranked chunks into numbered Citation objects.

        Citation IDs match the inline [1], [2]... markers the LLM is
        instructed to produce in the answer text.

        Deduplication: by (source, source_type) — keeps highest score.
        """
        seen:     dict[str, dict] = {}
        for chunk in chunks:
            source      = (chunk.get("source")      or "Unknown").strip()
            source_type = (
                chunk.get("source_type") or "unknown"
            ).lower().strip()
            key   = f"{source_type}::{source}"
            score = float(chunk.get("final_score", 0.0))
            if key not in seen or score > float(seen[key].get("final_score", 0.0)):
                seen[key] = chunk

        citations: list[Citation] = []
        for idx, chunk in enumerate(
            sorted(
                seen.values(),
                key=lambda c: float(c.get("final_score", 0.0)),
                reverse=True,
            ),
            start=1,
        ):
            source_type = (chunk.get("source_type") or "unknown").lower().strip()
            raw_weight  = AUTHORITY_WEIGHTS.get(source_type, 0.20)
            try:
                citations.append(Citation(
                    id              = idx,
                    title           = (
                        chunk.get("source")
                        or chunk.get("title")
                        or f"Source {idx}"
                    )[:500],
                    source_type     = source_type,
                    date            = (
                        chunk.get("date")
                        or chunk.get("metadata", {}).get("date", "unknown")
                    ),
                    authority_weight   = raw_weight,
                    relevance_score    = float(chunk.get("final_score", 0.0)),
                    url                = chunk.get("url"),
                    excerpt            = chunk.get("text", "")[:800],
                    company            = chunk.get("company") or chunk.get(
                        "metadata", {}
                    ).get("company"),
                    sector             = chunk.get("sector") or chunk.get(
                        "metadata", {}
                    ).get("sector"),
                ))
            except (ValueError, TypeError) as e:
                logger.debug(
                    f"[RAG PIPELINE] Skipping malformed citation: {e}"
                )
                continue

        return citations

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Query classification
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_query(query: str) -> str:
        """
        Classifies query into one of: REGULATORY, THESIS, COMPARATIVE, FACTUAL.

        Uses keyword matching against _QUERY_TYPE_PATTERNS.
        First pattern whose keywords match wins.
        FACTUAL is the default fallback (empty keyword list).

        Args:
            query: Raw or expanded query string.

        Returns:
            str — one of "REGULATORY", "THESIS", "COMPARATIVE", "FACTUAL".

        Examples:
            "What does SEBI say about F&O margins?"  → "REGULATORY"
            "Is HDFC Bank a buy at 2.5x book?"       → "THESIS"
            "Compare TCS vs Infosys EBITDA margins"  → "COMPARATIVE"
            "What is HDFC Bank's NIM for Q3 FY26?"   → "FACTUAL"
        """
        query_lower = query.lower()
        for query_type, keywords in _QUERY_TYPE_PATTERNS:
            if any(kw in query_lower for kw in keywords):
                return query_type
        return "FACTUAL"
# ─────────────────────────────────────────────
# HELPER: _build_citations
# ─────────────────────────────────────────────

    def _build_citations(
        self,
        chunks: List[dict],
    ) -> List[dict]:
        """
    Converts reranked chunk dicts into Citation-formatted dicts
    matching the QueryResponse / ThesisResponse schema.

    Each citation gets a sequential 1-based ID matching the inline
    [1], [2], [3] markers in the synthesized answer.

    Args:
        chunks: Reranked chunk dicts from _node_rerank or _node_retrieve.
                Expected keys: title, source, source_type, date, url,
                               authority_weight, final_score, text.

    Returns:
        List of Citation-compatible dicts, one per chunk, in rank order.
        Empty list if chunks is None or empty.

    Examples:
        citations = self.rag_pipeline._build_citations(reranked_chunks)
        # → [{"id": 1, "title": "HDFC Q3 FY26", "source_type": "annual_report", ...}]
        """
        if not chunks:
            return []

        citations = []
        for i, chunk in enumerate(chunks, start=1):
            # Extract a short excerpt (first 200 chars of chunk text)
            raw_text = chunk.get("text", "")
            excerpt  = raw_text[:200].strip()
            if len(raw_text) > 200:
                excerpt += "…"

            citations.append({
                "id"              : i,
                "title"           : chunk.get("title")  or chunk.get("source", f"Source {i}"),
                "source"          : chunk.get("source", ""),
                "source_type"     : chunk.get("source_type", "unknown"),
                "date"            : chunk.get("date", ""),
                "authority_weight": round(float(chunk.get("authority_weight", 0.20)), 3),
                "relevance_score" : round(float(chunk.get("final_score",      0.0)),  4),
                "url"             : chunk.get("url", ""),
                "excerpt"         : excerpt,
            })

        logger.debug(
            f"[RAG PIPELINE] _build_citations → {len(citations)} citations built"
        )
        return citations

# ─────────────────────────────────────────────
# HELPER: _infer_data_gaps
# ─────────────────────────────────────────────

    def _infer_data_gaps(
        self,
        chunks: List[dict],
        sub_queries: List[str],
    ) -> List[str]:
        """
    Identifies sub-queries that did not yield sufficiently relevant chunks.
    Returns human-readable gap strings surfaced in the QueryResponse.

    A sub-query is considered "unfulfilled" if:
        - No chunk in the result set contains any token from the sub-query
          (case-insensitive substring matching), OR
        - The best matching chunk has a final_score below MIN_SCORE_FOR_GAP.

    Args:
        chunks:      Reranked chunk list (may be empty on retrieval failure)
        sub_queries: Sub-queries generated by Agent 2 decompose()

    Returns:
        List of gap strings e.g.
            ["No data found for: ICICI GNPA Q3 FY26",
             "Insufficient evidence for: RBI rate cut timeline"]
        Returns [] if all sub-queries are covered or no sub_queries provided.

    Examples:
        gaps = self.rag_pipeline._infer_data_gaps(chunks, sub_queries)
        # → ["No data found for: peer NIM comparison Q3 FY26"]
        """
        MIN_SCORE_FOR_GAP: float = 0.30    # Chunks below this don't count as coverage

        if not sub_queries:
            return []

    # Build a flat lowercased corpus of all chunk text for quick token lookup
        chunk_corpus: List[str] = [
            (c.get("text", "") + " " + c.get("title", "") + " " + c.get("source", ""))
            .lower()    
            for c in chunks
            if c.get("final_score", 0.0) >= MIN_SCORE_FOR_GAP
        ]

        gaps: List[str] = []

        for query in sub_queries:
            query_lower  = query.lower().strip()
            query_tokens = [
                t for t in re.findall(r'\b[a-z]{3,}\b', query_lower)
                if t not in {
                    "the", "and", "for", "with", "what", "how", "are", "was",
                    "its", "this", "that", "from", "into", "about",
                }
            ]

            if not query_tokens:
                continue

        # A sub-query is "covered" if at least half its tokens appear in
        # at least one chunk (generous match to avoid false gap reports)
            required_matches = max(1, len(query_tokens) // 2)

            covered = any(
                sum(1 for t in query_tokens if t in corpus_text) >= required_matches
                for corpus_text in chunk_corpus
            )

            if not covered:
                # Shorten very long sub-queries for the gap message
                display_q = query if len(query) <= 80 else query[:77] + "…"
                gaps.append(f"No data found for: {display_q}")

        if gaps:
            logger.debug(
                f"[RAG PIPELINE] _infer_data_gaps → {len(gaps)} gap(s): {gaps}"
            )

        return gaps

# ─────────────────────────────────────────────
# HELPER: _build_reasoning_trace
# ─────────────────────────────────────────────

    def _build_reasoning_trace(
        self,
        state: dict,
    ) -> str:
        """
    Builds a human-readable reasoning trace from the pipeline's AgentState,
    summarizing what each node did and how long it took.

    Surfaced in QueryResponse.reasoning for transparency / explainability.
    Designed to be lightweight — pure string manipulation, no I/O.

    Args:
        state: Final AgentState dict after all nodes have run.
               Reads: query_type, sub_queries, chunks, reranked_chunks,
                      conflicts, quant_issues, confidence, node_timings, error.

    Returns:
        Multi-line reasoning string. Never raises — falls back to a
        minimal string on any unexpected key error.

    Examples:
        reasoning = self.rag_pipeline._build_reasoning_trace(final_state)
        # → "Query classified as COMPARATIVE | Decomposed into 4 sub-queries |
        #    Retrieved 8 chunks (3 sources) | Reranked: top score 0.921 |
        #    No conflicts detected | Confidence: High | Total: 342ms"
        """
        try:
            parts: List[str] = []

        # 1. Query type
            qtype = state.get("query_type", "UNKNOWN")
            parts.append(f"Query classified as {qtype}")

        # 2. Decomposition
            sub_queries = state.get("sub_queries", [])
            if len(sub_queries) > 1:
                parts.append(f"Decomposed into {len(sub_queries)} sub-queries")
            else:
                parts.append("Single-query retrieval (no decomposition needed)")

        # 3. Retrieval
            chunks = state.get("chunks", [])
            if chunks:
                source_names = list({
                    c.get("source_type", "unknown") for c in chunks
                })  
                parts.append(
                    f"Retrieved {len(chunks)} chunk(s) across "
                    f"{len(source_names)} source type(s): "
                    f"{', '.join(sorted(source_names))}"
                )
            else:
                parts.append("No chunks retrieved — possible retrieval failure")

        # 4. Reranking
            reranked = state.get("reranked_chunks", [])
            if reranked:
                top_score    = reranked[0].get("final_score", 0.0)
                bottom_score = reranked[-1].get("final_score", 0.0)
                parts.append(
                    f"Reranked {len(reranked)} chunk(s) | "
                    f"score range [{bottom_score:.3f} – {top_score:.3f}]"
                )

        # 5. Conflicts
            conflicts = state.get("conflicts", [])
            if conflicts:
                parts.append(
                    f"{len(conflicts)} conflict(s) detected and resolved "
                    f"(highest-authority source preferred)"
                )
            else:
                parts.append("No data conflicts detected across sources")

        # 6. Quant issues
            quant_issues = state.get("quant_issues", [])
            if quant_issues:
                parts.append(
                    f"{len(quant_issues)} quantitative warning(s): "
                    f"{'; '.join(str(q) for q in quant_issues[:2])}"
                    + (" …" if len(quant_issues) > 2 else "")
                )

        # 7. Confidence
            confidence = state.get("confidence", "Low")
            parts.append(f"Confidence: {confidence}")

        # 8. Error (if any)
            error = state.get("error")
            if error:
                parts.append(f"Pipeline error (degraded response): {error}")

        # 9. Total latency from node_timings
            timings = state.get("node_timings", {})
            if timings:
                total_ms = sum(timings.values())
                slowest_node = max(timings, key=timings.get)
                parts.append(
                    f"Total pipeline: {total_ms:.0f}ms | "
                    f"slowest node: {slowest_node} ({timings[slowest_node]:.0f}ms)"
                )

            return " | ".join(parts)

        except Exception as exc:
            logger.warning(f"[RAG PIPELINE] _build_reasoning_trace failed: {exc}")
            return f"Reasoning trace unavailable: {type(exc).__name__}"

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Prompt helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_conflicts_for_prompt(conflicts: list[Conflict]) -> str:
        """
        Formats detected conflicts into a compact block for prompt injection.

        Example output:
            - NIM: HDFC Q3 FY26 Annual Report says 4.2%
              vs Kotak Research Note says 4.5%
              → Prefer: HDFC Q3 FY26 Annual Report (more authoritative/recent)

        Capped at 3 conflicts to stay within token budget.
        """
        lines = []
        for cf in conflicts[:3]:
            lines.append(
                f"  • {cf.field}: "
                f"'{cf.source_a}' says {cf.value_a} "
                f"vs '{cf.source_b}' says {cf.value_b}. "
                f"[{cf.severity}] → Prefer: {cf.recommended_source}. "
                f"Reason: {cf.reason[:120]}"
            )
        return "\n".join(lines)

    @staticmethod
    def _query_type_suffix(query_type: str) -> str:
        """
        Returns a query-type-specific instruction appended to the RAG prompt.
        Tailors Groq's answer style to the nature of the question.

        Args:
            query_type: One of REGULATORY, THESIS, COMPARATIVE, FACTUAL.

        Returns:
            str — instruction suffix, or "" if no special handling needed.
        """
        _suffixes = {
            "REGULATORY": (
                "REGULATORY QUERY INSTRUCTIONS: "
                "Cite the exact RBI/SEBI circular or guideline number if available. "
                "Distinguish between draft guidelines and final circulars. "
                "If the regulation has changed, report the most recent version first."
            ),
            "THESIS": (
                "THESIS QUERY INSTRUCTIONS: "
                "Present the bull and bear case clearly. "
                "Cite valuation multiples, earnings estimates, and risk factors "
                "explicitly. Do not make a buy/sell recommendation — present evidence."
            ),
            "COMPARATIVE": (
                "COMPARATIVE QUERY INSTRUCTIONS: "
                "Use a structured comparison. Present metric-by-metric differences. "
                "If sources disagree on the same metric for the same entity, "
                "flag it explicitly."
            ),
            "FACTUAL": (
                "FACTUAL QUERY INSTRUCTIONS: "
                "Lead with the direct answer. Cite the exact figure and source. "
                "If the figure is for a different period than asked, note the difference."
            ),
        }
        return _suffixes.get(query_type, "")

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Fallback answer
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_answer(query: str, chunks: list[dict]) -> str:
        """
        Produces a best-effort answer from the top chunk when Groq is
        unavailable. Used only when synthesize_answer() fails.

        Strategy:
            - Return the text of the highest-scoring chunk with its source.
            - Prepend a disclaimer that the answer is unprocessed excerpt.

        Args:
            query:  The user's query (for context in the disclaimer).
            chunks: Ranked chunk list.

        Returns:
            str — fallback answer, always non-empty.
        """
        if not chunks:
            return (
                f"Unable to answer '{query[:80]}'. "
                f"No relevant documents found in the corpus."
            )

        top = chunks[0]
        source   = top.get("source",      "Unknown Source")
        excerpt  = top.get("text",        "")[:500]
        score    = float(top.get("final_score", 0.0))

        return (
            f"[Synthesis unavailable — showing top retrieved excerpt] "
            f"From {source} (relevance={score:.2f}):\n\n"
            f"{excerpt}\n\n"
            f"Note: Groq synthesis was unavailable. "
            f"This is a raw excerpt, not a synthesized answer."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Debug trace recorder
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _record_trace(
        state:          _PipelineState,
        agent_name:     str,
        elapsed_ms:     float,
        success:        bool,
        input_summary:  Optional[str] = None,
        output_summary: Optional[str] = None,
        error:          Optional[str] = None,
    ) -> None:
        """
        Appends an AgentTrace to state.agent_traces.
        Only used when settings.debug_mode=True — traces are omitted
        from production QueryResponse to reduce payload size.

        Args:
            state:          Current pipeline state (mutated in place).
            agent_name:     Name of the agent/stage being traced.
            elapsed_ms:     Time taken by this stage in milliseconds.
            success:        True if stage completed without exception.
            input_summary:  Short description of stage input (optional).
            output_summary: Short description of stage output (optional).
            error:          Error message if success=False (optional).
        """
        if not settings.debug_mode:
            return
        state.agent_traces.append(AgentTrace(
            agent_name     = agent_name,
            duration_ms    = round(elapsed_ms, 1),
            success        = success,
            input_summary  = input_summary,
            output_summary = output_summary,
            error          = error,
        ))

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Statistics
    # ──────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Returns runtime statistics for monitoring and /metrics endpoint.

        Returns:
            dict with keys:
                total_runs:          int   — total pipeline executions
                total_ms:            float — cumulative latency (ms)
                avg_ms_per_run:      float — mean latency per run
                agent_failures:      dict  — per-agent failure counts
                latency_target_ms:   int   — configured SLA target
                max_sub_queries:     int   — configured sub-query cap
        """
        return {
            "total_runs":        self._total_runs,
            "total_ms":          round(self._total_ms, 1),
            "avg_ms_per_run":    round(
                self._total_ms / self._total_runs, 1
            ) if self._total_runs else 0.0,
            "cache_hits":        self._cache_hits,
            "agent_failures":    dict(self._agent_failures),
            "latency_target_ms": LATENCY_TARGET_MS,
            "max_sub_queries":   MAX_SUB_QUERIES,
        }

    def record_cache_hit(self) -> None:
        """Increments the cache hit counter. Called by the API layer."""
        self._cache_hits += 1


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON EXPORT
# ══════════════════════════════════════════════════════════════════════════════

rag_pipeline = RAGPipeline()

__all__ = [
    "rag_pipeline",
    "RAGPipeline",
    "PIPELINE_NAME",
    "LATENCY_TARGET_MS",
    "_PipelineState",
]
