"""
pipeline/orchestrator.py
========================
Production-grade LangGraph orchestrator for FinThesisGuard AI.
Routes every incoming query through the correct pipeline with full
caching, observability, error handling, and circuit-breaker protection.

Dependency order (build after):
  config.py → utils/* → models/* → retrieval/* → agents/* → pipelines → THIS

Fixed in this version:
  - _initialized attribute consistent throughout (no .initialized mismatch)
  - Duplicate retrieval_filters key removed from initial_state
  - synthesize_answer 3-tuple unpacking (answer, confidence, reasoning)
  - agent2.is_complex_query / agent3.rerank handle sync and async safely
  - orchestrator.health() uses direct singleton imports, not rag_pipeline attrs
  - Caller-supplied filters preserved through _node_classify (merge, not overwrite)
  - shutdown() uses direct llm_client import, not self.rag_pipeline.llm_client
  - _safe_call() helper eliminates all 'object X can't be used in await' warnings
  - FIX #1: _node_stress_synthesize passes thesis_text= to agent7.synthesize()
  - FIX #2: All error fallbacks use 'Weak' not 'Unknown' for thesis_strength
  - FIX #3: _node_build_thesis_response coerces thesis_strength to valid Literal
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
import traceback
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from backend.models.query_models import QueryType
from backend.models.response_models import QueryResponse
from backend.models.thesis_models import ThesisResponse
from backend.pipeline.rag_pipeline import RAGPipeline
from backend.pipeline.thesis_pipeline import ThesisPipeline
from backend.utils.cache import cache
from backend.utils.logger import log_query, logger
from backend.config import settings

# ── Agent singletons — imported directly (not via pipeline attributes) ────────
from backend.agents.agent1_acronym_resolver   import acronym_resolver
from backend.agents.agent2_query_decomposer   import query_decomposer
from backend.agents.agent3_reranker           import reranker
from backend.agents.agent4_conflict_detector  import conflict_detector
from backend.agents.agent5_quant_validator    import quant_validator
from backend.agents.agent6_thesis_analyzer    import thesis_analyzer
from backend.agents.agent7_stress_synthesizer import stress_synthesizer
from backend.retrieval.retriever              import retriever as corpus_retriever

# ── FIX #3: valid Literal values for ThesisResponse.thesis_strength ──────────
# ThesisResponse.thesis_strength is Literal["Strong", "Medium", "Weak"].
# "Unknown" is NOT valid and causes a Pydantic ValidationError.
# Always coerce through this set before constructing ThesisResponse.
_VALID_STRENGTHS = {"Strong", "Medium", "Weak"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — async-safe call wrapper
# ─────────────────────────────────────────────────────────────────────────────

async def _safe_call(method, *args, **kwargs):
    """
    Calls a method that may be sync or async transparently.
    Awaits coroutines; returns plain values directly.
    Eliminates 'object bool/list can't be used in await expression' warnings
    that occur when agent methods are defined as plain def instead of async def.
    """
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _coerce_strength(value: Any) -> str:
    """
    Coerce any thesis_strength value to a valid Literal["Strong", "Medium", "Weak"].
    Falls back to "Weak" for None, "Unknown", or any unrecognised string.
    This prevents Pydantic ValidationError when Agent 7 fails or returns
    a non-standard strength label.
    """
    if value in _VALID_STRENGTHS:
        return value
    # Best-effort mapping of common non-standard values
    if isinstance(value, str):
        v = value.strip().title()
        if v in _VALID_STRENGTHS:
            return v
        if v in ("High", "Very Strong"):
            return "Strong"
        if v in ("Low", "Very Weak", "Unknown", "N/A", ""):
            return "Weak"
    return "Weak"


# ─────────────────────────────────────────────────────────────────────────────
# 1. STATE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    request_id:        str
    query:             str
    query_type:        str
    is_thesis:         bool

    # ── Pre-processing ────────────────────────────────────────────────────────
    expanded_query:    str
    sub_queries:       List[str]
    retrieval_filters: Dict[str, Any]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    chunks:            List[dict]
    reranked_chunks:   List[dict]

    # ── Validation ────────────────────────────────────────────────────────────
    conflicts:         List[dict]
    quant_issues:      List[str]

    # ── Synthesis ─────────────────────────────────────────────────────────────
    answer:            str
    thesis_data:       dict

    # ── Output ────────────────────────────────────────────────────────────────
    response:          dict
    confidence:        str

    # ── Observability ─────────────────────────────────────────────────────────
    node_timings:      Dict[str, float]
    total_latency_ms:  float
    error:             Optional[str]
    cache_hit:         bool


# ─────────────────────────────────────────────────────────────────────────────
# 2. CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Protects downstream services (Groq, Pinecone, Redis) from cascading
    failures. Configurable threshold, recovery timeout, and half-open probe.
    """

    def __init__(
        self,
        name:                str,
        failure_threshold:   int   = 5,
        recovery_timeout:    float = 30.0,
        half_open_max_calls: int   = 1,
    ) -> None:
        self.name                = name
        self.failure_threshold   = failure_threshold
        self.recovery_timeout    = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state             = CircuitState.CLOSED
        self._failure_count     = 0
        self._last_failure_time = 0.0
        self._half_open_calls   = 0
        self._lock              = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(f"[CB:{self.name}] OPEN → HALF_OPEN (probe started)")
        return self._state

    async def call(self, coro):
        async with self._lock:
            current = self.state
            if current == CircuitState.OPEN:
                raise RuntimeError(
                    f"Circuit breaker [{self.name}] is OPEN — "
                    f"service unavailable. Retry after {self.recovery_timeout:.0f}s."
                )
            if current == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise RuntimeError(
                        f"Circuit breaker [{self.name}] is HALF_OPEN — "
                        "waiting for probe call to complete."
                    )
                self._half_open_calls += 1

        try:
            result = await coro
            await self._on_success()
            return result
        except Exception as exc:
            await self._on_failure(exc)
            raise

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state in (CircuitState.HALF_OPEN, CircuitState.CLOSED):
                self._failure_count = 0
                self._state = CircuitState.CLOSED
                logger.debug(f"[CB:{self.name}] call succeeded → CLOSED")

    async def _on_failure(self, exc: Exception) -> None:
        async with self._lock:
            self._failure_count    += 1
            self._last_failure_time = time.monotonic()
            logger.warning(f"[CB:{self.name}] failure #{self._failure_count}: {exc}")
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.error(
                    f"[CB:{self.name}] threshold reached → OPEN "
                    f"(recovery in {self.recovery_timeout:.0f}s)"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 3. METRICS COLLECTOR
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCollector:
    """
    Thread-safe in-process counters and latency histograms.
    Replace with prometheus_client in production deployment.
    """

    def __init__(self) -> None:
        self._lock         = asyncio.Lock()
        self.total_queries = 0
        self.cache_hits    = 0
        self.errors        = 0
        self.latencies_ms: List[float] = []
        self.by_query_type: Dict[str, int] = {
            "FACTUAL": 0, "COMPARATIVE": 0,
            "REGULATORY": 0, "THESIS": 0, "CACHE_HIT": 0,
        }

    async def record(
        self,
        query_type: str,
        latency_ms: float,
        cache_hit:  bool,
        error:      bool,
    ) -> None:
        async with self._lock:
            self.total_queries += 1
            self.latencies_ms.append(latency_ms)
            if cache_hit:
                self.cache_hits += 1
            if error:
                self.errors += 1
            self.by_query_type[query_type] = (
                self.by_query_type.get(query_type, 0) + 1
            )

    def snapshot(self) -> dict:
        lats = self.latencies_ms or [0.0]
        s    = sorted(lats)
        n    = len(s)
        return {
            "total_queries"  : self.total_queries,
            "cache_hits"     : self.cache_hits,
            "cache_hit_rate" : round(self.cache_hits / max(self.total_queries, 1), 3),
            "errors"         : self.errors,
            "error_rate"     : round(self.errors / max(self.total_queries, 1), 3),
            "p50_latency_ms" : s[int(n * 0.50)],
            "p95_latency_ms" : s[int(n * 0.95)],
            "p99_latency_ms" : s[int(n * 0.99)],
            "by_query_type"  : self.by_query_type,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Central LangGraph-based router for FinThesisGuard AI.

    Lifecycle
    ---------
    __init__   → instantiate pipelines + circuit breakers + compile graphs
    initialize → warm up models, verify connectivity  (call at FastAPI startup)
    run_query  → main public entry point for financial Q&A
    run_thesis → main public entry point for thesis stress-testing
    shutdown   → clean up connections (call at FastAPI shutdown)
    """

    _instance: Optional["Orchestrator"] = None

    @classmethod
    def get_instance(cls) -> "Orchestrator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self) -> None:
        logger.info("Orchestrator.__init__ — building pipelines …")

        self.rag_pipeline    = RAGPipeline()
        self.thesis_pipeline = ThesisPipeline()
        self.metrics         = MetricsCollector()

        # Agent references — bound here so tests can patch orchestrator.agentN
        self.agent1   = acronym_resolver
        self.agent2   = query_decomposer
        self.agent3   = reranker
        self.agent4   = conflict_detector
        self.agent5   = quant_validator
        self.agent6   = thesis_analyzer
        self.agent7   = stress_synthesizer
        self.retriever = corpus_retriever

        self.cb_rag    = CircuitBreaker("rag-pipeline",    failure_threshold=5)
        self.cb_thesis = CircuitBreaker("thesis-pipeline", failure_threshold=3)
        self.cb_cache  = CircuitBreaker(
            "redis-cache", failure_threshold=10, recovery_timeout=10.0
        )

        self._rag_graph    = self._build_rag_graph()
        self._thesis_graph = self._build_thesis_graph()

        self._initialized = False
        logger.info("Orchestrator.__init__ complete — call initialize() before use")

    # ── Warm-up ───────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Must be called once at FastAPI startup.
        Verifies Redis connectivity and flips _initialized to True.
        """
        if self._initialized:
            return

        logger.info("Orchestrator.initialize — starting warm-up …")
        t0 = time.monotonic()

        await self._verify_cache()

        elapsed = (time.monotonic() - t0) * 1000
        self._initialized = True
        logger.info(f"Orchestrator.initialize complete in {elapsed:.0f}ms — ready ✅")

    async def _verify_cache(self) -> None:
        try:
            await cache.set_cache("__health_probe__", {"ok": True}, ttl=10)
            val = await cache.get_cached("__health_probe__")
            if val is None:
                raise RuntimeError("Cache write-then-read returned None")
            logger.debug("Redis cache probe: OK")
        except Exception as exc:
            logger.warning(f"Redis unavailable during init — cache degraded: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. LANGGRAPH GRAPH DEFINITIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_rag_graph(self) -> Any:
        g = StateGraph(AgentState)

        g.add_node("classify",          self._node_classify)
        g.add_node("acronym_resolve",   self._node_acronym_resolve)
        g.add_node("decompose",         self._node_decompose)
        g.add_node("retrieve",          self._node_retrieve)
        g.add_node("rerank",            self._node_rerank)
        g.add_node("validate_parallel", self._node_validate_parallel)
        g.add_node("synthesize",        self._node_synthesize)
        g.add_node("build_response",    self._node_build_rag_response)

        g.set_entry_point("classify")
        g.add_edge("classify",          "acronym_resolve")
        g.add_edge("acronym_resolve",   "decompose")
        g.add_edge("decompose",         "retrieve")
        g.add_edge("retrieve",          "rerank")
        g.add_edge("rerank",            "validate_parallel")
        g.add_edge("validate_parallel", "synthesize")
        g.add_edge("synthesize",        "build_response")
        g.add_edge("build_response",    END)

        return g.compile()

    def _build_thesis_graph(self) -> Any:
        g = StateGraph(AgentState)

        g.add_node("classify",                self._node_classify)
        g.add_node("acronym_resolve",         self._node_acronym_resolve)
        g.add_node("retrieve_thesis_context", self._node_retrieve_thesis_context)
        g.add_node("thesis_analyze_parallel", self._node_thesis_analyze_parallel)
        g.add_node("stress_synthesize",       self._node_stress_synthesize)
        g.add_node("build_response",          self._node_build_thesis_response)

        g.set_entry_point("classify")
        g.add_edge("classify",                "acronym_resolve")
        g.add_edge("acronym_resolve",         "retrieve_thesis_context")
        g.add_edge("retrieve_thesis_context", "thesis_analyze_parallel")
        g.add_edge("thesis_analyze_parallel", "stress_synthesize")
        g.add_edge("stress_synthesize",       "build_response")
        g.add_edge("build_response",          END)

        return g.compile()

    # ─────────────────────────────────────────────────────────────────────────
    # 6. ROUTING HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_query_type(query: str) -> str:
        q = query.lower()

        thesis_keywords = [
            "will outperform", "will underperform", "thesis", "investment case",
            "because", "driven by", "my view", "i believe", "overweight",
            "underweight", "buy case", "bear case", "bull case",
        ]
        if any(k in q for k in thesis_keywords) and len(query.split()) > 15:
            return QueryType.THESIS

        regulatory_keywords = [
            "rbi", "sebi", "circular", "regulation", "compliance",
            "irdai", "pfrda", "income tax", "gst", "fema", "mca",
            "penalty", "violation", "nri tax", "ltcg", "stcg",
        ]
        if any(k in q for k in regulatory_keywords):
            return QueryType.REGULATORY

        compare_keywords = [
            "vs", "versus", "compare", "comparison", "difference between",
            "better than", "worse than", "top 5",
        ]
        if any(k in q for k in compare_keywords):
            return QueryType.COMPARATIVE

        return QueryType.FACTUAL

    @staticmethod
    def _build_retrieval_filters(query_type: str) -> Dict[str, Any]:
        if query_type == QueryType.REGULATORY:
            return {
                "source_type": {"$in": ["rbi", "sebi", "mca", "irdai", "income_tax"]}
            }
        return {}

    # ─────────────────────────────────────────────────────────────────────────
    # 7. SHARED NODES
    # ─────────────────────────────────────────────────────────────────────────

    async def _node_classify(self, state: AgentState) -> AgentState:
        node = "classify"
        t0   = time.monotonic()
        try:
            qtype = self._classify_query_type(state["query"])
            state["query_type"] = qtype
            state["is_thesis"]  = (qtype == QueryType.THESIS)

            heuristic_filters = self._build_retrieval_filters(qtype)
            caller_filters    = state.get("retrieval_filters") or {}
            state["retrieval_filters"] = {**heuristic_filters, **caller_filters}

            state["node_timings"][node] = _elapsed_ms(t0)
            logger.debug(
                f"[{state['request_id']}] classify → {qtype} "
                f"({state['node_timings'][node]:.0f}ms)"
            )
        except Exception as exc:
            state = _record_error(state, node, exc)
        return state

    async def _node_acronym_resolve(self, state: AgentState) -> AgentState:
        node = "acronym_resolve"
        t0   = time.monotonic()
        try:
            resolved = await _safe_call(self.agent1.resolve, state["query"])
            state["expanded_query"]     = resolved or state["query"]
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.debug(
                f"[{state['request_id']}] acronym_resolve "
                f"({state['node_timings'][node]:.0f}ms)"
            )
        except Exception as exc:
            state["expanded_query"]     = state["query"]
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.warning(f"[{state['request_id']}] acronym_resolve WARN: {exc}")
        return state

    async def _node_quant_validate(self, state: AgentState) -> AgentState:
        node = "quant_validate"
        t0   = time.monotonic()
        try:
            chunks = state.get("reranked_chunks") or state.get("chunks") or []
            issues = await _safe_call(self.agent5.validate_quantities, chunks)
            state["quant_issues"]       = issues or []
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.debug(
                f"[{state['request_id']}] quant_validate → "
                f"{len(state['quant_issues'])} issue(s) "
                f"({state['node_timings'][node]:.0f}ms)"
            )
        except Exception as exc:
            state["quant_issues"]       = []
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.warning(f"[{state['request_id']}] quant_validate WARN: {exc}")
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # 8. RAG-SPECIFIC NODES
    # ─────────────────────────────────────────────────────────────────────────

    async def _node_decompose(self, state: AgentState) -> AgentState:
        node = "decompose"
        t0   = time.monotonic()
        try:
            qtype      = state.get("query_type", QueryType.FACTUAL)
            expanded   = state["expanded_query"]
            is_complex = await _safe_call(self.agent2.is_complex_query, expanded)

            if qtype not in (QueryType.COMPARATIVE,) and not is_complex:
                state["sub_queries"]        = [expanded]
                state["node_timings"][node] = _elapsed_ms(t0)
                logger.debug(
                    f"[{state['request_id']}] decompose skipped (simple query, "
                    f"{state['node_timings'][node]:.0f}ms)"
                )
                return state

            sub_queries = await _safe_call(self.agent2.decompose, expanded)
            state["sub_queries"]        = sub_queries or [expanded]
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.debug(
                f"[{state['request_id']}] decompose → "
                f"{len(state['sub_queries'])} sub-queries "
                f"({state['node_timings'][node]:.0f}ms)"
            )
        except Exception as exc:
            state["sub_queries"]        = [state["expanded_query"]]
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.warning(f"[{state['request_id']}] decompose WARN: {exc}")
        return state

    async def _node_retrieve(self, state: AgentState) -> AgentState:
        node = "retrieve"
        t0   = time.monotonic()
        try:
            chunks = await self.retriever.retrieve_multi(
                queries = state["sub_queries"],
                top_k   = settings.RAG_TOP_K,
                filters = state.get("retrieval_filters") or {},
            )
            state["chunks"]             = chunks or []
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.debug(
                f"[{state['request_id']}] retrieve → "
                f"{len(state['chunks'])} chunks "
                f"({state['node_timings'][node]:.0f}ms)"
            )
        except Exception as exc:
            state["chunks"]             = []
            state = _record_error(state, node, exc)
        return state

    async def _node_rerank(self, state: AgentState) -> AgentState:
        node = "rerank"
        t0   = time.monotonic()
        try:
            reranked = await _safe_call(
                self.agent3.rerank,
                chunks = state["chunks"],
                query  = state["expanded_query"],
            )
            state["reranked_chunks"]    = reranked or state.get("chunks", [])
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.debug(
                f"[{state['request_id']}] rerank → "
                f"{len(state['reranked_chunks'])} chunks "
                f"({state['node_timings'][node]:.0f}ms)"
            )
        except Exception as exc:
            state["reranked_chunks"]    = state.get("chunks", [])
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.warning(f"[{state['request_id']}] rerank WARN: {exc}")
        return state

    async def _node_conflict_detect(self, state: AgentState) -> AgentState:
        node = "conflict_detect"
        t0   = time.monotonic()
        try:
            chunks    = state.get("reranked_chunks") or state.get("chunks") or []
            conflicts = await _safe_call(self.agent4.detect_conflicts, chunks)
            state["conflicts"] = [
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in (conflicts or [])
            ]
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.debug(
                f"[{state['request_id']}] conflict_detect → "
                f"{len(state['conflicts'])} conflict(s) "
                f"({state['node_timings'][node]:.0f}ms)"
            )
        except Exception as exc:
            state["conflicts"]          = []
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.warning(f"[{state['request_id']}] conflict_detect WARN: {exc}")
        return state

    async def _node_validate_parallel(self, state: AgentState) -> AgentState:
        node = "validate_parallel"
        t0   = time.monotonic()

        cd_result, qv_result = await asyncio.gather(
            self._node_conflict_detect(dict(state)),
            self._node_quant_validate(dict(state)),
            return_exceptions=False,
        )

        state["conflicts"]    = cd_result.get("conflicts",    [])
        state["quant_issues"] = qv_result.get("quant_issues", [])

        if cd_result.get("error") and not state.get("error"):
            state["error"] = cd_result["error"]
        if qv_result.get("error") and not state.get("error"):
            state["error"] = qv_result["error"]

        state["node_timings"].update(cd_result.get("node_timings", {}))
        state["node_timings"].update(qv_result.get("node_timings", {}))
        state["node_timings"][node] = _elapsed_ms(t0)
        return state

    async def _node_synthesize(self, state: AgentState) -> AgentState:
        """
        Groq LLM — synthesise final cited answer from reranked chunks.
        synthesize_answer returns (answer, confidence, reasoning) 3-tuple.
        reasoning is discarded — _build_reasoning_trace generates its own trace.
        """
        node = "synthesize"
        t0   = time.monotonic()
        try:
            answer, confidence, _reasoning = await self.rag_pipeline.synthesize_answer(
                query     = state["expanded_query"],
                chunks    = state.get("reranked_chunks") or state.get("chunks") or [],
                conflicts = state.get("conflicts", []),
            )
            state["answer"]             = answer
            state["confidence"]         = confidence
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.debug(
                f"[{state['request_id']}] synthesize confidence={confidence} "
                f"({state['node_timings'][node]:.0f}ms)"
            )
        except Exception as exc:
            state["answer"]             = _fallback_answer(state.get("error"))
            state["confidence"]         = "Low"
            state = _record_error(state, node, exc)
        return state

    async def _node_build_rag_response(self, state: AgentState) -> AgentState:
        node = "build_response"
        t0   = time.monotonic()
        try:
            total_ms = sum(state.get("node_timings", {}).values())
            response = QueryResponse(
                answer       = state.get("answer", ""),
                citations    = self.rag_pipeline._build_citations(
                                   state.get("reranked_chunks", [])
                               ),
                confidence   = state.get("confidence", "Low"),
                conflicts    = state.get("conflicts",    []),
                quant_issues = state.get("quant_issues", []),
                data_gaps    = self.rag_pipeline._infer_data_gaps(
                                   state.get("reranked_chunks", []),
                                   state.get("sub_queries", []),
                               ),
                reasoning    = self.rag_pipeline._build_reasoning_trace(state),
                agents_used  = list(state.get("node_timings", {}).keys()),
                latency_ms   = int(total_ms),
            )
            state["response"]           = response.model_dump()
            state["total_latency_ms"]   = total_ms
            state["node_timings"][node] = _elapsed_ms(t0)
        except Exception as exc:
            state = _record_error(state, node, exc)
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # 9. THESIS-SPECIFIC NODES
    # ─────────────────────────────────────────────────────────────────────────

    async def _node_retrieve_thesis_context(self, state: AgentState) -> AgentState:
        node = "retrieve_context"
        t0   = time.monotonic()
        try:
            chunks = await self.retriever.retrieve_thesis_context(
                thesis = state["expanded_query"],
                top_k  = settings.THESIS_TOP_K,
            )
            state["chunks"]             = chunks or []
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.debug(
                f"[{state['request_id']}] retrieve_context → "
                f"{len(state['chunks'])} chunks "
                f"({state['node_timings'][node]:.0f}ms)"
            )
        except Exception as exc:
            state["chunks"]             = []
            state = _record_error(state, node, exc)
        return state

    async def _node_thesis_analyze(self, state: AgentState) -> AgentState:
        node = "thesis_analyze"
        t0   = time.monotonic()
        try:
            analysis = await _safe_call(
                self.agent6.analyze,
                thesis = state["expanded_query"],
                chunks = state.get("chunks", []),
            )
            state.setdefault("thesis_data", {}).update(analysis or {})
            state["node_timings"][node] = _elapsed_ms(t0)
            n_assumptions = len((analysis or {}).get("assumptions", []))
            logger.debug(
                f"[{state['request_id']}] thesis_analyze "
                f"assumptions={n_assumptions} "
                f"({state['node_timings'][node]:.0f}ms)"
            )
        except Exception as exc:
            state.setdefault("thesis_data", {}).update({
                "assumptions":      [],
                "dependency_chain": [],
            })
            state = _record_error(state, node, exc)
        return state

    async def _node_thesis_analyze_parallel(self, state: AgentState) -> AgentState:
        node = "thesis_analyze_parallel"
        t0   = time.monotonic()

        a6_result, a5_result = await asyncio.gather(
            self._node_thesis_analyze(dict(state)),
            self._node_quant_validate(dict(state)),
            return_exceptions=False,
        )

        state["thesis_data"]  = a6_result.get("thesis_data",  {})
        state["quant_issues"] = a5_result.get("quant_issues", [])

        if a6_result.get("error") and not state.get("error"):
            state["error"] = a6_result["error"]
        if a5_result.get("error") and not state.get("error"):
            state["error"] = a5_result["error"]

        state["node_timings"].update(a6_result.get("node_timings", {}))
        state["node_timings"].update(a5_result.get("node_timings", {}))
        state["node_timings"][node] = _elapsed_ms(t0)
        return state

    async def _node_stress_synthesize(self, state: AgentState) -> AgentState:
        """
        Agent 7 — risk scores, break conditions, thesis strength.

        FIX #1: pass thesis_text= which is required by StressSynthesizer.synthesize().
        FIX #2: error fallback uses 'Weak' not 'Unknown' (Pydantic rejects 'Unknown').
        """
        node = "stress_synthesize"
        t0   = time.monotonic()
        try:
            synthesis = await _safe_call(
                self.agent7.synthesize,
                thesis_text = state["query"],              # FIX #1: required positional arg
                thesis_data = state.get("thesis_data", {}),
                chunks      = state.get("chunks", []),
            )
            state.setdefault("thesis_data", {}).update(synthesis or {})
            state["node_timings"][node] = _elapsed_ms(t0)
            logger.debug(
                f"[{state['request_id']}] stress_synthesize "
                f"strength={state['thesis_data'].get('thesis_strength')} "
                f"({state['node_timings'][node]:.0f}ms)"
            )
        except Exception as exc:
            # FIX #2: 'Weak' is a valid Literal value; 'Unknown' is not
            state.setdefault("thesis_data", {}).update({
                "thesis_strength":  "Weak",
                "confidence":       "Low",
                "risks":            [],
                "break_conditions": [],
            })
            state = _record_error(state, node, exc)
        return state

    async def _node_build_thesis_response(self, state: AgentState) -> AgentState:
        """
        Assemble the final ThesisResponse Pydantic model.

        FIX #3: coerce thesis_strength through _coerce_strength() before passing
        to ThesisResponse. Pydantic rejects any value outside
        Literal["Strong", "Medium", "Weak"] with a ValidationError.
        """
        node = "build_response"
        t0   = time.monotonic()
        try:
            total_ms = sum(state.get("node_timings", {}).values())
            td       = state.get("thesis_data", {})

            # FIX #3: always coerce — never pass raw Agent 7 output directly
            thesis_strength = _coerce_strength(td.get("thesis_strength"))

            response = ThesisResponse(
                thesis_text           = state["query"],
                assumptions           = td.get("assumptions",           []),
                dependency_chain      = td.get("dependency_chain",      []),
                risks                 = td.get("risks",                 []),
                break_conditions      = td.get("break_conditions",      []),
                historical_analogs    = td.get("historical_analogs",    []),
                thesis_strength       = thesis_strength,               # FIX #3
                structural_robustness = td.get("structural_robustness", "Low"),
                confidence            = td.get("confidence",            "Low"),
                citations             = self.rag_pipeline._build_citations(
                                            state.get("chunks", [])
                                        ),
                quant_issues          = state.get("quant_issues", []),
                latency_ms            = int(total_ms),
            )
            state["response"]           = response.model_dump()
            state["total_latency_ms"]   = total_ms
            state["node_timings"][node] = _elapsed_ms(t0)
        except Exception as exc:
            state = _record_error(state, node, exc)
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # 10. PUBLIC ENTRY POINTS
    # ─────────────────────────────────────────────────────────────────────────

    async def run_query(
        self,
        query:     str,
        top_k:     int                      = settings.RAG_TOP_K,
        use_cache: bool                     = True,
        filters:   Optional[Dict[str, Any]] = None,
    ) -> QueryResponse:
        """Primary entry point for financial Q&A."""
        if not self._initialized:
            raise RuntimeError(
                "Orchestrator not initialized — call await orchestrator.initialize() first"
            )

        request_id = _new_request_id()
        t_start    = time.monotonic()
        logger.info(f"[{request_id}] run_query START  query={query[:80]!r}")

        cache_key = _make_cache_key("rag", query)
        if use_cache:
            cached = await self._safe_cache_get(cache_key)
            if cached:
                latency_ms = _elapsed_ms(t_start)
                logger.info(f"[{request_id}] CACHE HIT ({latency_ms:.0f}ms)")
                await self.metrics.record(
                    query_type="CACHE_HIT", latency_ms=latency_ms,
                    cache_hit=True, error=False,
                )
                return QueryResponse(**cached)

        initial_state: AgentState = {
            "request_id"       : request_id,
            "query"            : query,
            "query_type"       : "",
            "is_thesis"        : False,
            "expanded_query"   : query,
            "sub_queries"      : [],
            "retrieval_filters": filters or {},
            "chunks"           : [],
            "reranked_chunks"  : [],
            "conflicts"        : [],
            "quant_issues"     : [],
            "answer"           : "",
            "thesis_data"      : {},
            "response"         : {},
            "confidence"       : "Low",
            "node_timings"     : {},
            "total_latency_ms" : 0.0,
            "error"            : None,
            "cache_hit"        : False,
        }

        try:
            final_state = await self.cb_rag.call(
                self._rag_graph.ainvoke(initial_state)
            )
        except Exception as exc:
            logger.error(
                f"[{request_id}] run_query FATAL: {exc}\n"
                + traceback.format_exc()
            )
            await self.metrics.record(
                query_type="UNKNOWN", latency_ms=_elapsed_ms(t_start),
                cache_hit=False, error=True,
            )
            raise

        response_dict = final_state.get("response", {})
        total_latency = _elapsed_ms(t_start)
        qtype         = final_state.get("query_type") or "FACTUAL"

        log_query(
            query            = query,
            response_time_ms = total_latency,
            confidence       = final_state.get("confidence", "Low"),
            query_type       = qtype,
        )
        logger.info(
            f"[{request_id}] run_query DONE  type={qtype} "
            f"latency={total_latency:.0f}ms  "
            f"nodes={list(final_state.get('node_timings', {}).keys())}"
        )
        _log_node_timings(request_id, final_state.get("node_timings", {}))

        if total_latency > settings.MAX_LATENCY_MS:
            logger.warning(
                f"[{request_id}] LATENCY BUDGET EXCEEDED: "
                f"{total_latency:.0f}ms > {settings.MAX_LATENCY_MS}ms"
            )

        if use_cache and response_dict and not final_state.get("error"):
            await self._safe_cache_set(cache_key, response_dict)

        await self.metrics.record(
            query_type = qtype,
            latency_ms = total_latency,
            cache_hit  = False,
            error      = bool(final_state.get("error")),
        )

        if not response_dict:
            raise RuntimeError(
                f"[{request_id}] Pipeline produced empty response — check logs"
            )

        return QueryResponse(**response_dict)

    async def run_thesis(
        self,
        thesis:    str,
        context:   Optional[str] = None,
        use_cache: bool          = True,
    ) -> ThesisResponse:
        """
        Primary entry point for investment thesis stress-testing.
        NOTE: does NOT accept quick_mode — quick mode is handled by routes_thesis.py
        using a shorter timeout; the graph itself is identical.
        """
        if not self._initialized:
            raise RuntimeError(
                "Orchestrator not initialized — call await orchestrator.initialize() first"
            )

        request_id = _new_request_id()
        t_start    = time.monotonic()
        enriched   = f"{thesis}\n\nContext: {context}" if context else thesis

        logger.info(f"[{request_id}] run_thesis START  thesis={thesis[:80]!r}")

        # Validate BEFORE cache — zero LLM/Pinecone cost on rejection
        validation = self.thesis_pipeline.validate_thesis_input(enriched)
        if not validation.is_valid:
            raise ValueError(
                f"Thesis validation failed — {validation.rejection_reason}. "
                "Provide a subject, a claim, and a reason (because / driven by / due to)."
            )

        cache_key = _make_cache_key("thesis", enriched)
        if use_cache:
            cached = await self._safe_cache_get(cache_key)
            if cached:
                latency_ms = _elapsed_ms(t_start)
                logger.info(f"[{request_id}] CACHE HIT ({latency_ms:.0f}ms)")
                await self.metrics.record(
                    query_type="CACHE_HIT", latency_ms=latency_ms,
                    cache_hit=True, error=False,
                )
                return ThesisResponse(**cached)

        initial_state: AgentState = {
            "request_id"       : request_id,
            "query"            : enriched,
            "query_type"       : QueryType.THESIS,
            "is_thesis"        : True,
            "expanded_query"   : enriched,
            "sub_queries"      : [],
            "retrieval_filters": {},
            "chunks"           : [],
            "reranked_chunks"  : [],
            "conflicts"        : [],
            "quant_issues"     : [],
            "answer"           : "",
            "thesis_data"      : {},
            "response"         : {},
            "confidence"       : "Low",
            "node_timings"     : {},
            "total_latency_ms" : 0.0,
            "error"            : None,
            "cache_hit"        : False,
        }

        try:
            final_state = await self.cb_thesis.call(
                self._thesis_graph.ainvoke(initial_state)
            )
        except ValueError:
            raise
        except Exception as exc:
            logger.error(
                f"[{request_id}] run_thesis FATAL: {exc}\n"
                + traceback.format_exc()
            )
            await self.metrics.record(
                query_type="THESIS", latency_ms=_elapsed_ms(t_start),
                cache_hit=False, error=True,
            )
            raise

        response_dict = final_state.get("response", {})
        total_latency = _elapsed_ms(t_start)

        logger.info(
            f"[{request_id}] run_thesis DONE  "
            f"strength={final_state.get('thesis_data', {}).get('thesis_strength')}  "
            f"latency={total_latency:.0f}ms"
        )
        _log_node_timings(request_id, final_state.get("node_timings", {}))

        if total_latency > settings.MAX_LATENCY_MS * 1.5:
            logger.warning(
                f"[{request_id}] THESIS LATENCY HIGH: {total_latency:.0f}ms"
            )

        if use_cache and response_dict and not final_state.get("error"):
            await self._safe_cache_set(
                cache_key, response_dict, ttl=settings.CACHE_TTL
            )

        await self.metrics.record(
            query_type = "THESIS",
            latency_ms = total_latency,
            cache_hit  = False,
            error      = bool(final_state.get("error")),
        )

        if not response_dict:
            raise RuntimeError(
                f"[{request_id}] Thesis pipeline produced empty response — check logs"
            )

        return ThesisResponse(**response_dict)

    # ─────────────────────────────────────────────────────────────────────────
    # 11. CONVENIENCE
    # ─────────────────────────────────────────────────────────────────────────

    async def compare_theses(self, thesis_a: str, thesis_b: str) -> dict:
        """Runs both theses in parallel. Exposed via POST /api/thesis/compare."""
        result_a, result_b = await asyncio.gather(
            self.run_thesis(thesis_a),
            self.run_thesis(thesis_b),
            return_exceptions=True,
        )
        return {
            "thesis_a": result_a.model_dump()
                        if not isinstance(result_a, Exception)
                        else {"error": str(result_a)},
            "thesis_b": result_b.model_dump()
                        if not isinstance(result_b, Exception)
                        else {"error": str(result_b)},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 12. CACHE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    async def _safe_cache_get(self, key: str) -> Optional[dict]:
        try:
            return await self.cb_cache.call(cache.get_cached(key))
        except Exception as exc:
            logger.debug(f"cache GET skipped: {exc}")
            return None

    async def _safe_cache_set(
        self, key: str, value: dict, ttl: int = settings.CACHE_TTL
    ) -> None:
        try:
            await self.cb_cache.call(cache.set_cache(key, value, ttl=ttl))
        except Exception as exc:
            logger.debug(f"cache SET skipped: {exc}")

    async def flush_cache(self, namespace: str = "rag") -> None:
        await cache.flush_namespace(namespace)
        logger.info(f"Flushed cache namespace: {namespace}")

    # ─────────────────────────────────────────────────────────────────────────
    # 13. HEALTH + METRICS + SHUTDOWN
    # ─────────────────────────────────────────────────────────────────────────

    async def health(self) -> dict:
        checks: Dict[str, str] = {}

        try:
            await cache.get_cached("__health__")
            checks["redis"] = "ok"
        except Exception:
            checks["redis"] = "unreachable"

        try:
            from backend.retrieval.pinecone_client import pinecone_client
            stats = await pinecone_client.get_index_stats()
            checks["pinecone"] = f"ok — {stats.get('total_vector_count', 0):,} vectors"
        except Exception as exc:
            checks["pinecone"] = f"error: {exc}"

        try:
            from backend.utils.llm_client import llm_client
            await llm_client.call_llm(
                prompt="ping", system_prompt="reply ok", max_tokens=5
            )
            checks["groq"] = "ok"
        except Exception as exc:
            checks["groq"] = f"error: {exc}"

        checks["cb_rag"]    = self.cb_rag.state.value
        checks["cb_thesis"] = self.cb_thesis.state.value
        checks["cb_cache"]  = self.cb_cache.state.value

        overall = (
            "healthy"
            if all(v.startswith("ok") or v == "closed" for v in checks.values())
            else "degraded"
        )
        return {
            "status":      overall,
            "services":    checks,
            "initialized": self._initialized,
        }

    def get_metrics(self) -> dict:
        return self.metrics.snapshot()

    async def shutdown(self) -> None:
        logger.info("Orchestrator.shutdown — closing connections …")
        try:
            from backend.utils.llm_client import llm_client
            client = getattr(llm_client, "client", None)
            if client is not None:
                await client.close()
                logger.debug("Orchestrator.shutdown — Groq HTTP client closed")
        except Exception as exc:
            logger.warning(
                f"Orchestrator.shutdown Groq client close error (ignored): {exc}"
            )
        logger.info("Orchestrator.shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# 14. MODULE-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _elapsed_ms(t0: float) -> float:
    return (time.monotonic() - t0) * 1000


def _new_request_id() -> str:
    return str(uuid.uuid4())[:8]


def _make_cache_key(namespace: str, text: str) -> str:
    normalised = text.lower().strip()
    digest     = hashlib.md5(normalised.encode()).hexdigest()
    return f"{namespace}:{digest}"


def _record_error(state: AgentState, node: str, exc: Exception) -> AgentState:
    msg = f"[{node}] {type(exc).__name__}: {exc}"
    if not state.get("error"):
        state["error"] = msg
    logger.error(msg + "\n" + traceback.format_exc())
    return state


def _fallback_answer(error: Optional[str]) -> str:
    return (
        "I was unable to generate a complete answer due to an internal error. "
        "Please try again or rephrase your query. "
        + (f"(Internal: {error})" if error else "")
    )


def _log_node_timings(request_id: str, timings: Dict[str, float]) -> None:
    if not timings:
        return
    rows = "  ".join(f"{k}={v:.0f}ms" for k, v in timings.items())
    logger.info(f"[{request_id}] node timings: {rows}")


# ─────────────────────────────────────────────────────────────────────────────
# 15. MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

orchestrator: Orchestrator = Orchestrator.get_instance()