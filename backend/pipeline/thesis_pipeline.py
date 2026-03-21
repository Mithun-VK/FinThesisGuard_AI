# backend/pipeline/thesis_pipeline.py
"""
FinThesisGuard AI — Thesis Validation Pipeline
ArkAngel Financial Solutions

Orchestrates the full investment thesis stress-testing pipeline across
Agents 1, 5, 6, 7 + multi-query corpus retrieval.

Pipeline stages and latency budget (target < 3,000ms):
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Pre-flight  validate_thesis_input()         pure / sync   <   5ms   │
    │ Stage 1     Agent 1  — Acronym expansion    sequential    ~  50ms   │
    │ Stage 2     Retriever — Multi-query fetch   parallel      ~ 400ms   │
    │ Stage 3     PARALLEL ──────────────────────────────────── ~ 600ms   │
    │              ├── Agent 6: Assumptions + dependency chain            │
    │              └── Agent 5: Quantitative claim validation             │
    │ Stage 4     Agent 7  — Risk scoring + breaks + synthesis  ~ 700ms   │
    └──────────────────────────────────────────────────────────────────────┘
    Total: ~1,755ms  (well within 3,000ms hard target)

Parallel execution strategy:
    - Stage 2: All retrieval sub-queries run concurrently.
    - Stage 3: Agent 6 (heavy: 3 LLM calls) + Agent 5 (medium: 1 LLM call)
      run concurrently — both only need the expanded thesis + corpus chunks.
    - Stage 4: Agent 7 runs sequentially (needs both Stage 3 outputs).

Graceful degradation:
    Every stage is individually try/except-wrapped. Any agent failure
    produces a warning log + empty fallback. The pipeline always returns
    a valid ThesisResponse — never raises to the caller.

Input validation (validate_thesis_input):
    Accepts: Subject (company/asset) + Claim + Reason
    Rejects: Vague assertions, single-sentence proclamations, gibberish,
             pure price targets with no reasoning

    ACCEPT: "HDFC Bank will outperform because NIM expansion of 20bps
             is expected as RBI cuts rates by 75bps over FY26, driving
             PAT growth of 18-20% and a re-rating to 3.2x book value."
    REJECT: "HDFC is good"           — no claim, no reason
    REJECT: "Buy TCS"                — no reason
    REJECT: "NIM will expand"        — no subject, no reason
    REJECT: "HDFC target 2000"       — no causal reasoning

Depends on:
    backend/agents/agent1_acronym_resolver.py
    backend/agents/agent5_quant_validator.py
    backend/agents/agent6_thesis_analyzer.py
    backend/agents/agent7_stress_synthesizer.py
    backend/retrieval/retriever.py
    backend/models/thesis_models.py
    backend/models/response_models.py
    backend/utils/llm_client.py
    backend/utils/logger.py
    backend/config.py
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.agents.agent1_acronym_resolver import acronym_resolver
from backend.agents.agent5_quant_validator import quant_validator
from backend.agents.agent6_thesis_analyzer import thesis_analyzer
from backend.agents.agent7_stress_synthesizer import stress_synthesizer
from backend.config import (
    AGENT_LATENCY_TARGETS,
    AUTHORITY_WEIGHTS,
    KNOWN_COMPANIES,
    KNOWN_SECTORS,
    THESIS_MAX_LENGTH,
    THESIS_MIN_LENGTH,
    TOP_K_RETRIEVAL,
)
from backend.models.response_models import AgentTrace, Citation, ConfidenceLevel
from backend.models.thesis_models import (
    QuantitativeFlag,
    StructuralRobustness,
    ThesisResponse,
    ThesisStrength,
)
from backend.retrieval.retriever import RetrieverError, retriever
from backend.utils.logger import log_metric, logger
from backend.config import settings


# ── Pipeline identity ──────────────────────────────────────────────────────────
PIPELINE_NAME      = "thesis_pipeline"
LATENCY_TARGET_MS  = int(AGENT_LATENCY_TARGETS.get("thesis_pipeline", 2500))

# ── Retrieval: queries generated from thesis for corpus coverage ──────────────
THESIS_RETRIEVAL_QUERIES = 4        # Number of focused retrieval queries to build
THESIS_TOP_K_PER_QUERY   = 8        # Chunks per sub-query (merged + deduped later)
THESIS_TOP_K_FINAL       = 12       # Final ranked chunk count passed to agents

# ── Input validation thresholds ───────────────────────────────────────────────
_MIN_WORD_COUNT         = 12        # Minimum words in a valid thesis
_MIN_SENTENCE_COUNT     = 1         # Must be at least 1 complete sentence
_MIN_REASON_SIGNALS     = 1         # Must contain at least 1 causal connective
_MIN_CLAIM_SIGNALS      = 1         # Must contain at least 1 forward-looking verb

# ── Causal connectives — signals that a "reason" is present ───────────────────
_REASON_SIGNALS: frozenset[str] = frozenset({
    "because", "since", "as ", "due to", "driven by", "given",
    "on account of", "owing to", "thanks to", "supported by",
    "underpinned by", "premised on", "assuming", "provided that",
    "if ", "when ", "unless", "subject to", "contingent on",
    "as a result", "therefore", "hence", "consequently",
    "this will", "which will", "leading to", "enabling", "allowing",
    "resulting in", "catalyst", "tailwind", "headwind",
    "rests on", "depends on", "hinges on",
})

# ── Forward-looking claim verbs — signals that a "claim" is present ───────────
_CLAIM_SIGNALS: frozenset[str] = frozenset({
    "will ", "should ", "expect", "project", "forecast", "anticipate",
    "target", "estimate", "see ", "believe", "view", "think",
    "outperform", "underperform", "re-rate", "re-rating", "rerate",
    "expand", "compress", "grow", "decline", "rise", "fall",
    "improve", "deteriorate", "increase", "decrease", "accelerate",
    "decelerate", "beat", "miss", "upgrade", "downgrade",
    "buy", "sell", "hold", "strong buy", "accumulate", "reduce",
    "likely", "unlikely", "could ", "may ", "might ",
})

# ── Subject signals — company names, financial assets, sectors ────────────────
# Built dynamically from config + static financial asset terms
_STATIC_SUBJECT_SIGNALS: frozenset[str] = frozenset({
    "bank", "nbfc", "stock", "equity", "share", "bond", "fund",
    "index", "etf", "nifty", "sensex", "portfolio", "asset",
    "security", "debt", "credit", "yield", "spread", "rate",
    "commodity", "gold", "oil", "rupee", "dollar", "currency",
})


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ValidationResult:
    """
    Result of validate_thesis_input().

    Attributes:
        is_valid:       True if thesis passes all validation checks.
        rejection_reason: Human-readable explanation if is_valid=False.
        word_count:     Number of words in the thesis.
        has_subject:    True if a company/asset/sector was detected.
        has_claim:      True if a forward-looking claim was detected.
        has_reason:     True if causal reasoning was detected.
        detected_subjects: Company/sector names found in the thesis.
    """
    is_valid:          bool
    rejection_reason:  str                = ""
    word_count:        int                = 0
    has_subject:       bool               = False
    has_claim:         bool               = False
    has_reason:        bool               = False
    detected_subjects: tuple[str, ...]    = ()


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL PIPELINE STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _ThesisPipelineState:
    """
    Mutable state carried through each pipeline stage.
    Populated incrementally as stages complete.
    """
    original_thesis:   str                    = ""
    expanded_thesis:   str                    = ""
    retrieval_queries: list[str]              = field(default_factory=list)
    chunks:            list[dict]             = field(default_factory=list)
    thesis_data:       dict                   = field(default_factory=dict)
    quant_flags:       list[QuantitativeFlag] = field(default_factory=list)
    agents_used:       list[str]              = field(default_factory=list)
    agent_traces:      list[AgentTrace]       = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class ThesisPipeline:
    """
    Orchestrates the full investment thesis validation pipeline.

    Usage:
        from backend.pipeline.thesis_pipeline import thesis_pipeline

        response: ThesisResponse = await thesis_pipeline.run(
            thesis     = "HDFC Bank is a BUY because...",
            request_id = "req_abc123",
        )
    """

    def __init__(self) -> None:
        self._total_runs:    int   = 0
        self._total_ms:      float = 0.0
        self._rejected:      int   = 0
        self._agent_failures: dict[str, int] = {
            "agent1": 0, "retrieval": 0,
            "agent5": 0, "agent6": 0, "agent7": 0,
        }
        logger.info(
            f"[THESIS PIPELINE] Initialized | "
            f"latency_target={LATENCY_TARGET_MS}ms | "
            f"retrieval_queries={THESIS_RETRIEVAL_QUERIES} | "
            f"top_k_final={THESIS_TOP_K_FINAL}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def run(
        self,
        thesis:     str,
        request_id: Optional[str] = None,
    ) -> ThesisResponse:
        """
        Runs the full thesis validation pipeline.

        Pipeline:
            Pre-flight → validate_thesis_input()       (pure, sync)
            Stage 1    → Agent 1 acronym expansion     (sequential)
            Stage 2    → Multi-query corpus retrieval   (parallel)
            Stage 3    → Agent 6 + Agent 5 analysis    (parallel)
            Stage 4    → Agent 7 synthesis              (sequential)

        Args:
            thesis:     Raw investment thesis text from user.
            request_id: Optional trace ID from the API layer.

        Returns:
            ThesisResponse — fully assembled Pydantic model.
            Always returns a valid response — never raises to caller.
        """
        start_ms = time.perf_counter()
        self._total_runs += 1

        thesis_stripped = thesis.strip() if thesis else ""

        logger.info(
            f"[THESIS PIPELINE] Run start | "
            f"thesis_len={len(thesis_stripped)} | "
            f"request_id={request_id}"
        )

        # ── Pre-flight: input validation ──────────────────────────────────────
        validation = self.validate_thesis_input(thesis_stripped)
        if not validation.is_valid:
            self._rejected += 1
            elapsed_ms = (time.perf_counter() - start_ms) * 1000
            logger.warning(
                f"[THESIS PIPELINE] Rejected | "
                f"reason='{validation.rejection_reason}' | "
                f"thesis='{thesis_stripped[:60]}'"
            )
            return ThesisResponse.error_response(
                thesis_text   = thesis_stripped,
                error_message = validation.rejection_reason,
                request_id    = request_id,
                latency_ms    = int(elapsed_ms),
            )

        state = _ThesisPipelineState(
            original_thesis = thesis_stripped,
            expanded_thesis = thesis_stripped,
        )

        try:
            # ── Stage 1: Acronym expansion ────────────────────────────────────
            await self._stage_expand_acronyms(state)

            # ── Stage 2: Multi-query retrieval ────────────────────────────────
            await self._stage_retrieve(state)

            # ── Stage 3: Parallel — Agent 6 + Agent 5 ────────────────────────
            await self._stage_parallel_analyze(state)

            # ── Stage 4: Agent 7 synthesis ────────────────────────────────────
            response = await self._stage_synthesize(state, request_id, start_ms)

            elapsed_ms = (time.perf_counter() - start_ms) * 1000
            self._total_ms += elapsed_ms

            logger.info(
                f"[THESIS PIPELINE] Complete | "
                f"strength={response.thesis_strength} | "
                f"confidence={response.confidence} | "
                f"assumptions={len(response.assumptions)} | "
                f"risks={len(response.risks)} | "
                f"breaks={len(response.break_conditions)} | "
                f"chunks={len(state.chunks)} | "
                f"{elapsed_ms:.0f}ms"
            )
            log_metric("thesis_pipeline_duration_ms", elapsed_ms,            unit="ms")
            log_metric("thesis_pipeline_assumptions",  len(response.assumptions), unit="count")
            log_metric("thesis_pipeline_risks",        len(response.risks),       unit="count")

            if elapsed_ms > LATENCY_TARGET_MS:
                logger.warning(
                    f"[THESIS PIPELINE] Latency target breached | "
                    f"{elapsed_ms:.0f}ms > {LATENCY_TARGET_MS}ms"
                )

            return response

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_ms) * 1000
            self._total_ms += elapsed_ms
            logger.error(
                f"[THESIS PIPELINE] Unexpected error: "
                f"{type(e).__name__}: {e} | {elapsed_ms:.0f}ms"
            )
            return ThesisResponse.error_response(
                thesis_text   = thesis_stripped,
                error_message = f"{type(e).__name__}: {e}",
                request_id    = request_id,
                latency_ms    = int(elapsed_ms),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Input validation
    # ──────────────────────────────────────────────────────────────────────────

    def validate_thesis_input(self, thesis: str) -> ValidationResult:
        """
        Validates an investment thesis meets minimum quality requirements.

        A valid thesis must contain:
            1. A SUBJECT  — named company, asset, sector, or financial instrument
            2. A CLAIM    — forward-looking assertion (will, should, expect, etc.)
            3. A REASON   — causal connective (because, due to, driven by, etc.)

        Hard limits:
            - Minimum THESIS_MIN_LENGTH characters
            - Maximum THESIS_MAX_LENGTH characters
            - Minimum _MIN_WORD_COUNT words

        Acceptance examples:
            ✅ "HDFC Bank will outperform because NIM expansion of 20bps is
               expected as RBI cuts rates by 75bps over FY26."
            ✅ "TCS is a BUY due to AI-driven capex from hyperscalers driving
               12% revenue CAGR through FY28."

        Rejection examples:
            ❌ "HDFC is good"         → missing claim + reason
            ❌ "Buy TCS"              → missing reason
            ❌ "NIM will expand"      → missing subject + reason
            ❌ "HDFC target ₹2000"   → missing causal reasoning
            ❌ ""                     → empty
            ❌ "Buy buy buy!!!"       → no subject, no reason, too short

        Args:
            thesis: Raw thesis text to validate.

        Returns:
            ValidationResult — immutable dataclass with is_valid + diagnostics.
        """
        if not thesis or not thesis.strip():
            return ValidationResult(
                is_valid         = False,
                rejection_reason = "Thesis text is empty.",
            )

        thesis_clean = thesis.strip()

        # ── Hard length guards ────────────────────────────────────────────────
        if len(thesis_clean) < THESIS_MIN_LENGTH:
            return ValidationResult(
                is_valid         = False,
                rejection_reason = (
                    f"Thesis is too short ({len(thesis_clean)} characters). "
                    f"Minimum is {THESIS_MIN_LENGTH} characters. "
                    f"A valid thesis needs a subject, a claim, and a reason."
                ),
            )

        if len(thesis_clean) > THESIS_MAX_LENGTH:
            return ValidationResult(
                is_valid         = False,
                rejection_reason = (
                    f"Thesis exceeds maximum length "
                    f"({len(thesis_clean)} > {THESIS_MAX_LENGTH} characters). "
                    f"Please trim to the core investment argument."
                ),
            )

        thesis_lower = thesis_clean.lower()
        words        = thesis_lower.split()

        # ── Word count guard ──────────────────────────────────────────────────
        if len(words) < _MIN_WORD_COUNT:
            return ValidationResult(
                is_valid         = False,
                rejection_reason = (
                    f"Thesis is too brief ({len(words)} words). "
                    f"Minimum is {_MIN_WORD_COUNT} words. "
                    f"Include a subject, a directional claim, and a reason."
                ),
                word_count = len(words),
            )

        # ── Subject detection ─────────────────────────────────────────────────
        # Check known companies + sectors (from config) + static financial terms
        known_companies_lower = [c.lower() for c in KNOWN_COMPANIES]
        known_sectors_flat    = [
            kw.lower()
            for keywords in KNOWN_SECTORS.values()
            for kw in keywords
        ]

        detected_subjects: list[str] = []

        for company in known_companies_lower:
            if company in thesis_lower:
                detected_subjects.append(company)

        for sector_kw in known_sectors_flat:
            if sector_kw in thesis_lower and sector_kw not in detected_subjects:
                detected_subjects.append(sector_kw)

        for static_term in _STATIC_SUBJECT_SIGNALS:
            if static_term in thesis_lower and static_term not in detected_subjects:
                detected_subjects.append(static_term)

        has_subject = len(detected_subjects) > 0

        # ── Claim detection ───────────────────────────────────────────────────
        has_claim = any(signal in thesis_lower for signal in _CLAIM_SIGNALS)

        # ── Reason detection ──────────────────────────────────────────────────
        has_reason = any(signal in thesis_lower for signal in _REASON_SIGNALS)

        # ── Build detailed rejection message ──────────────────────────────────
        if not has_subject and not has_claim and not has_reason:
            return ValidationResult(
                is_valid         = False,
                rejection_reason = (
                    "Thesis is missing a subject, a claim, and a reason. "
                    "Example: 'HDFC Bank will outperform because NIM expansion "
                    "of 20bps is expected as RBI cuts rates by 75bps over FY26.'"
                ),
                word_count        = len(words),
                has_subject       = False,
                has_claim         = False,
                has_reason        = False,
                detected_subjects = (),
            )

        if not has_subject:
            return ValidationResult(
                is_valid         = False,
                rejection_reason = (
                    "Thesis does not identify a specific company, sector, or "
                    "financial instrument. Add a named subject "
                    "(e.g., 'HDFC Bank', 'banking sector', 'Nifty 50')."
                ),
                word_count  = len(words),
                has_subject = False,
                has_claim   = has_claim,
                has_reason  = has_reason,
            )

        if not has_claim:
            return ValidationResult(
                is_valid         = False,
                rejection_reason = (
                    "Thesis lacks a forward-looking claim. "
                    "Add directional language: 'will outperform', "
                    "'expected to expand', 'should re-rate to', etc."
                ),
                word_count        = len(words),
                has_subject       = True,
                has_claim         = False,
                has_reason        = has_reason,
                detected_subjects = tuple(detected_subjects[:5]),
            )

        if not has_reason:
            return ValidationResult(
                is_valid         = False,
                rejection_reason = (
                    "Thesis makes a claim but provides no reasoning. "
                    "Add causal logic: 'because...', 'due to...', "
                    "'driven by...', 'given that...', etc. "
                    f"Example: '{detected_subjects[0].title()} will outperform "
                    f"because [specific reason here]'."
                ),
                word_count        = len(words),
                has_subject       = True,
                has_claim         = True,
                has_reason        = False,
                detected_subjects = tuple(detected_subjects[:5]),
            )

        # ── All checks passed ─────────────────────────────────────────────────
        logger.debug(
            f"[THESIS PIPELINE] Validation passed | "
            f"words={len(words)} | "
            f"subjects={detected_subjects[:3]} | "
            f"has_claim={has_claim} | has_reason={has_reason}"
        )
        return ValidationResult(
            is_valid          = True,
            rejection_reason  = "",
            word_count        = len(words),
            has_subject       = True,
            has_claim         = True,
            has_reason        = True,
            detected_subjects = tuple(detected_subjects[:5]),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Stage 1 — Acronym expansion
    # ──────────────────────────────────────────────────────────────────────────

    async def _stage_expand_acronyms(
        self,
        state: _ThesisPipelineState,
    ) -> None:
        """
        Stage 1 — Agent 1: Expands financial acronyms in the thesis.

        Critical for retrieval quality: "NIM expansion as RBI cuts MCLR"
        becomes "Net Interest Margin expansion as Reserve Bank of India cuts
        Marginal Cost of Funds-based Lending Rate" — Pinecone matches improve
        significantly on full-term queries.

        Failure: Non-fatal. Original thesis passes through unchanged.
        Target: 50ms.
        """
        stage_start = time.perf_counter()
        try:
            expanded = await acronym_resolver.expand(state.original_thesis)
            state.expanded_thesis = expanded if expanded else state.original_thesis
            state.agents_used.append("agent1_acronym_resolver")

            elapsed = (time.perf_counter() - stage_start) * 1000
            logger.debug(
                f"[THESIS PIPELINE] Stage 1 (acronym) | "
                f"original_len={len(state.original_thesis)} | "
                f"expanded_len={len(state.expanded_thesis)} | "
                f"{elapsed:.0f}ms"
            )
            self._record_trace(
                state          = state,
                agent_name     = "agent1_acronym_resolver",
                elapsed_ms     = elapsed,
                success        = True,
                input_summary  = state.original_thesis[:100],
                output_summary = state.expanded_thesis[:100],
            )

        except Exception as e:
            elapsed = (time.perf_counter() - stage_start) * 1000
            self._agent_failures["agent1"] += 1
            logger.warning(
                f"[THESIS PIPELINE] Stage 1 (acronym) failed: "
                f"{type(e).__name__}: {e}. Using original thesis."
            )
            self._record_trace(
                state      = state,
                agent_name = "agent1_acronym_resolver",
                elapsed_ms = elapsed,
                success    = False,
                error      = str(e),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Stage 2 — Multi-query corpus retrieval
    # ──────────────────────────────────────────────────────────────────────────

    async def _stage_retrieve(
        self,
        state: _ThesisPipelineState,
    ) -> None:
        """
        Stage 2 — Parallel multi-query corpus retrieval.

        Strategy:
            Build THESIS_RETRIEVAL_QUERIES focused sub-queries from the
            expanded thesis, then retrieve chunks for all in parallel.
            Merge results by chunk_id (keep highest final_score on collision).
            Re-sort merged set and cap at THESIS_TOP_K_FINAL.

        Sub-query types generated:
            Q1: Full expanded thesis (broadest coverage)
            Q2: Subject + core financial metric (e.g., "HDFC Bank NIM")
            Q3: Macro / regulatory context (e.g., "RBI rate cut repo")
            Q4: Historical / valuation context (e.g., "HDFC book value re-rate")

        Failure: Non-fatal — state.chunks remains empty. Agent 6 + 7 will
        produce low-confidence output with no corpus evidence.
        Target: 400ms.
        """
        stage_start = time.perf_counter()

        state.retrieval_queries = self._build_retrieval_queries(
            thesis          = state.original_thesis,
            expanded_thesis = state.expanded_thesis,
        )

        async def _fetch(query: str) -> list[dict]:
            try:
                return await retriever.retrieve(
                    query                = query,
                    top_k                = THESIS_TOP_K_PER_QUERY,
                    search_all_namespaces = True,
                    min_score            = 0.25,
                )
            except RetrieverError as e:
                logger.warning(
                    f"[THESIS PIPELINE] Retrieval failed for "
                    f"query='{query[:60]}': {e}"
                )
                return []

        try:
            results_per_query: list[list[dict]] = await asyncio.gather(
                *[_fetch(q) for q in state.retrieval_queries],
                return_exceptions=False,
            )

            # Merge + deduplicate by chunk id
            merged: dict[str, dict] = {}
            for result_list in results_per_query:
                for chunk in result_list:
                    cid   = chunk.get("id") or chunk.get("chunk_id") or ""
                    score = float(chunk.get("final_score", 0.0))
                    if cid not in merged or score > float(
                        merged[cid].get("final_score", 0.0)
                    ):
                        merged[cid] = chunk

            # Sort and cap
            state.chunks = sorted(
                merged.values(),
                key=lambda c: float(c.get("final_score", 0.0)),
                reverse=True,
            )[:THESIS_TOP_K_FINAL]

            elapsed = (time.perf_counter() - stage_start) * 1000
            total_raw = sum(len(r) for r in results_per_query)

            logger.debug(
                f"[THESIS PIPELINE] Stage 2 (retrieve) | "
                f"queries={len(state.retrieval_queries)} | "
                f"total_raw={total_raw} | "
                f"after_dedup={len(state.chunks)} | "
                f"{elapsed:.0f}ms"
            )
            self._record_trace(
                state          = state,
                agent_name     = "retriever",
                elapsed_ms     = elapsed,
                success        = True,
                input_summary  = f"{len(state.retrieval_queries)} retrieval queries",
                output_summary = f"{len(state.chunks)} unique chunks "
                                 f"(raw={total_raw})",
            )

            if not state.chunks:
                logger.warning(
                    f"[THESIS PIPELINE] Stage 2 (retrieve): "
                    f"Zero chunks returned. Analysis will proceed with "
                    f"no corpus evidence — confidence will be Low."
                )

        except Exception as e:
            elapsed = (time.perf_counter() - stage_start) * 1000
            self._agent_failures["retrieval"] += 1
            logger.error(
                f"[THESIS PIPELINE] Stage 2 (retrieve) failed: "
                f"{type(e).__name__}: {e}"
            )
            self._record_trace(
                state      = state,
                agent_name = "retriever",
                elapsed_ms = elapsed,
                success    = False,
                error      = str(e),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Stage 3 — Parallel Agent 6 + Agent 5
    # ──────────────────────────────────────────────────────────────────────────

    async def _stage_parallel_analyze(
        self,
        state: _ThesisPipelineState,
    ) -> None:
        """
        Stage 3 — Parallel: Agent 6 (thesis analysis) + Agent 5 (quant validation).

        Both agents read only from:
            - state.expanded_thesis  (the thesis text)
            - state.chunks           (the corpus evidence)
        They have zero data dependency on each other → run concurrently.

        Agent 6 — thesis_analyzer.analyze():
            Extracts assumptions, builds dependency DAG, finds historical
            analogs. Returns thesis_data dict with 7 keys.

        Agent 5 — quant_validator.validate():
            Validates numerical claims in the thesis against corpus data.
            Returns List[QuantitativeFlag] for discrepancies.

        Failures:
            Agent 6 failure → thesis_data = {} → Agent 7 receives empty
            assumptions; ThesisResponse will be weak/low confidence.
            Agent 5 failure → quant_flags = [] → non-fatal.
        Target: max(600ms, 300ms) = 600ms (Agent 6 dominates, both parallel).
        """
        async def _run_agent6() -> dict:
            t = time.perf_counter()
            try:
                result = await thesis_analyzer.analyze(state.expanded_thesis)
                elapsed = (time.perf_counter() - t) * 1000
                state.agents_used.append("agent6_thesis_analyzer")
                self._record_trace(
                    state          = state,
                    agent_name     = "agent6_thesis_analyzer",
                    elapsed_ms     = elapsed,
                    success        = True,
                    input_summary  = f"thesis_len={len(state.expanded_thesis)}",
                    output_summary = (
                        f"assumptions={len(result.get('assumptions', []))} | "
                        f"robustness={result.get('structural_robustness')} | "
                        f"circular={result.get('has_circular_dependency')}"
                    ),
                )
                logger.debug(
                    f"[THESIS PIPELINE] Agent 6 complete | "
                    f"assumptions={len(result.get('assumptions', []))} | "
                    f"{elapsed:.0f}ms"
                )
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - t) * 1000
                self._agent_failures["agent6"] += 1
                logger.warning(
                    f"[THESIS PIPELINE] Agent 6 failed: "
                    f"{type(e).__name__}: {e}. "
                    f"Continuing with empty thesis_data."
                )
                self._record_trace(
                    state      = state,
                    agent_name = "agent6_thesis_analyzer",
                    elapsed_ms = elapsed,
                    success    = False,
                    error      = str(e),
                )
                return {
                    "assumptions":            [],
                    "dependency_chain":       [],
                    "dependency_nodes":       [],
                    "historical_analogs":     [],
                    "structural_robustness":  "Low",
                    "has_circular_dependency": False,
                    "cycle_paths":            [],
                    "agents_used":            [],
                }

        async def _run_agent5() -> list[QuantitativeFlag]:
            t = time.perf_counter()
            try:
                result = await quant_validator.validate(
                    query  = state.expanded_thesis,
                    chunks = state.chunks,
                )
                elapsed = (time.perf_counter() - t) * 1000
                state.agents_used.append("agent5_quant_validator")
                self._record_trace(
                    state          = state,
                    agent_name     = "agent5_quant_validator",
                    elapsed_ms     = elapsed,
                    success        = True,
                    input_summary  = f"chunks={len(state.chunks)}",
                    output_summary = f"flags={len(result or [])}",
                )
                logger.debug(
                    f"[THESIS PIPELINE] Agent 5 complete | "
                    f"flags={len(result or [])} | {elapsed:.0f}ms"
                )
                return result or []
            except Exception as e:
                elapsed = (time.perf_counter() - t) * 1000
                self._agent_failures["agent5"] += 1
                logger.warning(
                    f"[THESIS PIPELINE] Agent 5 failed: "
                    f"{type(e).__name__}: {e}. Quant flags = []."
                )
                self._record_trace(
                    state      = state,
                    agent_name = "agent5_quant_validator",
                    elapsed_ms = elapsed,
                    success    = False,
                    error      = str(e),
                )
                return []

        # Run both agents concurrently
        thesis_data, quant_flags = await asyncio.gather(
            _run_agent6(),
            _run_agent5(),
            return_exceptions=False,
        )

        state.thesis_data  = thesis_data
        state.quant_flags  = quant_flags

        # Merge upstream agents_used from Agent 6
        for agent in thesis_data.get("agents_used", []):
            if agent not in state.agents_used:
                state.agents_used.append(agent)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Stage 4 — Agent 7 synthesis
    # ──────────────────────────────────────────────────────────────────────────

    async def _stage_synthesize(
        self,
        state:      _ThesisPipelineState,
        request_id: Optional[str],
        start_ms:   float,
    ) -> ThesisResponse:
        """
        Stage 4 — Agent 7: Risk scoring + break conditions + final synthesis.

        Delegates to stress_synthesizer.synthesize() which:
            1. Scores 6 risk dimensions in parallel (Groq × 6 concurrent).
            2. Identifies break conditions (1 Groq call).
            3. Calculates thesis_strength (pure computation).
            4. Assigns confidence (heuristic).
            5. Generates synthesis paragraph (1 Groq call).
            6. Assembles complete ThesisResponse.

        Injects:
            - thesis_text:        state.expanded_thesis
            - thesis_data:        state.thesis_data  (Agent 6 output)
            - chunks:             state.chunks        (corpus evidence)
            - quantitative_flags: state.quant_flags   (Agent 5 output)
            - request_id:         for tracing

        On complete Agent 7 failure (rare), falls back to
        ThesisResponse.error_response() with the accumulated
        thesis_data and agents_used.
        Target: 700ms.
        """
        stage_start = time.perf_counter()

        # Append pipeline-level agents_used before passing to Agent 7
        state.thesis_data["agents_used"] = list(set(
            state.thesis_data.get("agents_used", []) + state.agents_used
        ))

        try:
            response = await stress_synthesizer.synthesize(
                thesis_text        = state.expanded_thesis,
                thesis_data        = state.thesis_data,
                chunks             = state.chunks,
                quantitative_flags = state.quant_flags,
                request_id         = request_id,
            )

            # Append pipeline name to agents_used if not already present
            if PIPELINE_NAME not in response.agents_used:
                # ThesisResponse is immutable after construction; rebuild
                # agents_used via model_copy (Pydantic v2)
                updated_agents = [*response.agents_used, PIPELINE_NAME]
                response = response.model_copy(
                    update={"agents_used": updated_agents}
                )

            elapsed = (time.perf_counter() - stage_start) * 1000
            self._record_trace(
                state          = state,
                agent_name     = "agent7_stress_synthesizer",
                elapsed_ms     = elapsed,
                success        = True,
                input_summary  = (
                    f"assumptions={len(state.thesis_data.get('assumptions', []))} | "
                    f"chunks={len(state.chunks)} | "
                    f"quant_flags={len(state.quant_flags)}"
                ),
                output_summary = (
                    f"strength={response.thesis_strength} | "
                    f"confidence={response.confidence} | "
                    f"risks={len(response.risks)} | "
                    f"breaks={len(response.break_conditions)}"
                ),
            )
            return response

        except Exception as e:
            elapsed = (time.perf_counter() - stage_start) * 1000
            self._agent_failures["agent7"] += 1
            total_elapsed = (time.perf_counter() - start_ms) * 1000
            logger.error(
                f"[THESIS PIPELINE] Stage 4 (Agent 7) failed: "
                f"{type(e).__name__}: {e} | stage={elapsed:.0f}ms"
            )
            self._record_trace(
                state      = state,
                agent_name = "agent7_stress_synthesizer",
                elapsed_ms = elapsed,
                success    = False,
                error      = str(e),
            )
            return ThesisResponse.error_response(
                thesis_text   = state.expanded_thesis,
                error_message = (
                    f"Risk synthesis failed: {type(e).__name__}: {e}. "
                    f"Assumptions extracted: "
                    f"{len(state.thesis_data.get('assumptions', []))}."
                ),
                request_id = request_id,
                latency_ms = int(total_elapsed),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Retrieval query builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_retrieval_queries(
        self,
        thesis:          str,
        expanded_thesis: str,
    ) -> list[str]:
        """
        Builds THESIS_RETRIEVAL_QUERIES focused sub-queries from the thesis.

        Query construction strategy:
            Q1: Full expanded thesis (broadest semantic coverage)
            Q2: Subject + financial metric keyphrases
                (extracted from thesis text → targets corporate disclosures)
            Q3: Regulatory / macro context
                (RBI, SEBI, repo rate, inflation → targets regulatory corpus)
            Q4: Historical / valuation analogs
                (book value, P/E, historical range → targets research notes)

        Deduplicates and ensures all queries are non-empty.
        Falls back to [expanded_thesis] if extraction produces nothing useful.

        Args:
            thesis:          Original (pre-expansion) thesis text.
            expanded_thesis: Acronym-expanded thesis text.

        Returns:
            list[str] — 1 to THESIS_RETRIEVAL_QUERIES non-empty query strings.
        """
        queries: list[str] = [expanded_thesis]
        thesis_lower       = thesis.lower()
        expanded_lower     = expanded_thesis.lower()

        # ── Q2: Subject + financial metric (corporate-focused) ────────────────
        # Detect company names
        found_companies = [
            c for c in KNOWN_COMPANIES if c.lower() in thesis_lower
        ]
        # Detect financial metrics
        metric_signals = [
            "nim", "net interest margin", "gnpa", "npa", "credit growth",
            "pat", "revenue", "ebitda", "margin", "casa", "book value",
            "roe", "roa", "capital adequacy", "crar", "lcr", "nim",
            "earnings", "capex", "attrition", "wage",
        ]
        found_metrics = [m for m in metric_signals if m in expanded_lower][:3]

        if found_companies and found_metrics:
            q2 = f"{' '.join(found_companies[:2])} {' '.join(found_metrics[:2])}"
            queries.append(q2)
        elif found_companies:
            queries.append(f"{' '.join(found_companies[:2])} financial results")
        elif found_metrics:
            queries.append(f"Indian banking {' '.join(found_metrics[:2])}")

        # ── Q3: Regulatory / macro context ────────────────────────────────────
        regulatory_signals = [
            "rbi", "reserve bank", "sebi", "repo rate", "inflation",
            "monetary policy", "mpc", "lcr", "crr", "slr", "regulatory",
            "mclr", "interest rate",
        ]
        found_regulatory = [r for r in regulatory_signals if r in expanded_lower][:2]

        if found_regulatory:
            q3 = f"{' '.join(found_regulatory)} India FY26"
            queries.append(q3)
        else:
            # Generic macro context for any thesis
            queries.append("India macroeconomic outlook FY26 interest rate inflation")

        # ── Q4: Valuation / historical analog ─────────────────────────────────
        valuation_signals = [
            "book value", "p/b", "pe ratio", "p/e", "price target",
            "valuation", "multiple", "re-rate", "fair value", "historical",
        ]
        found_valuation = [v for v in valuation_signals if v in expanded_lower][:2]

        if found_valuation and found_companies:
            q4 = (
                f"{found_companies[0]} "
                f"{' '.join(found_valuation[:1])} historical"
            )
            queries.append(q4)
        elif found_companies:
            queries.append(f"{found_companies[0]} historical performance analyst")

        # ── Deduplicate + cap ─────────────────────────────────────────────────
        seen:   set[str]  = set()
        final:  list[str] = []
        for q in queries:
            q_clean = q.strip()
            if q_clean and q_clean.lower() not in seen:
                seen.add(q_clean.lower())
                final.append(q_clean)
            if len(final) >= THESIS_RETRIEVAL_QUERIES:
                break

        logger.debug(
            f"[THESIS PIPELINE] Retrieval queries built: {final}"
        )
        return final if final else [expanded_thesis]

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Debug trace recorder
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _record_trace(
        state:          _ThesisPipelineState,
        agent_name:     str,
        elapsed_ms:     float,
        success:        bool,
        input_summary:  Optional[str] = None,
        output_summary: Optional[str] = None,
        error:          Optional[str] = None,
    ) -> None:
        """
        Appends an AgentTrace to state.agent_traces.
        Only populated when settings.debug_mode=True to avoid
        payload bloat in production responses.
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
                total_runs:           int   — total pipeline executions
                total_ms:             float — cumulative latency
                avg_ms_per_run:       float — mean latency per run
                rejected:             int   — rejected at validation
                agent_failures:       dict  — per-agent failure counts
                latency_target_ms:    int   — configured SLA target
                retrieval_queries:    int   — queries built per thesis
                top_k_final:          int   — chunks passed to agents
        """
        return {
            "total_runs":        self._total_runs,
            "total_ms":          round(self._total_ms, 1),
            "avg_ms_per_run":    round(
                self._total_ms / self._total_runs, 1
            ) if self._total_runs else 0.0,
            "rejected":          self._rejected,
            "agent_failures":    dict(self._agent_failures),
            "latency_target_ms": LATENCY_TARGET_MS,
            "retrieval_queries": THESIS_RETRIEVAL_QUERIES,
            "top_k_final":       THESIS_TOP_K_FINAL,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON EXPORT
# ══════════════════════════════════════════════════════════════════════════════

thesis_pipeline = ThesisPipeline()

__all__ = [
    "thesis_pipeline",
    "ThesisPipeline",
    "PIPELINE_NAME",
    "LATENCY_TARGET_MS",
    "ValidationResult",
    "_ThesisPipelineState",
]
