# backend/agents/agent7_stress_synthesizer.py
"""
FinThesisGuard AI — Agent 7: Stress Synthesizer
ArkAngel Financial Solutions

Consumes Agent 6 thesis_data + retrieved corpus chunks to produce a
complete, stress-tested ThesisResponse.

Pipeline position:
    Agent 6 (Thesis Analyzer) → Agent 7 (Stress Synthesizer) → ThesisResponse

What this agent produces (fills the remaining ThesisResponse fields):
    - risks:              List[RiskScore]    — 6 dimensions, each 1–10
    - break_conditions:   List[BreakCondition] — measurable invalidation events
    - thesis_strength:    ThesisStrength     — Strong / Medium / Weak
    - confidence:         ConfidenceLevel    — High / Medium / Low
    - synthesis:          str                — final prose paragraph
    - citations:          List[Citation]     — corpus sources used

Latency budget (target 700ms):
    ┌───────────────────────────────────────────────┐
    │ Phase 1 — Parallel (asyncio.gather)           │
    │   score_risk_dimensions   6× Groq  ≈ 350ms   │
    │   identify_break_conditions 1× Groq ≈ 300ms  │
    ├───────────────────────────────────────────────┤
    │ Phase 2 — Sequential (needs Phase 1 output)   │
    │   generate_synthesis      1× Groq  ≈ 200ms   │
    │   calculate_thesis_strength  pure  <   1ms   │
    │   assign_confidence          pure  <   1ms   │
    │   _build_citations           pure  <   1ms   │
    └───────────────────────────────────────────────┘
    Total ≈ 550ms (Phase 1 parallel) + 200ms (Phase 2) = ~550ms ✅

Depends on:
    backend/agents/agent6_thesis_analyzer.py  (upstream)
    backend/utils/llm_client.py
    backend/models/thesis_models.py
    backend/models/response_models.py
    backend/retrieval/retriever.py
    backend/utils/logger.py
    backend/config.py
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.config import (
    AGENT_LATENCY_TARGETS,
    AUTHORITY_WEIGHTS,
    LLM_MAX_TOKENS_SYNTHESIS,
    LLM_MAX_TOKENS_THESIS,
    LLM_TEMPERATURE_CREATIVE,
    LLM_TEMPERATURE_THESIS,
    THESIS_MAX_BREAK_CONDITIONS,
    THESIS_RISK_DIMENSIONS,
    THESIS_STRENGTH_THRESHOLDS,
)
from backend.models.response_models import Citation, ConfidenceLevel
from backend.models.thesis_models import (
    Assumption,
    BreakCondition,
    BreakProbability,
    DependencyNode,
    HistoricalAnalog,
    RiskScore,
    StructuralRobustness,
    ThesisResponse,
    ThesisStrength,
)
from backend.retrieval.retriever import RetrieverError, retriever
from backend.utils.llm_client import LLMError, llm_client
from backend.utils.logger import log_metric, logger

# ── Agent identity ─────────────────────────────────────────────────────────────
AGENT_NAME       = "agent7_stress_synthesizer"
LATENCY_TARGET_MS = int(AGENT_LATENCY_TARGETS.get(AGENT_NAME, 700))

# ── Per-dimension retrieval: chunk count per scoring call ─────────────────────
CHUNKS_PER_DIMENSION = 4

# ── Minimum chunks before we trust heuristic over LLM for risk scoring ────────
MIN_CHUNKS_FOR_LLM_RISK = 1

# ── Risk dimension metadata ────────────────────────────────────────────────────
# Maps dimension key → label, description, related assumption categories,
# and keyword signals for chunk filtering.
_DIMENSION_META: dict[str, dict] = {
    "demand_risk": {
        "label":       "Demand Risk",
        "description": "How fragile is the demand / revenue growth assumption?",
        "categories":  ["demand", "competitive"],
        "keywords":    [
            "revenue", "volume", "demand", "growth", "sales", "market share",
            "customer", "order", "shipment", "unit", "yoy", "cagr",
        ],
    },
    "margin_risk": {
        "label":       "Margin Risk",
        "description": "How sensitive is the margin / profitability assumption?",
        "categories":  ["margin", "operational"],
        "keywords":    [
            "nim", "ebitda", "margin", "profitability", "cost", "wage",
            "attrition", "operating leverage", "pat", "gross margin",
            "mclr", "repricing", "spread", "yield",
        ],
    },
    "valuation_risk": {
        "label":       "Valuation Risk",
        "description": "How stretched is the current valuation assumption?",
        "categories":  ["valuation"],
        "keywords":    [
            "pe", "p/b", "book value", "valuation", "multiple", "re-rate",
            "price target", "fair value", "dcf", "ev/ebitda", "premium",
        ],
    },
    "regulatory_risk": {
        "label":       "Regulatory Risk",
        "description": "Any SEBI / RBI / MCA regulatory risks?",
        "categories":  ["regulatory"],
        "keywords":    [
            "rbi", "sebi", "mca", "regulatory", "regulation", "circular",
            "lcr", "crr", "slr", "norms", "compliance", "penalty",
            "licence", "approval", "restriction",
        ],
    },
    "macro_risk": {
        "label":       "Macro Risk",
        "description": "Interest rate / inflation / GDP / FX exposure",
        "categories":  ["macro", "financial"],
        "keywords":    [
            "repo rate", "interest rate", "inflation", "gdp", "cpi", "wpi",
            "fx", "dollar", "currency", "global", "fed", "mpc", "monetary",
            "liquidity", "credit growth",
        ],
    },
    "competitive_risk": {
        "label":       "Competitive Risk",
        "description": "Competitive moat / market share / disruption risk",
        "categories":  ["competitive"],
        "keywords":    [
            "competition", "competitor", "market share", "moat", "pricing power",
            "disruption", "new entrant", "substitute", "consolidation", "jio",
            "platform", "switching cost",
        ],
    },
}

# ── Weighted scoring for thesis_strength calculation ─────────────────────────
# Higher weight = dimension matters more to overall thesis health.
_DIMENSION_WEIGHTS: dict[str, float] = {
    "demand_risk":      1.3,
    "margin_risk":      1.2,
    "valuation_risk":   1.1,
    "regulatory_risk":  1.0,
    "macro_risk":       1.1,
    "competitive_risk": 1.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _StressAnalysis:
    """
    Intermediate result produced during synthesize().
    Holds all scored components before ThesisResponse assembly.
    """
    risks:              list[RiskScore]      = field(default_factory=list)
    break_conditions:   list[BreakCondition] = field(default_factory=list)
    thesis_strength:    ThesisStrength       = "Weak"
    confidence:         ConfidenceLevel      = "Low"
    synthesis:          str                  = ""
    citations:          list[Citation]       = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class StressSynthesizer:
    """
    Agent 7 — Investment Thesis Stress Synthesizer.

    Usage:
        from backend.agents.agent7_stress_synthesizer import stress_synthesizer

        response: ThesisResponse = await stress_synthesizer.synthesize(
            thesis_text  = raw_thesis_text,
            thesis_data  = agent6_result,
            chunks       = reranked_chunks,
            request_id   = "req_abc123",
        )
    """

    def __init__(self) -> None:
        self._total_runs:    int   = 0
        self._total_ms:      float = 0.0
        self._llm_failures:  int   = 0
        self._heuristic_fallbacks: int = 0

        logger.info(
            f"[STRESS SYNTHESIZER] Agent 7 ready | "
            f"dimensions={len(THESIS_RISK_DIMENSIONS)} | "
            f"latency_target={LATENCY_TARGET_MS}ms | "
            f"max_break_conditions={THESIS_MAX_BREAK_CONDITIONS}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def synthesize(
        self,
        thesis_text:       str,
        thesis_data:       dict,
        chunks:            list[dict],
        quantitative_flags: Optional[list] = None,
        request_id:        Optional[str]   = None,
    ) -> ThesisResponse:
        """
        Orchestrates the full stress-synthesis pipeline.

        Pipeline:
            Phase 1 (parallel):
                score_risk_dimensions()    — 6× Groq calls
                identify_break_conditions() — 1× Groq call
            Phase 2 (sequential, needs Phase 1):
                generate_synthesis()        — 1× Groq call (text)
                calculate_thesis_strength() — pure computation
                assign_confidence()         — heuristic
                _build_citations()          — pure, from chunks

        Args:
            thesis_text:        Raw investment thesis text from user.
            thesis_data:        Output dict from Agent 6 (extract_assumptions +
                                build_dependency_chain + find_historical_analogs).
            chunks:             Reranked corpus chunks from upstream retrieval.
            quantitative_flags: Optional Agent 5 quantitative flags.
            request_id:         Optional request ID for tracing.

        Returns:
            ThesisResponse — fully assembled, Pydantic-validated.
            Falls back to ThesisResponse.error_response() on unexpected errors.
        """
        start_ms = time.perf_counter()
        self._total_runs += 1

        # Guard: empty thesis or missing Agent 6 data
        if not thesis_text or not thesis_text.strip():
            return ThesisResponse.error_response(
                thesis_text   = thesis_text or "",
                error_message = "Thesis text is empty.",
                request_id    = request_id,
                latency_ms    = 0,
            )

        assumptions: list[Assumption]      = thesis_data.get("assumptions", [])
        dependency_chain: list[str]        = thesis_data.get("dependency_chain", [])
        dependency_nodes: list[DependencyNode] = thesis_data.get("dependency_nodes", [])
        analogs: list[HistoricalAnalog]    = thesis_data.get("historical_analogs", [])
        structural_robustness: StructuralRobustness = thesis_data.get(
            "structural_robustness", "Low"
        )
        has_circular:  bool                = thesis_data.get("has_circular_dependency", False)
        upstream_agents: list[str]         = thesis_data.get("agents_used", [])

        logger.info(
            f"[STRESS SYNTHESIZER] Synthesize start | "
            f"thesis_len={len(thesis_text)} | assumptions={len(assumptions)} | "
            f"chunks={len(chunks)} | circular={has_circular}"
        )

        try:
            # ── Phase 1: Parallel scoring ─────────────────────────────────────
            risk_task  = self.score_risk_dimensions(assumptions, thesis_text, chunks)
            break_task = self.identify_break_conditions(assumptions, thesis_text, chunks)

            risks, break_conditions = await asyncio.gather(
                risk_task, break_task, return_exceptions=False
            )

            # ── Phase 2: Sequential — needs Phase 1 output ────────────────────
            thesis_strength = self.calculate_thesis_strength(risks)
            confidence      = self.assign_confidence(assumptions, risks)
            synthesis       = await self.generate_synthesis(
                thesis_text       = thesis_text,
                assumptions       = assumptions,
                risks             = risks,
                break_conditions  = break_conditions,
                analogs           = analogs,
                thesis_strength   = thesis_strength,
                confidence        = confidence,
                structural_robustness = structural_robustness,
            )
            citations = self._build_citations(chunks)

            duration_ms = (time.perf_counter() - start_ms) * 1000
            self._total_ms += duration_ms

            response = ThesisResponse(
                thesis_text           = thesis_text,
                thesis_strength       = thesis_strength,
                structural_robustness = structural_robustness,
                confidence            = confidence,
                assumptions           = assumptions,
                dependency_chain      = dependency_chain,
                dependency_nodes      = dependency_nodes,
                quantitative_flags    = quantitative_flags or [],
                risks                 = risks,
                break_conditions      = break_conditions,
                historical_analogs    = analogs,
                synthesis             = synthesis,
                citations             = citations,
                latency_ms            = int(duration_ms),
                agents_used           = [*upstream_agents, AGENT_NAME],
                request_id            = request_id,
            )

            logger.info(
                f"[STRESS SYNTHESIZER] Complete | "
                f"strength={thesis_strength} | confidence={confidence} | "
                f"risks={len(risks)} | breaks={len(break_conditions)} | "
                f"citations={len(citations)} | {duration_ms:.0f}ms"
            )
            log_metric("stress_synthesizer_duration_ms", duration_ms, unit="ms")
            log_metric("stress_synthesizer_risks",        len(risks),   unit="count")
            log_metric("stress_synthesizer_breaks",       len(break_conditions), unit="count")

            if duration_ms > LATENCY_TARGET_MS:
                logger.warning(
                    f"[STRESS SYNTHESIZER] Latency target breached | "
                    f"{duration_ms:.0f}ms > {LATENCY_TARGET_MS}ms"
                )

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_ms) * 1000
            self._total_ms += duration_ms
            logger.error(
                f"[STRESS SYNTHESIZER] Unexpected error: "
                f"{type(e).__name__}: {e} | {duration_ms:.0f}ms"
            )
            return ThesisResponse.error_response(
                thesis_text   = thesis_text,
                error_message = f"{type(e).__name__}: {e}",
                request_id    = request_id,
                latency_ms    = int(duration_ms),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: score_risk_dimensions
    # ──────────────────────────────────────────────────────────────────────────

    async def score_risk_dimensions(
        self,
        assumptions: list[Assumption],
        thesis_text: str,
        chunks:      list[dict],
    ) -> list[RiskScore]:
        """
        Scores all 6 risk dimensions in parallel via Groq.

        Strategy:
            - For each dimension, filter chunks to the most relevant N.
            - Run 6 scoring calls concurrently with asyncio.gather.
            - On LLM failure for any dimension, fall back to heuristic score.
            - Always returns exactly 6 RiskScore objects (one per dimension).

        Args:
            assumptions: Extracted thesis assumptions (from Agent 6).
            thesis_text: Original investment thesis text.
            chunks:      Reranked corpus chunks.

        Returns:
            List[RiskScore] — 6 items, one per THESIS_RISK_DIMENSIONS entry.
        """
        tasks = [
            self._score_single_dimension(
                dimension   = dim,
                assumptions = assumptions,
                thesis_text = thesis_text,
                chunks      = self._filter_chunks_for_dimension(dim, chunks),
            )
            for dim in THESIS_RISK_DIMENSIONS
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        risk_scores: list[RiskScore] = []
        for dim, result in zip(THESIS_RISK_DIMENSIONS, results):
            if isinstance(result, Exception):
                logger.warning(
                    f"[STRESS SYNTHESIZER] score_risk_dimensions: "
                    f"{dim} failed ({type(result).__name__}: {result}). "
                    f"Using heuristic fallback."
                )
                risk_scores.append(
                    self._heuristic_risk_score(dim, assumptions)
                )
            else:
                risk_scores.append(result)

        logger.debug(
            f"[STRESS SYNTHESIZER] Risk scoring complete | "
            f"scores={[r.score for r in risk_scores]} | "
            f"avg={sum(r.score for r in risk_scores)/len(risk_scores):.1f}"
        )
        return risk_scores

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: identify_break_conditions
    # ──────────────────────────────────────────────────────────────────────────

    async def identify_break_conditions(
        self,
        assumptions: list[Assumption],
        thesis_text: str,
        chunks:      list[dict],
    ) -> list[BreakCondition]:
        """
        Uses Groq to identify specific, measurable conditions that would
        invalidate the investment thesis.

        Prompt design:
            - Provides full thesis + assumption list + top corpus chunks.
            - Asks model to produce THESIS_MAX_BREAK_CONDITIONS conditions.
            - Each condition: {condition, trigger_metric, threshold,
              probability, monitoring_frequency, related_assumption_ids}.
            - Post-processing validates Pydantic + checks if already triggered.

        Args:
            assumptions: Extracted thesis assumptions.
            thesis_text: Original thesis.
            chunks:      Corpus chunks for grounding.

        Returns:
            List[BreakCondition] — at most THESIS_MAX_BREAK_CONDITIONS items.
            Returns [] on LLM failure (non-fatal, synthesis still proceeds).
        """
        if not assumptions:
            return []

        # Use highest-scoring chunks as grounding context
        top_chunks = sorted(
            chunks, key=lambda c: c.get("final_score", 0.0), reverse=True
        )[:6]

        prompt = self._build_break_conditions_prompt(
            thesis_text = thesis_text,
            assumptions = assumptions,
            chunks      = top_chunks,
        )

        try:
            raw = await llm_client.call_thesis_json(
                prompt      = prompt,
                agent_name  = AGENT_NAME,
                max_tokens  = LLM_MAX_TOKENS_THESIS,
            )
        except Exception as e:
            self._llm_failures += 1
            logger.warning(
                f"[STRESS SYNTHESIZER] identify_break_conditions LLM failed: "
                f"{type(e).__name__}: {e}. Returning heuristic break conditions."
            )
            return self._heuristic_break_conditions(assumptions)

        # Unwrap if LLM returned {"conditions": [...]} or {"break_conditions": [...]}
        items = raw
        if isinstance(raw, dict):
            items = (
                raw.get("conditions")
                or raw.get("break_conditions")
                or raw.get("items")
                or raw.get("data")
                or []
            )

        if not isinstance(items, list):
            logger.warning(
                f"[STRESS SYNTHESIZER] identify_break_conditions: "
                f"LLM returned unexpected type {type(raw)}. "
                f"Returning heuristic break conditions."
            )
            return self._heuristic_break_conditions(assumptions)

        valid_ids = {a.id for a in assumptions}
        break_conditions: list[BreakCondition] = []

        for item in items[:THESIS_MAX_BREAK_CONDITIONS]:
            if not isinstance(item, dict):
                continue
            try:
                prob_raw = str(item.get("probability", "Medium")).strip().title()
                if prob_raw not in ("High", "Medium", "Low"):
                    prob_raw = "Medium"

                # Check if triggered: current corpus already shows this condition
                condition_text = str(item.get("condition", "")).strip()
                triggered = self._check_if_triggered(condition_text, chunks)

                related_ids = [
                    int(x) for x in item.get("related_assumption_ids", [])
                    if str(x).isdigit() and int(x) in valid_ids
                ]

                bc = BreakCondition(
                    condition            = condition_text[:500] or "Thesis invalidation condition",
                    trigger_metric       = str(item.get("trigger_metric", ""))[:200].strip()
                                          or "Monitor closely",
                    threshold            = str(item.get("threshold", ""))[:200].strip()
                                          or "Below acceptable range",
                    probability          = prob_raw,
                    data_source          = str(item.get("data_source", "corpus"))[:200],
                    monitoring_frequency = str(item.get("monitoring_frequency", "Quarterly"))[:100],
                    triggered            = triggered,
                    related_assumption_ids = related_ids,
                )
                break_conditions.append(bc)

            except (ValueError, TypeError, KeyError) as e:
                logger.debug(
                    f"[STRESS SYNTHESIZER] Skipping malformed break condition: "
                    f"{item} | {e}"
                )
                continue

        # Sort: triggered first, then by probability (High → Medium → Low)
        _prob_order = {"High": 0, "Medium": 1, "Low": 2}
        break_conditions.sort(
            key=lambda b: (not b.triggered, _prob_order.get(b.probability, 3))
        )

        logger.debug(
            f"[STRESS SYNTHESIZER] Break conditions: {len(break_conditions)} | "
            f"triggered={sum(1 for b in break_conditions if b.triggered)} | "
            f"high_prob={sum(1 for b in break_conditions if b.probability == 'High')}"
        )
        return break_conditions

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: calculate_thesis_strength
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_thesis_strength(
        self,
        risks: list[RiskScore],
    ) -> ThesisStrength:
        """
        Computes ThesisStrength from a weighted average of risk scores.

        Weighting logic:
            Each dimension has a weight in _DIMENSION_WEIGHTS (1.0–1.3).
            Higher-weight dimensions (demand, margin) matter more.

        Thresholds (from config.THESIS_STRENGTH_THRESHOLDS):
            Strong: weighted_avg < 4.0
            Medium: 4.0 ≤ weighted_avg ≤ 7.0
            Weak:   weighted_avg > 7.0

        Args:
            risks: List of RiskScore from score_risk_dimensions().

        Returns:
            "Strong" | "Medium" | "Weak"
        """
        if not risks:
            return "Weak"

        strong_threshold = THESIS_STRENGTH_THRESHOLDS.get("strong", 4.0)
        medium_threshold = THESIS_STRENGTH_THRESHOLDS.get("medium", 7.0)

        total_weight = 0.0
        weighted_sum = 0.0

        for risk in risks:
            weight = _DIMENSION_WEIGHTS.get(risk.dimension, 1.0)
            weighted_sum += risk.score * weight
            total_weight += weight

        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 5.0

        if weighted_avg < strong_threshold:
            return "Strong"
        elif weighted_avg <= medium_threshold:
            return "Medium"
        return "Weak"

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: assign_confidence
    # ──────────────────────────────────────────────────────────────────────────

    def assign_confidence(
        self,
        assumptions: list[Assumption],
        risks:       list[RiskScore],
    ) -> ConfidenceLevel:
        """
        Assigns a ConfidenceLevel based on corpus coverage + risk profile.

        Decision matrix:
            High:   support_rate ≥ 70%  AND  avg_risk < 5.0
            Low:    support_rate < 30%  OR   avg_risk > 7.0  OR
                    no assumptions extracted
            Medium: all other cases

        Args:
            assumptions: List[Assumption] with historical_support populated.
            risks:       List[RiskScore] from score_risk_dimensions().

        Returns:
            "High" | "Medium" | "Low"
        """
        if not assumptions:
            return "Low"

        supported_count = sum(1 for a in assumptions if a.historical_support)
        support_rate    = supported_count / len(assumptions)
        avg_risk        = (
            sum(r.score for r in risks) / len(risks) if risks else 7.0
        )
        critical_unsupported = sum(
            1 for a in assumptions
            if a.is_critical and not a.historical_support
        )

        logger.debug(
            f"[STRESS SYNTHESIZER] assign_confidence | "
            f"support_rate={support_rate:.1%} | avg_risk={avg_risk:.1f} | "
            f"critical_unsupported={critical_unsupported}"
        )

        # Hard Low conditions
        if support_rate < 0.30 or avg_risk > 7.0 or critical_unsupported >= 2:
            return "Low"

        # High condition
        if support_rate >= 0.70 and avg_risk < 5.0 and critical_unsupported == 0:
            return "High"

        return "Medium"

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: generate_synthesis
    # ──────────────────────────────────────────────────────────────────────────

    async def generate_synthesis(
        self,
        thesis_text:           str,
        assumptions:           list[Assumption],
        risks:                 list[RiskScore],
        break_conditions:      list[BreakCondition],
        analogs:               list[HistoricalAnalog],
        thesis_strength:       ThesisStrength,
        confidence:            ConfidenceLevel,
        structural_robustness: StructuralRobustness,
    ) -> str:
        """
        Generates the final synthesis paragraph using Groq (text mode, T=0.3).

        Synthesis structure:
            1. Verdict sentence (strength + confidence)
            2. Most critical assumption + its support status
            3. Highest-scoring risk dimension with rationale excerpt
            4. Most dangerous break condition (if any)
            5. Historical analog lesson (if available)
            6. Closing hedge / monitoring note

        Falls back to a deterministic heuristic paragraph if Groq fails.

        Args:
            thesis_text:           Original thesis.
            assumptions:           Extracted assumptions.
            risks:                 Scored risk dimensions.
            break_conditions:      Identified break conditions.
            analogs:               Historical analogs.
            thesis_strength:       Computed thesis strength.
            confidence:            Computed confidence.
            structural_robustness: From Agent 6.

        Returns:
            str — synthesis paragraph (100–350 words).
        """
        prompt = self._build_synthesis_prompt(
            thesis_text           = thesis_text,
            assumptions           = assumptions,
            risks                 = risks,
            break_conditions      = break_conditions,
            analogs               = analogs,
            thesis_strength       = thesis_strength,
            confidence            = confidence,
            structural_robustness = structural_robustness,
        )

        try:
            synthesis = await llm_client.call_thesis(
                prompt      = prompt,
                agent_name  = AGENT_NAME,
                max_tokens  = LLM_MAX_TOKENS_SYNTHESIS,
            )
            # Strip markdown artifacts if model wraps in code fences
            synthesis = synthesis.strip().strip("`").strip()
            if synthesis.lower().startswith("synthesis"):
                synthesis = synthesis[len("synthesis"):].lstrip(":").strip()
            return synthesis

        except Exception as e:
            self._llm_failures += 1
            logger.warning(
                f"[STRESS SYNTHESIZER] generate_synthesis LLM failed: "
                f"{type(e).__name__}: {e}. Using heuristic synthesis."
            )
            return self._heuristic_synthesis(
                thesis_text       = thesis_text,
                assumptions       = assumptions,
                risks             = risks,
                break_conditions  = break_conditions,
                thesis_strength   = thesis_strength,
                confidence        = confidence,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: _score_single_dimension
    # ──────────────────────────────────────────────────────────────────────────

    async def _score_single_dimension(
        self,
        dimension:   str,
        assumptions: list[Assumption],
        thesis_text: str,
        chunks:      list[dict],
    ) -> RiskScore:
        """
        Scores a single risk dimension using Groq.

        Falls back to _heuristic_risk_score() on any LLM failure.
        Never raises — always returns a RiskScore.

        Args:
            dimension:   One of THESIS_RISK_DIMENSIONS.
            assumptions: All extracted assumptions.
            thesis_text: Original thesis.
            chunks:      Pre-filtered chunks for this dimension.

        Returns:
            RiskScore with score 1–10, rationale, supporting_data.
        """
        meta = _DIMENSION_META.get(dimension, {})
        label = meta.get("label", dimension.replace("_", " ").title())

        # Assumptions relevant to this dimension
        related_cats = meta.get("categories", [])
        related_assumptions = [
            a for a in assumptions if a.category in related_cats
        ] or assumptions  # Fall back to all if no categorical match

        prompt = self._build_risk_scoring_prompt(
            dimension            = dimension,
            dimension_label      = label,
            dimension_description = meta.get("description", ""),
            thesis_text          = thesis_text,
            assumptions          = related_assumptions,
            chunks               = chunks,
        )

        try:
            if len(chunks) < MIN_CHUNKS_FOR_LLM_RISK:
                # No evidence context — use heuristic
                raise LLMError(
                    message     = "Insufficient chunks for LLM risk scoring",
                    agent_name  = AGENT_NAME,
                )

            raw = await llm_client.call_thesis_json(
                prompt      = prompt,
                agent_name  = AGENT_NAME,
                max_tokens  = 800,
            )

            # Unwrap if LLM wrapped in outer dict
            if isinstance(raw, dict) and "score" not in raw:
                raw = (
                    raw.get(dimension)
                    or raw.get("risk_score")
                    or raw.get("result")
                    or raw
                )

            score_raw = raw.get("score", 5)
            score     = max(1, min(10, int(score_raw)))

            rationale = str(raw.get("rationale", "")).strip()
            if len(rationale) < 20:
                rationale = (
                    f"{label} scored {score}/10 based on available corpus evidence."
                )

            supporting_data = [
                str(d)[:300] for d in raw.get("supporting_data", [])
                if str(d).strip()
            ][:5]

            valid_ids = {a.id for a in assumptions}
            related_ids = [
                int(x) for x in raw.get("related_assumption_ids", [])
                if str(x).isdigit() and int(x) in valid_ids
            ]

            logger.debug(
                f"[STRESS SYNTHESIZER] {dimension}: score={score} | "
                f"evidence_chunks={len(chunks)} | "
                f"related_assumptions={len(related_ids)}"
            )

            return RiskScore(
                dimension              = dimension,
                score                  = score,
                rationale              = rationale[:1000],
                supporting_data        = supporting_data,
                related_assumption_ids = related_ids,
            )

        except Exception as e:
            self._llm_failures    += 1
            self._heuristic_fallbacks += 1
            logger.warning(
                f"[STRESS SYNTHESIZER] _score_single_dimension({dimension}) failed: "
                f"{type(e).__name__}: {e}. Using heuristic."
            )
            return self._heuristic_risk_score(dimension, assumptions)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Prompt builders
    # ──────────────────────────────────────────────────────────────────────────

    def _build_risk_scoring_prompt(
        self,
        dimension:             str,
        dimension_label:       str,
        dimension_description: str,
        thesis_text:           str,
        assumptions:           list[Assumption],
        chunks:                list[dict],
    ) -> str:
        """Constructs the Groq prompt for scoring a single risk dimension."""

        assumption_lines = "\n".join(
            f"  [{a.id}] [{a.category}] conf={a.confidence:.2f} "
            f"{'(CRITICAL)' if a.is_critical else ''} "
            f"{'(SUPPORTED)' if a.historical_support else '(UNSUPPORTED)'}: "
            f"{a.text}"
            for a in assumptions
        ) or "  No specific assumptions extracted."

        evidence_lines = "\n\n".join(
            f"  [Source {i+1}] [{chunk.get('source_type', 'unknown').upper()}] "
            f"{chunk.get('source', 'Unknown')} "
            f"(relevance={chunk.get('final_score', 0.0):.2f}):\n"
            f"  {chunk.get('text', '')[:400]}"
            for i, chunk in enumerate(chunks)
        ) or "  No relevant corpus evidence available for this dimension."

        return f"""You are a senior investment risk analyst specializing in Indian capital markets.

Score the {dimension_label} for the following investment thesis.
{dimension_description}

THESIS:
{thesis_text[:1500]}

RELEVANT ASSUMPTIONS:
{assumption_lines}

CORPUS EVIDENCE:
{evidence_lines}

INSTRUCTIONS:
1. Score {dimension_label} from 1 (minimal risk) to 10 (thesis-breaking risk).
2. Be specific to THIS thesis — cite numbers, percentages, and named entities.
3. If corpus evidence contradicts a key assumption, increase the score.
4. If evidence supports a key assumption, decrease the score.
5. Mandatory fields: score, rationale (60–200 words), supporting_data (1–3 data points).

Return ONLY valid JSON. No explanation outside JSON:
{{
  "score": <int 1-10>,
  "rationale": "<60-200 word explanation citing specific evidence>",
  "supporting_data": ["<specific data point 1>", "<specific data point 2>"],
  "related_assumption_ids": [<int>, ...]
}}"""

    def _build_break_conditions_prompt(
        self,
        thesis_text: str,
        assumptions: list[Assumption],
        chunks:      list[dict],
    ) -> str:
        """Constructs the Groq prompt for identifying thesis break conditions."""

        assumption_lines = "\n".join(
            f"  [{a.id}] [{a.category}] {'(CRITICAL)' if a.is_critical else ''}: {a.text}"
            for a in assumptions
        )

        evidence_lines = "\n\n".join(
            f"  [{chunk.get('source_type', 'unknown').upper()}] "
            f"{chunk.get('source', 'Unknown')}:\n  {chunk.get('text', '')[:300]}"
            for chunk in chunks[:4]
        ) or "  No corpus evidence available."

        return f"""You are a stress-testing expert for investment theses in Indian equities.

Identify the top {THESIS_MAX_BREAK_CONDITIONS} SPECIFIC, MEASURABLE conditions \
that would INVALIDATE this thesis.
Each condition must be actionable — an investor must be able to monitor it.

THESIS:
{thesis_text[:1500]}

ASSUMPTIONS THE THESIS RESTS ON:
{assumption_lines}

CURRENT CORPUS EVIDENCE (for grounding probability assessments):
{evidence_lines}

INSTRUCTIONS:
1. Each condition must have a specific, measurable trigger metric.
2. Threshold must be a concrete number or event (not vague language).
3. Probability: "High" if already showing early signs, "Medium" if plausible,
   "Low" if tail-risk only.
4. Flag triggered=true if current corpus data already shows this condition.
5. monitoring_frequency: "Monthly", "Quarterly", "Real-time", etc.

Return ONLY valid JSON array:
[
  {{
    "condition": "<what specific event breaks the thesis>",
    "trigger_metric": "<measurable metric to monitor>",
    "threshold": "<specific level that breaks the thesis>",
    "probability": "High"|"Medium"|"Low",
    "monitoring_frequency": "<how often to check>",
    "data_source": "<source of probability assessment>",
    "triggered": false,
    "related_assumption_ids": [<int>, ...]
  }},
  ...
]"""

    def _build_synthesis_prompt(
        self,
        thesis_text:           str,
        assumptions:           list[Assumption],
        risks:                 list[RiskScore],
        break_conditions:      list[BreakCondition],
        analogs:               list[HistoricalAnalog],
        thesis_strength:       ThesisStrength,
        confidence:            ConfidenceLevel,
        structural_robustness: StructuralRobustness,
    ) -> str:
        """Constructs the Groq prompt for the final synthesis paragraph."""

        # Identify most critical assumption
        critical_assumptions = [a for a in assumptions if a.is_critical]
        top_assumption = (
            min(critical_assumptions, key=lambda a: a.confidence)
            if critical_assumptions else (assumptions[0] if assumptions else None)
        )

        # Highest-risk dimension
        highest_risk = max(risks, key=lambda r: r.score) if risks else None

        # Most dangerous break condition
        top_break = (
            next((b for b in break_conditions if b.triggered), None)
            or next((b for b in break_conditions if b.probability == "High"), None)
            or (break_conditions[0] if break_conditions else None)
        )

        # Most relevant analog
        top_analog = analogs[0] if analogs else None

        support_rate = (
            round(sum(1 for a in assumptions if a.historical_support)
                  / len(assumptions) * 100, 1)
            if assumptions else 0.0
        )

        avg_risk = round(
            sum(r.score for r in risks) / len(risks), 1
        ) if risks else 0.0

        # Build risk summary table
        risk_summary = " | ".join(
            f"{_DIMENSION_META.get(r.dimension, {}).get('label', r.dimension)}: "
            f"{r.score}/10"
            for r in sorted(risks, key=lambda x: -x.score)
        ) if risks else "No risk scores"

        critical_asm_text = (
            f"[{top_assumption.id}] {top_assumption.text} "
            f"(conf={top_assumption.confidence:.2f}, "
            f"{'corpus-supported' if top_assumption.historical_support else 'NOT corpus-supported'})"
            if top_assumption else "None identified"
        )

        highest_risk_text = (
            f"{_DIMENSION_META.get(highest_risk.dimension, {}).get('label', highest_risk.dimension)}: "
            f"{highest_risk.score}/10 — {highest_risk.rationale[:200]}"
            if highest_risk else "None"
        )

        top_break_text = (
            f"{top_break.condition} "
            f"[trigger: {top_break.trigger_metric} {top_break.threshold}] "
            f"[probability: {top_break.probability}]"
            if top_break else "None identified"
        )

        analog_text = (
            f"{top_analog.title} ({top_analog.period}): {top_analog.lesson[:200]}"
            if top_analog else "No historical analog available"
        )

        return f"""You are FinThesisGuard AI by ArkAngel Financial Solutions.
Write a precise, honest investment thesis validation summary (150-300 words).

VALIDATION RESULTS:
  Thesis Strength:       {thesis_strength}
  Overall Confidence:    {confidence}
  Structural Robustness: {structural_robustness}
  Corpus Support Rate:   {support_rate}% of assumptions supported
  Avg Risk Score:        {avg_risk}/10

RISK DIMENSION SCORES:
  {risk_summary}

MOST CRITICAL ASSUMPTION:
  {critical_asm_text}

HIGHEST RISK:
  {highest_risk_text}

PRIMARY BREAK CONDITION:
  {top_break_text}

HISTORICAL ANALOG:
  {analog_text}

ORIGINAL THESIS (first 400 chars):
  {thesis_text[:400]}

INSTRUCTIONS:
1. Start with a direct verdict: "[Strength] thesis with [Confidence] confidence."
2. Identify the single most critical assumption and its evidential status.
3. Name the most dangerous risk dimension and why it matters here.
4. Describe the top break condition an investor must monitor.
5. End with the analog lesson if applicable, or a monitoring recommendation.
6. Be specific — cite numbers and named entities from the thesis.
7. Do NOT use bullet points. Write flowing prose. No markdown headers."""

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Chunk filtering
    # ──────────────────────────────────────────────────────────────────────────

    def _filter_chunks_for_dimension(
        self,
        dimension: str,
        chunks:    list[dict],
        top_n:     int = CHUNKS_PER_DIMENSION,
    ) -> list[dict]:
        """
        Returns the top-N chunks most relevant to a given risk dimension.

        Relevance score:
            base_score × (1.0 + keyword_overlap_bonus)
        where keyword_overlap_bonus = number of matching keywords / 3 (capped at 1.0).

        Falls back to top-N by final_score if no keyword matches.
        """
        keywords = [
            kw.lower()
            for kw in _DIMENSION_META.get(dimension, {}).get("keywords", [])
        ]

        if not keywords:
            return sorted(
                chunks, key=lambda c: c.get("final_score", 0.0), reverse=True
            )[:top_n]

        def _relevance(chunk: dict) -> float:
            text  = chunk.get("text", "").lower()
            base  = float(chunk.get("final_score", 0.5))
            hits  = sum(1 for kw in keywords if kw in text)
            bonus = min(hits / 3.0, 1.0)
            return base * (1.0 + bonus)

        return sorted(chunks, key=_relevance, reverse=True)[:top_n]

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Heuristic fallbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _heuristic_risk_score(
        self,
        dimension:   str,
        assumptions: list[Assumption],
    ) -> RiskScore:
        """
        Computes a heuristic RiskScore when LLM is unavailable.

        Algorithm:
            1. Find assumptions whose category matches this dimension.
            2. Base risk = round((1 - avg_confidence) × 9) + 1  (1–10)
            3. +2 penalty if any related critical assumption is unsupported
            4. Clamp to [1, 10]

        Always returns a valid RiskScore — never raises.
        """
        meta         = _DIMENSION_META.get(dimension, {})
        label        = meta.get("label", dimension.replace("_", " ").title())
        related_cats = meta.get("categories", [])

        related = [a for a in assumptions if a.category in related_cats]
        if not related:
            related = assumptions

        if related:
            avg_conf = sum(a.confidence for a in related) / len(related)
            base_risk = round((1.0 - avg_conf) * 9) + 1
        else:
            base_risk = 6  # Neutral default when no assumptions exist

        # Penalty: critical assumption with no support
        unsupported_critical = sum(
            1 for a in related if a.is_critical and not a.historical_support
        )
        base_risk += min(unsupported_critical * 2, 3)
        score = max(1, min(10, base_risk))

        return RiskScore(
            dimension   = dimension,
            score       = score,
            rationale   = (
                f"{label} score of {score}/10 derived from assumption confidence "
                f"analysis. "
                f"{'LLM scoring unavailable — heuristic used.' if True else ''} "
                f"Average confidence of related assumptions: "
                f"{sum(a.confidence for a in related)/len(related):.2f}"
                if related else f"No related assumptions found for {label}."
            ),
            supporting_data        = [],
            related_assumption_ids = [a.id for a in related],
        )

    def _heuristic_break_conditions(
        self,
        assumptions: list[Assumption],
    ) -> list[BreakCondition]:
        """
        Generates deterministic break conditions from critical assumptions
        when Groq is unavailable.

        Produces one BreakCondition per critical assumption, capped at
        THESIS_MAX_BREAK_CONDITIONS.
        """
        self._heuristic_fallbacks += 1
        conditions: list[BreakCondition] = []

        critical = [a for a in assumptions if a.is_critical]
        for assumption in critical[:THESIS_MAX_BREAK_CONDITIONS]:
            conditions.append(BreakCondition(
                condition   = (
                    f"Assumption fails: {assumption.text[:200]}"
                ),
                trigger_metric = f"Monitor: {assumption.category} indicators",
                threshold      = "Below minimum required level",
                probability    = "High" if assumption.confidence < 0.5 else "Medium",
                data_source    = "heuristic_fallback",
                monitoring_frequency = "Quarterly",
                triggered        = False,
                related_assumption_ids = [assumption.id],
            ))

        return conditions

    def _heuristic_synthesis(
        self,
        thesis_text:      str,
        assumptions:      list[Assumption],
        risks:            list[RiskScore],
        break_conditions: list[BreakCondition],
        thesis_strength:  ThesisStrength,
        confidence:       ConfidenceLevel,
    ) -> str:
        """
        Deterministic synthesis paragraph generated without LLM.
        Used as final fallback when generate_synthesis() Groq call fails.
        """
        avg_risk    = round(sum(r.score for r in risks) / len(risks), 1) if risks else 5.0
        highest_risk = max(risks, key=lambda r: r.score) if risks else None
        support_count = sum(1 for a in assumptions if a.historical_support)
        support_rate  = round(support_count / len(assumptions) * 100) if assumptions else 0
        top_break     = next(
            (b for b in break_conditions if b.probability == "High"), None
        ) or (break_conditions[0] if break_conditions else None)
        highest_label = (
            _DIMENSION_META.get(highest_risk.dimension, {}).get("label", "")
            if highest_risk else "overall risk"
        )

        synthesis = (
            f"This is a {thesis_strength.lower()} thesis with {confidence.lower()} "
            f"confidence in FinThesisGuard's validation. "
            f"Of {len(assumptions)} extracted assumptions, {support_count} "
            f"({support_rate}%) are supported by corpus evidence. "
        )

        if highest_risk:
            synthesis += (
                f"The highest risk dimension is {highest_label} "
                f"(score: {highest_risk.score}/10), "
                f"with an overall average risk of {avg_risk}/10. "
            )

        if top_break:
            synthesis += (
                f"The primary break condition to monitor is: "
                f"{top_break.condition[:150]}. "
                f"Trigger metric: {top_break.trigger_metric}. "
            )

        synthesis += (
            "Investors should track the identified break conditions "
            "on a quarterly basis and reassess position sizing if "
            "any high-probability condition shows early signs of triggering."
        )

        return synthesis

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Citation builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_citations(self, chunks: list[dict]) -> list[Citation]:
        """
        Converts corpus chunks into deduplicated Citation objects.

        Deduplication: by (source, source_type) pair — keeps the
        highest-scoring chunk when duplicates exist.

        Authority weighting (from config.AUTHORITY_WEIGHTS):
            rbi / sebi / annual_report → score boosted
            news / blog               → score penalised

        Returns up to 10 citations sorted by adjusted relevance score.
        """
        seen:     dict[str, dict] = {}   # key → best chunk
        for chunk in chunks:
            source      = (chunk.get("source")    or "Unknown Source").strip()
            source_type = (chunk.get("source_type") or "unknown").strip()
            key         = f"{source_type}::{source}"
            score       = float(chunk.get("final_score", 0.0))
            if key not in seen or score > seen[key].get("final_score", 0.0):
                seen[key] = chunk

        citations: list[Citation] = []
        for chunk in sorted(
            seen.values(),
            key=lambda c: float(c.get("final_score", 0.0)) *
                          AUTHORITY_WEIGHTS.get(
                              c.get("source_type", "unknown"), 1.0
                          ),
            reverse=True,
        )[:10]:
            source_type = chunk.get("source_type", "unknown")
            try:
                citations.append(Citation(
                    title          = chunk.get("source", "Unknown Source")[:200],
                    source         = chunk.get("source", "")[:200],
                    source_type    = source_type,
                    date           = chunk.get("date", chunk.get("metadata", {}).get("date", "")),
                    relevance_score = float(chunk.get("final_score", 0.0)),
                    snippet        = chunk.get("text", "")[:300],
                ))
            except (ValueError, TypeError) as e:
                logger.debug(
                    f"[STRESS SYNTHESIZER] Skipping malformed citation: {e}"
                )
                continue

        return citations

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Trigger detection
    # ──────────────────────────────────────────────────────────────────────────

    def _check_if_triggered(
        self,
        condition_text: str,
        chunks:         list[dict],
    ) -> bool:
        """
        Heuristically checks whether a break condition has already triggered
        based on current corpus evidence.

        Strategy:
            1. Extract numeric thresholds from condition_text.
            2. Search high-score chunks for the same metric at or past threshold.
            3. Returns True if ≥2 high-authority chunks corroborate.

        This is a lightweight signal — not a definitive trigger declaration.
        """
        if not condition_text or not chunks:
            return False

        condition_lower = condition_text.lower()

        # Extract key phrases: "below X%", "above X%", "below X bps", etc.
        trigger_patterns = re.findall(
            r"(?:below|above|exceeds?|falls?\s+below|rises?\s+above|"
            r"drops?\s+below|jumps?\s+above)\s+[\d.]+",
            condition_lower,
        )
        if not trigger_patterns:
            return False

        # Check top-scoring chunks for corroboration
        high_score_chunks = [
            c for c in chunks
            if float(c.get("final_score", 0.0)) >= 0.7
        ]

        # Extract key terms from condition (nouns + numbers)
        key_terms = [
            w for w in re.findall(r"[a-z]{4,}", condition_lower)
            if w not in {
                "will", "when", "that", "this", "from", "with",
                "have", "been", "than", "below", "above", "falls",
            }
        ][:5]

        corroborating = sum(
            1 for chunk in high_score_chunks
            if sum(1 for t in key_terms if t in chunk.get("text", "").lower()) >= 2
        )
        return corroborating >= 2

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Stats
    # ──────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Returns runtime statistics for monitoring and health checks.

        Returns:
            dict with keys:
                total_runs:            int   — number of synthesize() calls
                total_ms:              float — cumulative latency
                avg_ms_per_run:        float — mean latency per run
                llm_failures:          int   — total LLM call failures
                heuristic_fallbacks:   int   — fallbacks to heuristic scoring
                risk_dimensions:       int   — number of scored dimensions
                max_break_conditions:  int   — configured max
        """
        return {
            "total_runs":           self._total_runs,
            "total_ms":             round(self._total_ms, 1),
            "avg_ms_per_run":       round(
                self._total_ms / self._total_runs, 1
            ) if self._total_runs else 0.0,
            "llm_failures":         self._llm_failures,
            "heuristic_fallbacks":  self._heuristic_fallbacks,
            "risk_dimensions":      len(THESIS_RISK_DIMENSIONS),
            "max_break_conditions": THESIS_MAX_BREAK_CONDITIONS,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON EXPORT
# ══════════════════════════════════════════════════════════════════════════════

stress_synthesizer = StressSynthesizer()

__all__ = [
    "stress_synthesizer",
    "StressSynthesizer",
    "AGENT_NAME",
    "LATENCY_TARGET_MS",
    "_DIMENSION_META",
    "_DIMENSION_WEIGHTS",
    "_StressAnalysis",
]
