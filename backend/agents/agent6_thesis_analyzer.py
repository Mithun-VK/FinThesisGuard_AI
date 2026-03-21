# backend/agents/agent6_thesis_analyzer.py
"""
FinThesisGuard AI — Agent 6: Thesis Analyzer
ArkAngel Financial Solutions

Decomposes an investment thesis into structured, verifiable assumptions,
builds a causal dependency graph, finds historical analogs from the corpus,
and assesses each assumption's support from retrieved documents.

Pipeline position:
    Thesis input
        → [Agent 6: Thesis Analyzer]   ← this file
        → Agent 7 (Stress Synthesizer)
        → ThesisResponse

What this agent produces (feeds into ThesisResponse):
    - assumptions:       List[Assumption]       — structured causal claims
    - dependency_chain:  List[str]              — ordered label sequence A→B→C→Outcome
    - dependency_nodes:  List[DependencyNode]   — rich graph nodes for UI rendering
    - historical_analogs:List[HistoricalAnalog] — analogous market events
    - structural_robustness: High/Medium/Low    — derived from graph analysis

Latency target: 600ms (AGENT_LATENCY_TARGETS["agent6_thesis_analyzer"])
    Budget breakdown:
        Retrieval (parallel):  ~120ms
        LLM extract_assumptions: ~350ms  (Groq llama-3.3-70b)
        Support assessment:      ~80ms   (regex + heuristics, no extra LLM)
        Analogs + graph:         ~50ms
        Total:                  ~600ms

Depends on:
    utils/llm_client.py       — Groq inference
    retrieval/retriever.py    — corpus retrieval for support + analogs
    models/thesis_models.py   — Assumption, DependencyNode, HistoricalAnalog
    config.py                 — thresholds, constants
    utils/logger.py           — structured logging
"""

import re
import time
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from backend.models.thesis_models import (
    Assumption,
    DependencyNode,
    HistoricalAnalog,
    AssumptionCategory,
    ThesisStrength,
    StructuralRobustness,
)
from backend.retrieval.retriever import retriever, RetrieverError
from backend.utils.llm_client import llm_client, LLMError
from backend.utils.logger import logger, log_metric
from backend.config import (
    THESIS_MAX_ASSUMPTIONS,
    THESIS_MIN_LENGTH,
    THESIS_MAX_LENGTH,
    AGENT_LATENCY_TARGETS,
    GROQ_MODEL_FAST,
)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

AGENT_NAME = "agent6_thesis_analyzer"
LATENCY_TARGET_MS: int = AGENT_LATENCY_TARGETS.get(AGENT_NAME, 600)

# Minimum corpus chunks needed for meaningful assumption support assessment.
# If fewer, we still proceed but mark support as "insufficient data".
MIN_CHUNKS_FOR_SUPPORT: int = 2

# Retrieval settings for this agent
RETRIEVAL_TOP_K:        int   = 10    # Chunks to retrieve for support assessment
ANALOG_TOP_K:           int   = 6     # Chunks for historical analog search
SUPPORT_SCORE_THRESHOLD: float = 0.35  # Minimum final_score to count as evidence

# Evidence extraction: max characters per snippet stored in supporting_evidence
EVIDENCE_SNIPPET_LEN: int = 200

# Circular dependency risk label appended to chain
CYCLE_RISK_LABEL: str = "⚠ CIRCULAR DEPENDENCY RISK"


# ─────────────────────────────────────────────
# CURATED HISTORICAL ANALOG LIBRARY
# Pre-seeded knowledge of key Indian + global market analogs.
# Augmented at runtime by vector retrieval from corpus.
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class _AnalogSeed:
    title:         str
    period:        str
    keywords:      list[str]   # matched against thesis text (case-insensitive)
    outcome:       str
    lesson:        str
    base_score:    float       # base similarity if keyword match found (0.0–1.0)

_ANALOG_SEEDS: list[_AnalogSeed] = [
    _AnalogSeed(
        title     = "Indian Banking NIM Expansion Cycle",
        period    = "FY2003–FY2007",
        keywords  = ["nim", "net interest margin", "banking", "rate cycle", "credit growth",
                     "hdfc", "icici", "sbi", "axis", "loan growth"],
        outcome   = (
            "Indian banks experienced sustained NIM expansion of 50–100bps as credit "
            "growth averaged 28% YoY. HDFC Bank's NIM peaked at 4.8% in FY2005. "
            "Cycle ended abruptly when RBI raised repo rate 300bps in FY2008–09."
        ),
        lesson    = (
            "NIM expansion theses are highly sensitive to the rate cycle inflection point. "
            "Monitor RBI MPC commentary and systemic liquidity (LAF) for early signals. "
            "Asset quality deterioration typically lags the credit boom by 18–24 months."
        ),
        base_score = 0.82,
    ),
    _AnalogSeed(
        title     = "Telecom ARPU Compression — Jio Disruption",
        period    = "FY2017–FY2019",
        keywords  = ["arpu", "telecom", "jio", "reliance", "tariff", "4g", "data", "spectrum",
                     "airtel", "vodafone", "pricing power", "competition"],
        outcome   = (
            "Reliance Jio's FY2017 launch triggered industry-wide ARPU collapse from "
            "~₹180 to ~₹80 within 24 months. Vodafone Idea's market share fell from "
            "25% to 19%. Bharti Airtel's revenue declined 20% YoY in FY2018. "
            "Only Jio and Airtel survived as scaled players."
        ),
        lesson    = (
            "Pricing-power assumptions collapse instantly under deep-pocketed disruptors. "
            "For any thesis dependent on stable ARPU or margin, map the disruptor landscape "
            "before building in pricing assumptions. Capital moat matters more than market share."
        ),
        base_score = 0.78,
    ),
    _AnalogSeed(
        title     = "Indian IT Sector Margin Compression",
        period    = "FY2022–FY2024",
        keywords  = ["it", "tcs", "infosys", "wipro", "margin", "attrition", "wage", "hiring",
                     "ebitda margin", "offshore", "digital", "cloud"],
        outcome   = (
            "Post-COVID demand surge drove TCS and Infosys margins from ~25% to 23–24% as "
            "attrition spiked to 21%+. Wage inflation of 10–15% YoY eroded operating leverage. "
            "Margin recovery took 6 quarters as attrition normalized to 12% by Q4 FY24."
        ),
        lesson    = (
            "Margin assumptions in talent-intensive sectors must account for the wage-attrition "
            "feedback loop. High demand → high attrition → wage inflation → margin compression "
            "is a well-established pattern in Indian IT. Build in a 200–300bps margin buffer."
        ),
        base_score = 0.75,
    ),
    _AnalogSeed(
        title     = "NBFC Liquidity Crisis — IL&FS Contagion",
        period    = "FY2019–FY2020",
        keywords  = ["nbfc", "liquidity", "aum", "alm", "cp", "commercial paper", "il&fs",
                     "credit", "dewan", "hdfc", "bajaj", "indiabulls", "funding"],
        outcome   = (
            "IL&FS's ₹91,000 crore default in September 2018 triggered a systemic NBFC "
            "liquidity freeze. Mutual funds stopped rolling over CP. DHFL, Indiabulls Housing, "
            "and Reliance Capital faced AUM run-off of 20–40%. Bajaj Finance's cost of "
            "funds spiked 80bps in one quarter."
        ),
        lesson    = (
            "NBFC valuation theses must include tail-risk on wholesale funding markets. "
            "ALM mismatches (short-term CP funding long-term assets) create cliff-risk "
            "during systemic liquidity events. Diversified funding mix is a key differentiator."
        ),
        base_score = 0.80,
    ),
    _AnalogSeed(
        title     = "Global Semiconductor Capex Supercycle Peak",
        period    = "FY2021–FY2023",
        keywords  = ["semiconductor", "capex", "chip", "tsmc", "nvidia", "ai", "datacenter",
                     "hyperscaler", "demand", "supply chain", "inventory"],
        outcome   = (
            "Global semiconductor capex peaked in 2022 at $185 billion. Inventory correction "
            "in 2023 led to 18-month demand drought. TSMC's revenue fell 14% in CY2023. "
            "AI-driven recovery began in late CY2023 led by NVIDIA H100 demand surge."
        ),
        lesson    = (
            "Capex-driven demand theses have a structural inventory-cycle lag of 12–18 months. "
            "AI-linked hyperscaler capex is stickier than consumer/PC-driven cycles, but "
            "concentration risk (3 hyperscalers = 60% of AI chip demand) creates binary outcomes."
        ),
        base_score = 0.73,
    ),
    _AnalogSeed(
        title     = "Indian Pharma USFDA Warning Letter Cycle",
        period    = "FY2014–FY2017",
        keywords  = ["pharma", "usfda", "fda", "api", "generics", "gmp", "warning letter",
                     "sun pharma", "dr reddy", "cipla", "wockhardt", "import alert"],
        outcome   = (
            "Between FY2014–FY2017, 10+ Indian pharma plants received USFDA import alerts. "
            "Sun Pharma's Halol plant closure cut US revenue by ₹2,000 crore annually. "
            "Sector re-rated from 24x to 17x P/E in 18 months. Recovery took 3 years."
        ),
        lesson    = (
            "Regulatory risk is the most underpriced factor in Indian pharma theses. "
            "GMP compliance has binary impact on US revenue. Any thesis assuming stable "
            "US approvals must include a plant-specific FDA inspection calendar risk."
        ),
        base_score = 0.76,
    ),
    _AnalogSeed(
        title     = "Indian Auto Demand Slowdown — NBFC Credit Squeeze",
        period    = "FY2019–FY2020",
        keywords  = ["auto", "automobile", "ev", "vehicle", "maruti", "tata motors",
                     "two-wheeler", "bajaj auto", "hero", "retail credit", "financing"],
        outcome   = (
            "Auto industry volumes fell 18% in FY2020 — worst decline in two decades. "
            "NBFC credit tightening post-IL&FS reduced retail financing availability. "
            "Maruti's volumes fell from 1.8mn to 1.6mn units. EBITDA margins compressed "
            "300bps as operating leverage reversed."
        ),
        lesson    = (
            "Auto volume theses are highly correlated to retail credit availability, "
            "not just consumer sentiment. NBFC health is a leading indicator for auto demand. "
            "Monitor CP market spreads and NBFC AUM growth as 3-month leading indicators."
        ),
        base_score = 0.74,
    ),
    _AnalogSeed(
        title     = "Dot-com Valuation Reset — Indian Tech Startups",
        period    = "FY2022–FY2023",
        keywords  = ["startup", "valuation", "pe ratio", "p/e", "growth", "profitability",
                     "ebitda", "revenue multiple", "saas", "tech", "nykaa", "paytm",
                     "zomato", "ipo", "pre-profit"],
        outcome   = (
            "New-age tech IPOs (Paytm, Nykaa, Zomato) lost 50–80% from listing highs by "
            "CY2022. Paytm fell from ₹2,150 to ₹440. Global rate hike cycle (Fed +525bps) "
            "compressed revenue multiples from 15x to 4x. Path to profitability replaced "
            "growth-at-any-cost as the key investment framework."
        ),
        lesson    = (
            "High-multiple valuation theses for pre-profit companies are acutely sensitive "
            "to global risk-free rate. A 200bps rate hike compresses a 20x revenue multiple "
            "by ~30% through DCF mechanics alone. Always stress-test valuation at +200bps rates."
        ),
        base_score = 0.77,
    ),
    _AnalogSeed(
        title     = "PSU Bank NPA Supercycle",
        period    = "FY2015–FY2018",
        keywords  = ["npa", "gnpa", "nnpa", "psu bank", "sbi", "pnb", "bank of baroda",
                     "stressed assets", "provisioning", "pcr", "ibc", "nclt", "write-off"],
        outcome   = (
            "RBI's Asset Quality Review (AQR) in FY2016 forced recognition of ₹9 lakh crore "
            "in stressed assets. SBI's GNPA rose from 4.3% to 10.9% in 3 years. PSU bank "
            "index fell 45% from peak. IBC resolution took 3–5 years for large accounts."
        ),
        lesson    = (
            "NPA thesis inflection is driven by regulatory recognition cycles, not just "
            "economic cycles. AQR-type regulatory events can accelerate recognition by 2–3 years. "
            "Provisioning coverage and capital adequacy are the two metrics to watch as "
            "leading indicators of when the NPA cycle bottoms."
        ),
        base_score = 0.83,
    ),
    _AnalogSeed(
        title     = "RBI Rate Cut Transmission Lag — NIM Delay",
        period    = "FY2020–FY2021",
        keywords  = ["repo rate", "rbi", "rate cut", "mclr", "transmission", "nim", "monetary",
                     "easing", "liquidity", "surplus", "covid"],
        outcome   = (
            "RBI cut repo rate 250bps between Feb 2020–May 2020. However, MCLR transmission "
            "was only 40–50% within 12 months due to fixed-rate legacy book. Banks' NIM "
            "initially compressed 15–20bps before stabilizing as deposit repricing caught up."
        ),
        lesson    = (
            "Rate cut benefit assumptions for bank NIM must account for MCLR transmission lag "
            "(typically 2–4 quarters). Fixed vs floating loan mix is the key variable. "
            "Deposit repricing speed further complicates the NIM trajectory in a cut cycle."
        ),
        base_score = 0.79,
    ),
]


# ─────────────────────────────────────────────
# INTERNAL DATACLASSES
# ─────────────────────────────────────────────

@dataclass
class _GraphAnalysis:
    """
    Results of the dependency graph analysis.
    Used internally to build the dependency_chain and dependency_nodes.
    """
    ordered_chain:    list[str]            # Topological order of assumption labels
    dependency_nodes: list[DependencyNode] # Rich nodes for UI
    has_cycles:       bool                 # True if circular dependencies detected
    cycle_paths:      list[list[int]]      # Detected cycle paths (list of assumption IDs)
    critical_path:    list[int]            # Longest dependency chain (IDs)
    structural_robustness: StructuralRobustness


# ─────────────────────────────────────────────
# THESIS ANALYZER CLASS
# ─────────────────────────────────────────────

class ThesisAnalyzer:
    """
    Agent 6: Extracts assumptions, builds dependency graph, finds historical
    analogs, and assesses corpus support for each assumption.

    Usage:
        from backend.agents.agent6_thesis_analyzer import thesis_analyzer

        result = await thesis_analyzer.analyze(thesis_text, chunks=reranked_chunks)
        # result is a dict with keys:
        #   assumptions, dependency_chain, dependency_nodes,
        #   historical_analogs, structural_robustness, agents_used

        # Plug directly into ThesisResponse constructor.

    Performance:
        Parallelizes retrieval and LLM inference where possible.
        Target: 600ms end-to-end.
    """

    def __init__(self):
        self._total_runs:    int   = 0
        self._total_ms:      float = 0.0
        self._llm_failures:  int   = 0

        logger.info(
            f"[THESIS ANALYZER] Agent 6 ready | "
            f"max_assumptions={THESIS_MAX_ASSUMPTIONS} | "
            f"latency_target={LATENCY_TARGET_MS}ms | "
            f"analog_seeds={len(_ANALOG_SEEDS)}"
        )

    # ─────────────────────────────────────────
    # PUBLIC: analyze
    # ─────────────────────────────────────────

    async def analyze(
        self,
        thesis:  str,
        chunks:  Optional[list[dict]] = None,
        top_k:   int = RETRIEVAL_TOP_K,
    ) -> dict:
        """
        Main entry point. Orchestrates full thesis analysis.

        Pipeline:
            1. Validate thesis length
            2. Parallel:
               a. Retrieve corpus chunks (if not pre-supplied)
               b. extract_assumptions() via Groq LLM
            3. assess_assumption_support() for each assumption (against chunks)
            4. build_dependency_chain() from assumptions
            5. find_historical_analogs() from seeds + corpus
            6. Derive structural_robustness from graph analysis
            7. Return assembled dict

        Args:
            thesis: Raw investment thesis text (20–5000 chars)
            chunks: Pre-retrieved chunks from upstream pipeline.
                    If None, the agent retrieves its own from the corpus.
            top_k:  Number of chunks to retrieve if chunks not pre-supplied.

        Returns:
            dict with keys:
                assumptions:           list[Assumption]
                dependency_chain:      list[str]         (ordered labels)
                dependency_nodes:      list[DependencyNode]
                historical_analogs:    list[HistoricalAnalog]
                structural_robustness: StructuralRobustness  ("High"/"Medium"/"Low")
                has_circular_dependency: bool
                agents_used:           list[str]

        Raises:
            Nothing — all errors handled gracefully.
            Returns minimal valid dict on failure.

        Latency:
            ~600ms (LLM-bound). Retrieval parallelized with LLM call.

        Examples:
            result = await thesis_analyzer.analyze(
                "HDFC Bank is a BUY because NIM will expand 20bps as RBI cuts rates..."
            )
            # → {
            #     "assumptions": [Assumption(id=1, text="RBI will cut rates...", ...)],
            #     "dependency_chain": ["Rate Cut", "NIM Expansion", "PAT Growth", "Re-rating"],
            #     "structural_robustness": "Medium",
            #     ...
            # }
        """
        start_ms = time.perf_counter()
        self._total_runs += 1

        # ── Guard: thesis length ──────────────────────────────────────────
        thesis = thesis.strip()
        if len(thesis) < THESIS_MIN_LENGTH:
            logger.warning(
                f"[THESIS ANALYZER] Thesis too short: {len(thesis)} chars "
                f"(min={THESIS_MIN_LENGTH})"
            )
            return self._empty_result(reason="thesis_too_short")

        if len(thesis) > THESIS_MAX_LENGTH:
            logger.warning(
                f"[THESIS ANALYZER] Thesis truncated: {len(thesis)} → {THESIS_MAX_LENGTH} chars"
            )
            thesis = thesis[:THESIS_MAX_LENGTH]

        logger.info(
            f"[THESIS ANALYZER] Analyze start | "
            f"thesis_len={len(thesis)} | "
            f"chunks_supplied={len(chunks) if chunks else 0}"
        )

        try:
            # ── Step 1: Parallel retrieval + LLM assumption extraction ────
            retrieval_task = self._retrieve_for_thesis(thesis, chunks, top_k)
            extraction_task = self.extract_assumptions(thesis)

            retrieved_chunks, assumptions = await asyncio.gather(
                retrieval_task,
                extraction_task,
                return_exceptions=False,
            )

            logger.debug(
                f"[THESIS ANALYZER] Parallel phase complete | "
                f"chunks={len(retrieved_chunks)} | "
                f"assumptions={len(assumptions)}"
            )

            # ── Step 2: Assess assumption support (heuristic, fast) ───────
            if retrieved_chunks:
                assumptions = await self._assess_all_assumptions(
                    assumptions, retrieved_chunks
                )

            # ── Step 3: Build dependency graph ────────────────────────────
            graph = self.build_dependency_chain(assumptions)

            # ── Step 4: Find historical analogs ───────────────────────────
            analogs = await self.find_historical_analogs(thesis, retrieved_chunks)

            # ── Step 5: Assemble result ───────────────────────────────────
            duration_ms = (time.perf_counter() - start_ms) * 1000
            self._total_ms += duration_ms

            result = {
                "assumptions":             assumptions,
                "dependency_chain":        graph.ordered_chain,
                "dependency_nodes":        graph.dependency_nodes,
                "historical_analogs":      analogs,
                "structural_robustness":   graph.structural_robustness,
                "has_circular_dependency": graph.has_cycles,
                "cycle_paths":             graph.cycle_paths,
                "agents_used":             [AGENT_NAME],
            }

            logger.info(
                f"[THESIS ANALYZER] Complete | "
                f"assumptions={len(assumptions)} | "
                f"analogs={len(analogs)} | "
                f"chain_len={len(graph.ordered_chain)} | "
                f"cycles={graph.has_cycles} | "
                f"robustness={graph.structural_robustness} | "
                f"{duration_ms:.0f}ms"
            )

            log_metric("thesis_analyzer_duration_ms",  duration_ms,          unit="ms")
            log_metric("thesis_analyzer_assumptions",  len(assumptions),      unit="count")
            log_metric("thesis_analyzer_analogs",      len(analogs),          unit="count")

            if duration_ms > LATENCY_TARGET_MS:
                logger.warning(
                    f"[THESIS ANALYZER] Latency target breached: "
                    f"{duration_ms:.0f}ms > {LATENCY_TARGET_MS}ms"
                )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_ms) * 1000
            logger.error(
                f"[THESIS ANALYZER] Unexpected error: {type(e).__name__}: {e} | "
                f"{duration_ms:.0f}ms"
            )
            return self._empty_result(reason=f"error: {type(e).__name__}")

    # ─────────────────────────────────────────
    # PUBLIC: extract_assumptions
    # ─────────────────────────────────────────

    async def extract_assumptions(self, thesis: str) -> list[Assumption]:
        """
        Extracts structured causal assumptions from the thesis using Groq LLM.

        Prompt design:
            - Instructs model to identify CAUSAL claims (not factual statements)
            - Each assumption must be falsifiable — testable against data
            - Categories: demand, margin, valuation, regulatory, macro,
                          competitive, operational, financial
            - dependency_on: IDs of OTHER assumptions this one requires to be true
            - is_critical: True if thesis thesis completely fails without this

        LLM Output (JSON array):
            [
              {
                "id": 1,
                "text": "RBI will cut repo rate by 50bps in H2 FY26",
                "category": "macro",
                "confidence": 0.7,
                "dependency_on": [],
                "is_critical": true
              },
              ...
            ]

        Post-processing:
            - Deduplicates overlapping assumptions
            - Caps at THESIS_MAX_ASSUMPTIONS (8)
            - Sets historical_support=False (filled later by assess_assumption_support)
            - Validates via Assumption Pydantic model — skips invalid items

        Args:
            thesis: Investment thesis text (already validated for length)

        Returns:
            list[Assumption] — 1 to THESIS_MAX_ASSUMPTIONS items.
            Returns [] on LLM failure (logged as warning, not error).

        Latency: ~350ms (Groq llama-3.3-70b)
        """
        prompt = f"""You are a senior investment analyst specializing in Indian equities and macro.

Analyze this investment thesis and extract all CAUSAL ASSUMPTIONS it depends on.

THESIS:
\"\"\"
{thesis}
\"\"\"

INSTRUCTIONS:
1. Extract 3–{THESIS_MAX_ASSUMPTIONS} causal assumptions the thesis implicitly or explicitly relies on.
2. Each assumption must be a FALSIFIABLE CLAIM (testable against market/financial data).
3. Do NOT extract descriptive facts — only forward-looking causal claims.
4. For dependency_on: list the IDs of OTHER assumptions this one requires to be true first.
   Example: If assumption 3 depends on assumption 1 being true, set dependency_on: [1].
5. Mark is_critical=true if the ENTIRE thesis breaks if this assumption fails.
6. Confidence = your belief this assumption will hold (0.0–1.0).

CATEGORIES (pick ONE):
- "demand"       : Revenue/volume growth claims
- "margin"       : Profitability/cost/operating leverage claims
- "valuation"    : Multiple expansion/compression claims
- "regulatory"   : Government/RBI/SEBI policy claims
- "macro"        : Interest rates, GDP, inflation, FX claims
- "competitive"  : Market share, competitive moat claims
- "operational"  : Capacity utilization, efficiency claims
- "financial"    : Capital structure, leverage, funding claims

Return ONLY a valid JSON array. No explanation text outside JSON.
Each item must have exactly these fields:
{{
  "id": <int 1-{THESIS_MAX_ASSUMPTIONS}>,
  "text": "<assumption statement, 15–300 chars>",
  "category": "<one of the categories above>",
  "confidence": <float 0.0–1.0>,
  "dependency_on": [<list of ints — IDs this depends on, or empty []>],
  "is_critical": <true|false>
}}"""

        try:
            raw = await llm_client.call_llm_json(
                prompt      = prompt,
                agent_name  = AGENT_NAME,
                max_tokens  = 1200,
                temperature = 0.15,
            )
        except LLMError as e:
            self._llm_failures += 1
            logger.warning(
                f"[THESIS ANALYZER] extract_assumptions LLM failure: {e}. "
                f"Falling back to regex extraction."
            )
            return self._regex_extract_assumptions(thesis)

        # Normalise: LLM may return dict with an "assumptions" key or bare array
        items = raw if isinstance(raw, list) else (
            raw.get("assumptions", raw.get("items", raw.get("data", [])))
        )
        if not isinstance(items, list):
            logger.warning(
                f"[THESIS ANALYZER] LLM returned unexpected type: {type(raw)}. "
                f"Using regex fallback."
            )
            return self._regex_extract_assumptions(thesis)

        assumptions: list[Assumption] = []
        seen_texts: set[str] = set()

        for item in items[:THESIS_MAX_ASSUMPTIONS]:
            if not isinstance(item, dict):
                continue
            try:
                # Sanitise fields
                item_id  = int(item.get("id", len(assumptions) + 1))
                text     = str(item.get("text", "")).strip()
                category = str(item.get("category", "unknown")).lower().strip()
                conf_raw = item.get("confidence", 0.5)
                dep_on   = [int(x) for x in (item.get("dependency_on") or []) if str(x).isdigit()]
                critical = bool(item.get("is_critical", False))

                # Skip duplicates and too-short texts
                if len(text) < 10:
                    continue
                text_key = text[:60].lower()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                # Validate category
                valid_categories = {
                    "demand", "margin", "valuation", "regulatory",
                    "macro", "competitive", "operational", "financial", "unknown"
                }
                if category not in valid_categories:
                    category = "unknown"

                confidence = max(0.0, min(1.0, float(conf_raw)))

                # Ensure assumption ID is within bounds
                if not (1 <= item_id <= THESIS_MAX_ASSUMPTIONS):
                    item_id = len(assumptions) + 1

                assumption = Assumption(
                    id                  = item_id,
                    text                = text,
                    category            = category,
                    confidence          = confidence,
                    historical_support  = False,   # filled by assess_assumption_support
                    supporting_evidence = [],
                    dependency_on       = dep_on,
                    is_critical         = critical,
                )
                assumptions.append(assumption)

            except (ValueError, TypeError, KeyError) as e:
                logger.debug(
                    f"[THESIS ANALYZER] Skipping malformed assumption item: {item} | {e}"
                )
                continue

        # Ensure assumption IDs are unique and sequential
        assumptions = self._renumber_assumptions(assumptions)

        logger.debug(
            f"[THESIS ANALYZER] Extracted {len(assumptions)} assumptions via LLM"
        )
        return assumptions

    # ─────────────────────────────────────────
    # PUBLIC: build_dependency_chain
    # ─────────────────────────────────────────

    def build_dependency_chain(
        self,
        assumptions: list[Assumption],
    ) -> _GraphAnalysis:
        """
    Builds a directed dependency graph from assumptions and returns
    a topologically ordered chain + rich DependencyNode objects.

    Algorithm:
        1. Build adjacency list from assumption.dependency_on fields
        2. Detect cycles using iterative DFS with an explicit stack
        3. Topological sort using Kahn's algorithm (BFS from zero-in-degree)
           — On cycle detection, appends CYCLE_RISK_LABEL to chain
        4. Build DependencyNode objects with risk_score heuristics
        5. Critical path via DP — SKIPPED for cyclic graphs (prev forms
           a closed loop → infinite traceback if not guarded)
        6. Compute structural_robustness

    Critical path safety rules:
        - DP only runs when has_cycles=False (DAG guaranteed)
        - Traceback additionally guarded by seen_in_path set (belt + braces)
        """
        if not assumptions:
            return _GraphAnalysis(
                ordered_chain         = [],
                dependency_nodes      = [],
                has_cycles            = False,
                cycle_paths           = [],
                critical_path         = [],
                structural_robustness = "Low",
            )

        id_to_assumption: dict[int, Assumption] = {a.id: a for a in assumptions}
        ids = [a.id for a in assumptions]

    # ── Build adjacency lists ─────────────────────────────────────────────
        adj_in:  dict[int, set[int]] = defaultdict(set)
        adj_out: dict[int, set[int]] = defaultdict(set)

        for assumption in assumptions:
            for dep_id in assumption.dependency_on:
                if dep_id in id_to_assumption and dep_id != assumption.id:
                    adj_in[assumption.id].add(dep_id)
                    adj_out[dep_id].add(assumption.id)

    # ── Iterative DFS cycle detection ─────────────────────────────────────
        has_cycles   = False
        cycle_paths: list[list[int]] = []
        visited      = set()
        rec_stack    = set()

        for start in ids:
            if start in visited:
                continue

            stack: list[tuple[int, object, list[int]]] = [
                (start, iter(sorted(adj_out.get(start, set()))), [start])
            ]
            visited.add(start)
            rec_stack.add(start)

            while stack:
                node, neighbors, path = stack[-1]

                try:
                    neighbor = next(neighbors)

                    if neighbor not in visited:
                        visited.add(neighbor)
                        rec_stack.add(neighbor)
                        stack.append((
                            neighbor,
                            iter(sorted(adj_out.get(neighbor, set()))),
                            path + [neighbor],          # new list — no shared state
                        ))

                    elif neighbor in rec_stack:
                        has_cycles = True
                        try:
                            cycle_start = path.index(neighbor)
                            cycle = path[cycle_start:] + [neighbor]
                        except ValueError:
                            cycle = path + [neighbor]
                        cycle_paths.append(cycle)       # continue — find all cycles

                except StopIteration:
                    stack.pop()
                    rec_stack.discard(node)             # clean backtrack

    # ── Topological sort — Kahn's algorithm ──────────────────────────────
        in_degree: dict[int, int] = {aid: len(adj_in[aid]) for aid in ids}
        queue = deque(aid for aid in ids if in_degree[aid] == 0)
        topo_order: list[int] = []

        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for child in sorted(adj_out.get(node, set())):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    # Cycle nodes have in_degree > 0 → never enter queue → append in order
        in_topo = set(topo_order)
        topo_order.extend(aid for aid in ids if aid not in in_topo)

    # ── Build ordered chain labels ─────────────────────────────────────────
        ordered_chain: list[str] = [
            _shorten_label(id_to_assumption[aid].text)
            for aid in topo_order
        ]
        if has_cycles:
            ordered_chain.append(CYCLE_RISK_LABEL)

    # ── Build DependencyNode objects ───────────────────────────────────────
        dependency_nodes: list[DependencyNode] = []
        for i, aid in enumerate(topo_order):
            assumption  = id_to_assumption[aid]
            risk_score  = _confidence_to_risk(assumption.confidence, assumption.historical_support)
            is_terminal = (i == len(topo_order) - 1)

            dependency_nodes.append(DependencyNode(
                id          = assumption.id,
                label       = _shorten_label(assumption.text),
                description = assumption.text[:180],
                depends_on  = list(adj_in.get(aid, set())),
                risk_score  = risk_score,
                is_terminal = is_terminal,
            ))

    # ── Critical path (longest chain) via DP ─────────────────────────────
    #
    # SAFETY: Skip entirely when has_cycles=True.
    #
    # Why: The DP iterates topo_order once and is always finite. But on a
    # cyclic graph the back-edge in topo_order (e.g. aid=3 → child=1)
    # overwrites prev[1] = 3, closing the loop:
    #     prev = {1: 3,  2: 1,  3: 2}  →  1→3→2→1→...  ∞
    # The `while cur is not None` traceback then never terminates.
    # Critical path is also logically undefined on a cyclic dependency graph.
    #
        critical_path: list[int] = []

        if not has_cycles:
            dist: dict[int, int]           = {aid: 0 for aid in ids}
            prev: dict[int, Optional[int]] = {aid: None for aid in ids}

            for aid in topo_order:
                for child in adj_out.get(aid, set()):
                    if child in dist and dist[aid] + 1 > dist[child]:
                        dist[child] = dist[aid] + 1
                        prev[child] = aid

            end_node = max(dist, key=lambda x: dist[x])
            cur: Optional[int] = end_node
            seen_in_path: set[int] = set()              # belt-and-braces guard

            while cur is not None and cur not in seen_in_path:
                seen_in_path.add(cur)
                critical_path.insert(0, cur)
                cur = prev.get(cur)

    # ── Structural robustness ─────────────────────────────────────────────
        structural_robustness = self._compute_robustness(assumptions, has_cycles)

        return _GraphAnalysis(
            ordered_chain         = ordered_chain,
            dependency_nodes      = dependency_nodes,
            has_cycles            = has_cycles,
            cycle_paths           = cycle_paths,
            critical_path         = critical_path,
            structural_robustness = structural_robustness,
        )

    # ─────────────────────────────────────────
    # PUBLIC: find_historical_analogs
    # ─────────────────────────────────────────

    async def find_historical_analogs(
        self,
        thesis:  str,
        chunks:  Optional[list[dict]] = None,
    ) -> list[HistoricalAnalog]:
        """
        Finds historical market events analogous to the thesis.

        Two-stage process:
            Stage 1 — Curated seed matching (fast, no LLM, ~5ms):
                Scans _ANALOG_SEEDS for keyword overlap with thesis text.
                Score = base_score × (matched_keywords / total_seed_keywords)
                Returns seeds where score > 0.35.

            Stage 2 — Corpus retrieval (async, ~80ms):
                Queries retriever with thesis-derived query fragments.
                Extracts analog signals from returned chunks.
                Merges with Stage 1 results.

        Deduplication:
            Analogs are deduplicated by title similarity.
            Maximum 4 analogs returned (to avoid noise).

        Args:
            thesis: Investment thesis text
            chunks: Pre-retrieved chunks (optional, used to avoid re-retrieval)

        Returns:
            list[HistoricalAnalog] sorted by similarity_score descending.
            Maximum 4 items.

        Examples:
            thesis = "HDFC NIM will expand as RBI cuts rates..."
            analogs → [
                HistoricalAnalog(
                    title="Indian Banking NIM Expansion Cycle",
                    period="FY2003–FY2007",
                    similarity_score=0.76,
                    outcome="...",
                    lesson="...",
                ),
                HistoricalAnalog(
                    title="RBI Rate Cut Transmission Lag — NIM Delay",
                    period="FY2020–FY2021",
                    similarity_score=0.68,
                    ...
                ),
            ]
        """
        thesis_lower = thesis.lower()
        thesis_words = set(re.findall(r'\b[a-z]{3,}\b', thesis_lower))

        analogs: list[HistoricalAnalog] = []

        # ── Stage 1: Curated seed matching ───────────────────────────────
        for seed in _ANALOG_SEEDS:
            matched_kw  = sum(1 for kw in seed.keywords if kw.lower() in thesis_lower)
            total_kw    = len(seed.keywords)
            if total_kw == 0 or matched_kw == 0:
                continue

            kw_ratio    = matched_kw / total_kw
            score       = round(seed.base_score * (0.4 + 0.6 * kw_ratio), 3)
            score       = min(score, 0.98)

            if score < 0.30:
                continue

            analogs.append(HistoricalAnalog(
                title            = seed.title,
                period           = seed.period,
                similarity_score = score,
                outcome          = seed.outcome,
                lesson           = seed.lesson,
                source           = "FinThesisGuard Curated Analog Library",
            ))

        logger.debug(
            f"[THESIS ANALYZER] Stage 1 analogs (seed match): {len(analogs)}"
        )

        # ── Stage 2: Corpus-based analog retrieval ────────────────────────
        # Build a focused query from key thesis terms
        analog_query = _build_analog_query(thesis)

        try:
            if chunks and len(chunks) >= MIN_CHUNKS_FOR_SUPPORT:
                # Use pre-supplied chunks — look for historical signals
                corpus_analogs = self._extract_corpus_analogs(chunks, thesis_lower)
            else:
                # Retrieve fresh — search across all namespaces
                corpus_chunks = await retriever.retrieve(
                    query                = analog_query,
                    top_k                = ANALOG_TOP_K,
                    search_all_namespaces = True,
                    min_score            = 0.25,
                )
                corpus_analogs = self._extract_corpus_analogs(corpus_chunks, thesis_lower)

            analogs.extend(corpus_analogs)
            logger.debug(
                f"[THESIS ANALYZER] Stage 2 analogs (corpus): {len(corpus_analogs)}"
            )

        except (RetrieverError, Exception) as e:
            logger.warning(
                f"[THESIS ANALYZER] Corpus analog retrieval failed: {e}. "
                f"Using seed analogs only."
            )

        # ── Deduplicate and sort ──────────────────────────────────────────
        analogs = _deduplicate_analogs(analogs)
        analogs.sort(key=lambda a: a.similarity_score, reverse=True)

        # Cap at 4 to avoid noise
        return analogs[:4]

    # ─────────────────────────────────────────
    # PUBLIC: assess_assumption_support
    # ─────────────────────────────────────────

    async def assess_assumption_support(
        self,
        assumption: Assumption,
        chunks:     list[dict],
    ) -> bool:
        """
        Checks whether the given assumption is supported by retrieved corpus chunks.

        Support assessment strategy (no LLM — purely heuristic for speed):
            1. Keyword extraction: tokenize assumption text into key terms
               (nouns, metrics, company names, percentages)
            2. For each chunk with final_score ≥ SUPPORT_SCORE_THRESHOLD:
               - Count how many key terms appear in chunk text (case-insensitive)
               - Apply authority weight boost (rbi/annual_report count more)
            3. Support score = Σ(term_matches × authority_weight) across chunks
               normalized by max_possible_score
            4. Returns True if support_score ≥ 0.25 (at least 25% of terms
               found in high-authority sources)

        Args:
            assumption: Assumption to check
            chunks:     Retrieved corpus chunks (with final_score and metadata)

        Returns:
            bool — True if assumption is supported by corpus evidence

        Side effects:
            Mutates assumption.supporting_evidence with matched snippets.
            Mutates assumption.historical_support with the result.

        Examples:
            assumption.text = "RBI will cut repo rate by 50bps in H2 FY26"
            chunks with RBI MPC statements mentioning "rate cut" and "accommodative"
            → True (RBI source, high authority, term overlap)

            assumption.text = "Revenue CAGR will be 35% for next 5 years"
            chunks with actual revenue growing 8%
            → False (no corroborating evidence)
        """
        if not chunks:
            return False

        # Extract key terms from assumption text
        key_terms = _extract_key_terms(assumption.text)
        if not key_terms:
            return False

        support_score      = 0.0
        max_possible_score = 0.0
        evidence_snippets: list[str] = []

        from backend.retrieval.retriever import get_authority_weight

        for chunk in chunks:
            chunk_score = float(chunk.get("final_score", chunk.get("score", 0.0)))
            if chunk_score < SUPPORT_SCORE_THRESHOLD:
                continue

            chunk_text  = chunk.get("text", "").lower()
            source_type = chunk.get("source_type", chunk.get("metadata", {}).get("source_type", "unknown"))
            authority   = get_authority_weight(source_type)

            max_possible_score += len(key_terms) * authority

            matched_terms: list[str] = []
            for term in key_terms:
                if term.lower() in chunk_text:
                    matched_terms.append(term)
                    support_score += authority

            # Collect evidence snippet if terms matched
            if matched_terms and len(evidence_snippets) < 5:
                src_name = (
                    chunk.get("source")
                    or chunk.get("metadata", {}).get("source", "Unknown Source")
                )
                snippet_start = max(0, chunk_text.find(matched_terms[0].lower()) - 40)
                snippet       = chunk.get("text", "")[snippet_start:snippet_start + EVIDENCE_SNIPPET_LEN]
                evidence_snippets.append(
                    f"[{src_name}] ...{snippet.strip()}..."
                )

        # Normalize score
        if max_possible_score == 0:
            return False

        normalized = support_score / max_possible_score
        is_supported = normalized >= 0.25

        # Attach evidence to assumption (mutate via model copy — Pydantic v2 safe)
        if evidence_snippets:
            object.__setattr__(assumption, "supporting_evidence", evidence_snippets[:5])
        object.__setattr__(assumption, "historical_support", is_supported)

        logger.debug(
            f"[THESIS ANALYZER] Assumption {assumption.id} support: "
            f"score={normalized:.2f} | supported={is_supported} | "
            f"evidence_count={len(evidence_snippets)}"
        )

        return is_supported

    # ─────────────────────────────────────────
    # PUBLIC: get_stats
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        avg_ms = round(self._total_ms / max(1, self._total_runs), 1)
        return {
            "total_runs":     self._total_runs,
            "total_ms":       round(self._total_ms, 1),
            "avg_ms_per_run": avg_ms,
            "llm_failures":   self._llm_failures,
            "analog_seeds":   len(_ANALOG_SEEDS),
        }

    # ─────────────────────────────────────────
    # PRIVATE: parallel helpers
    # ─────────────────────────────────────────

    async def _retrieve_for_thesis(
        self,
        thesis: str,
        chunks: Optional[list[dict]],
        top_k:  int,
    ) -> list[dict]:
        """
        Returns chunks: uses pre-supplied ones or retrieves from corpus.
        Never raises — returns [] on failure.
        """
        if chunks and len(chunks) >= MIN_CHUNKS_FOR_SUPPORT:
            return chunks

        query = thesis[:400]  # First 400 chars as query — captures main claim

        try:
            return await retriever.retrieve(
                query                = query,
                top_k                = top_k,
                search_all_namespaces = True,
                min_score            = 0.25,
            )
        except RetrieverError as e:
            logger.warning(
                f"[THESIS ANALYZER] Corpus retrieval failed: {e}. "
                f"Proceeding without corpus support."
            )
            return []

    async def _assess_all_assumptions(
        self,
        assumptions: list[Assumption],
        chunks:      list[dict],
    ) -> list[Assumption]:
        """
        Runs assess_assumption_support for all assumptions concurrently.
        Updates each assumption in-place. Returns the updated list.
        """
        tasks = [
            self.assess_assumption_support(a, chunks)
            for a in assumptions
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        return assumptions

    # ─────────────────────────────────────────
    # PRIVATE: corpus analog extraction
    # ─────────────────────────────────────────

    def _extract_corpus_analogs(
        self,
        chunks:      list[dict],
        thesis_lower: str,
    ) -> list[HistoricalAnalog]:
        """
        Searches retrieved chunks for historical period references
        that might constitute analogs to the current thesis.

        Looks for:
            - Year ranges (e.g., "FY2003–FY2007", "2001–2003")
            - Named crises ("dot-com", "il&fs", "subprime", "covid")
            - Comparative language ("similar to", "like the", "reminiscent of")
        """
        analogs: list[HistoricalAnalog] = []

        # Named event patterns
        event_patterns = [
            (r'\b(?:dot.?com|tech\s*bubble)\b', "Dot-com Bubble", "FY2000–FY2002",
             "Tech valuation bubble followed by 80%+ correction", 0.65),
            (r'\b(?:subprime|GFC|global\s*financial\s*crisis)\b', "Global Financial Crisis",
             "FY2008–FY2009", "Systemic credit crisis, Indian markets fell 60%", 0.70),
            (r'\b(?:il&fs|ILFS|IL&FS)\b', "IL&FS NBFC Contagion", "FY2019",
             "NBFC liquidity crisis triggered by IL&FS default", 0.75),
            (r'\b(?:demonetisation|demonetization|note\s*ban)\b', "Demonetisation",
             "FY2017", "Liquidity shock → near-term consumption slowdown", 0.72),
            (r'\b(?:covid|pandemic|lockdown)\b', "COVID-19 Disruption", "FY2020–FY2021",
             "Economic shock, fiscal response, rapid V-shaped recovery in markets", 0.68),
            (r'\b(?:jio|reliance\s*jio)\b', "Jio Telecom Disruption", "FY2017–FY2019",
             "Pricing disruption led to industry consolidation", 0.73),
        ]

        seen_titles: set[str] = set()

        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            chunk_lower = chunk_text.lower()

            # Check how relevant this chunk is to the thesis
            thesis_overlap = len(
                set(re.findall(r'\b[a-z]{4,}\b', thesis_lower)) &
                set(re.findall(r'\b[a-z]{4,}\b', chunk_lower))
            )
            if thesis_overlap < 3:
                continue  # Not relevant enough

            src_name = (
                chunk.get("source")
                or chunk.get("metadata", {}).get("source", "Corpus")
            )

            for pattern, title, period, outcome, base_score in event_patterns:
                if title in seen_titles:
                    continue
                if re.search(pattern, chunk_text, re.IGNORECASE):
                    seen_titles.add(title)
                    # Adjust score based on chunk relevance
                    score = round(min(base_score + (thesis_overlap / 100), 0.95), 3)
                    analogs.append(HistoricalAnalog(
                        title            = title,
                        period           = period,
                        similarity_score = score,
                        outcome          = outcome,
                        lesson           = (
                            f"This historical event was referenced in your corpus documents. "
                            f"Study its precedent for the thesis scenario you are evaluating."
                        ),
                        source = src_name,
                    ))

        return analogs

    # ─────────────────────────────────────────
    # PRIVATE: structural robustness
    # ─────────────────────────────────────────

    def _compute_robustness(
        self,
        assumptions: list[Assumption],
        has_cycles:  bool,
    ) -> StructuralRobustness:
        """
        Derives structural robustness from assumptions and graph shape.

        Rules (in order of priority):
            - has_cycles=True                    → always "Low"
            - 2+ critical assumptions unsupported → "Low"
            - avg confidence < 0.4               → "Low"
            - avg confidence ≥ 0.65 and no unsupported critical → "High"
            - else                               → "Medium"
        """
        if not assumptions:
            return "Low"

        if has_cycles:
            return "Low"

        unsupported_critical = sum(
            1 for a in assumptions if a.is_critical and not a.historical_support
        )
        if unsupported_critical >= 2:
            return "Low"

        avg_conf = sum(a.confidence for a in assumptions) / len(assumptions)

        if avg_conf < 0.4:
            return "Low"
        if avg_conf >= 0.65 and unsupported_critical == 0:
            return "High"
        return "Medium"

    # ─────────────────────────────────────────
    # PRIVATE: regex fallback assumption extraction
    # ─────────────────────────────────────────

    def _regex_extract_assumptions(self, thesis: str) -> list[Assumption]:
        """
        Fallback assumption extractor when LLM is unavailable.
        Uses regex to identify forward-looking causal statements.
        Returns 1–4 assumptions — less accurate than LLM but never fails.
        """
        assumptions: list[Assumption] = []

        # Patterns that indicate a causal assumption
        causal_patterns = [
            (r'(?:will|shall|expected\s+to|likely\s+to|projected\s+to|forecast\s+to)\s+\w+', 0.55),
            (r'(?:if|assuming|provided\s+that|given\s+that)\s+[\w\s,]+', 0.45),
            (r'(?:growth|expansion|improvement|compression|slowdown)\s+(?:of|in|to)\s+\d', 0.60),
            (r'(?:rate\s*cut|rate\s*hike|rbi|sebi|regulatory)', 0.50),
        ]

        # Category inference keywords
        category_map: list[tuple[list[str], AssumptionCategory]] = [
            (["revenue", "demand", "volume", "growth", "market share"], "demand"),
            (["margin", "ebitda", "nim", "npa", "profitability", "cost"], "margin"),
            (["pe", "p/e", "valuation", "multiple", "re-rate", "price"], "valuation"),
            (["rbi", "sebi", "regulatory", "policy", "regulation", "law"], "regulatory"),
            (["rate", "inflation", "gdp", "macro", "fx", "dollar", "rate cut"], "macro"),
            (["competitor", "competition", "market share", "moat"], "competitive"),
        ]

        sentences = re.split(r'[.!?;]', thesis)
        seen: set[str] = set()
        aid = 1

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue

            for pattern, base_conf in causal_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    key = sentence[:40].lower()
                    if key in seen:
                        continue
                    seen.add(key)

                    # Infer category
                    sent_lower = sentence.lower()
                    category   = "unknown"
                    for keywords, cat in category_map:
                        if any(kw in sent_lower for kw in keywords):
                            category = cat
                            break

                    assumptions.append(Assumption(
                        id                 = aid,
                        text               = sentence[:300],
                        category           = category,
                        confidence         = base_conf,
                        historical_support = False,
                        supporting_evidence= [],
                        dependency_on      = [aid - 1] if aid > 1 else [],
                        is_critical        = aid == 1,  # First assumption usually critical
                    ))
                    aid += 1
                    if aid > THESIS_MAX_ASSUMPTIONS:
                        break

            if aid > THESIS_MAX_ASSUMPTIONS:
                break

        logger.debug(
            f"[THESIS ANALYZER] Regex fallback extracted {len(assumptions)} assumptions"
        )
        return assumptions

    # ─────────────────────────────────────────
    # PRIVATE: helpers
    # ─────────────────────────────────────────

    def _renumber_assumptions(
        self, assumptions: list[Assumption]
    ) -> list[Assumption]:
        """
        Ensures assumption IDs are unique sequential integers 1..n.
        Rewrites dependency_on references to match new IDs.
        """
        old_to_new: dict[int, int] = {}
        renumbered: list[Assumption] = []

        for i, assumption in enumerate(assumptions, start=1):
            old_to_new[assumption.id] = i
            renumbered.append(assumption.model_copy(update={"id": i}))

        # Rewrite dependency_on references
        final: list[Assumption] = []
        for assumption in renumbered:
            new_deps = [
                old_to_new[d] for d in assumption.dependency_on
                if d in old_to_new and old_to_new[d] != assumption.id
            ]
            final.append(assumption.model_copy(update={"dependency_on": new_deps}))

        return final

    @staticmethod
    def _empty_result(reason: str = "") -> dict:
        """Returns a minimal valid result dict on failure."""
        logger.warning(f"[THESIS ANALYZER] Returning empty result | reason={reason}")
        return {
            "assumptions":             [],
            "dependency_chain":        [],
            "dependency_nodes":        [],
            "historical_analogs":      [],
            "structural_robustness":   "Low",
            "has_circular_dependency": False,
            "cycle_paths":             [],
            "agents_used":             [AGENT_NAME],
        }


# ─────────────────────────────────────────────
# MODULE-LEVEL HELPER FUNCTIONS
# ─────────────────────────────────────────────

def _shorten_label(text: str, max_len: int = 40) -> str:
    """
    Creates a short label from assumption text for chain display.
    Takes first 2–4 meaningful words, removes filler words.
    """
    filler = {"the", "a", "an", "is", "in", "on", "at", "to", "of", "and",
               "or", "for", "by", "with", "this", "that", "it", "will", "be"}
    words  = [w for w in re.findall(r'\b\w+\b', text) if w.lower() not in filler]
    label  = " ".join(words[:5])
    if len(label) > max_len:
        label = label[:max_len - 1] + "…"
    return label.title()


def _confidence_to_risk(confidence: float, historical_support: bool) -> int:
    """
    Converts confidence + support into a 1–10 risk score for DependencyNode.
    Higher risk = lower confidence + no historical support.
    """
    base_risk = round((1 - confidence) * 9) + 1  # 1 (high conf) → 10 (low conf)
    if not historical_support:
        base_risk = min(10, base_risk + 2)        # Penalty for no corpus support
    return int(max(1, min(10, base_risk)))


def _extract_key_terms(text: str) -> list[str]:
    """
    Extracts key searchable terms from assumption text for support assessment.

    Extracts:
        - Financial metrics (NIM, GNPA, ROE, etc.)
        - Percentages and numbers
        - Company names (HDFC, RBI, TCS, etc.)
        - Key action verbs (expand, cut, compress, grow)
        - Directional words (above, below, increase, decrease)
    """
    terms: list[str] = []

    # Financial acronyms (2–6 uppercase letters)
    acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
    terms.extend(acronyms)

    # Percentages and numeric values
    nums = re.findall(r'\d+(?:\.\d+)?%?', text)
    terms.extend(nums[:3])  # Cap at 3 numeric terms

    # Important single words (length > 4)
    important_words = [
        w for w in re.findall(r'\b[a-zA-Z]{5,}\b', text)
        if w.lower() not in {
            "which", "where", "their", "there", "these", "those",
            "would", "could", "should", "about", "above", "below",
            "because", "therefore", "however", "although", "while",
        }
    ]
    terms.extend(important_words[:6])

    return list(dict.fromkeys(terms))   # Deduplicate, preserve order


def _build_analog_query(thesis: str) -> str:
    """
    Extracts a concise search query from the thesis for historical analog retrieval.
    Takes the first sentence + financial terms.
    """
    first_sentence = re.split(r'[.!?]', thesis)[0].strip()
    acronyms       = re.findall(r'\b[A-Z]{2,6}\b', thesis)
    query_parts    = [first_sentence[:150]] + acronyms[:5]
    return " ".join(query_parts)[:300]


def _deduplicate_analogs(analogs: list[HistoricalAnalog]) -> list[HistoricalAnalog]:
    """
    Deduplicates analogs by title similarity (first 30 chars).
    Keeps highest similarity_score when titles overlap.
    """
    seen:   dict[str, HistoricalAnalog] = {}
    for analog in analogs:
        key = analog.title[:30].lower()
        if key not in seen or analog.similarity_score > seen[key].similarity_score:
            seen[key] = analog
    return list(seen.values())


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

thesis_analyzer = ThesisAnalyzer()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "thesis_analyzer",
    "ThesisAnalyzer",
]
