# backend/agents/agent2_query_decomposer.py
"""
FinThesisGuard AI — Agent 2: Query Decomposer
ArkAngel Financial Solutions

Breaks complex multi-part financial queries into focused atomic
sub-queries for parallel retrieval. Feeds into Agent 3 (retriever).

Decomposition strategies:
    SIMPLE       → query returned as-is (no decomposition needed)
    COMPARISON   → "X vs Y metric" → [X metric, Y metric]
    MULTI_ENTITY → "X, Y, Z metric" → [X metric, Y metric, Z metric]
    MULTI_METRIC → "entity metric1 + metric2" → [entity m1, entity m2]
    TEMPORAL     → "trend over FY24 FY25 FY26" → [q FY24, q FY25, q FY26]
    COMPOSITE    → mixed — LLM decomposes freely

Decision tree (in order, first match wins):
    1. is_complex_query() == False      → SIMPLE (0ms, no LLM)
    2. Has "vs"/"compare" + 2 entities  → COMPARISON (rule-based, ~2ms)
    3. Has 3+ company mentions          → MULTI_ENTITY (rule-based, ~2ms)
    4. Has 3+ metric mentions           → MULTI_METRIC (rule-based, ~3ms)
    5. Has multiple fiscal year tokens  → TEMPORAL (rule-based, ~2ms)
    6. Everything else                  → COMPOSITE (Groq LLM, ~180ms)

Key contracts:
    - Max sub-queries: 6 (prevents over-decomposition & token bloat)
    - Fallback: if Groq fails, returns [original_query] as single sub-query
    - All sub-queries tagged with focus_entity, metric_hint, time_range
    - Thread-safe: no shared mutable state
    - Idempotent: same input always returns structurally same decomposition
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.utils.llm_client      import llm_client, LLMError
from backend.models.query_models   import QueryType
from backend.utils.logger          import logger, log_metric


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MAX_SUB_QUERIES:     int   = 6       # Hard cap on decomposed sub-queries
MIN_SUBQUERY_LEN:    int   = 5       # Minimum chars for a valid sub-query
LATENCY_TARGET_MS:   float = 200.0   # Target latency for LLM path

AGENT_NAME = "agent2_decomposer"


# ─────────────────────────────────────────────
# DECOMPOSITION STRATEGY ENUM
# ─────────────────────────────────────────────

class DecomposeStrategy(str, Enum):
    SIMPLE       = "SIMPLE"       # No decomposition: single entity, single metric
    COMPARISON   = "COMPARISON"   # Entity A vs Entity B
    MULTI_ENTITY = "MULTI_ENTITY" # Entity A, B, C with same metric
    MULTI_METRIC = "MULTI_METRIC" # Entity with metric A, B, C
    TEMPORAL     = "TEMPORAL"     # Same entity/metric across time periods
    COMPOSITE    = "COMPOSITE"    # Complex: LLM-assisted decomposition


# ─────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────

@dataclass
class SubQuery:
    """
    A single atomic sub-query produced by decomposition.
    Carries routing hints for the retrieval agent.
    """
    text:         str                      # The actual sub-query string
    focus_entity: str              = ""    # Primary company/entity focus
    metric_hint:  str              = ""    # Financial metric being queried
    time_range:   str              = ""    # Fiscal period (e.g., "Q3 FY26")
    priority:     int              = 1     # 1 = highest, 5 = lowest
    source_hint:  Optional[str]    = None  # Suggested source_type for retrieval

    def __post_init__(self):
        self.text = self.text.strip()

    def to_dict(self) -> dict:
        return {
            "text":         self.text,
            "focus_entity": self.focus_entity,
            "metric_hint":  self.metric_hint,
            "time_range":   self.time_range,
            "priority":     self.priority,
            "source_hint":  self.source_hint,
        }


@dataclass
class DecomposeResult:
    """
    Complete output of the Query Decomposer agent.

    Fields:
        sub_queries:    Ordered list of atomic sub-queries (max 6)
        strategy:       Which decomposition strategy was applied
        query_type:     Classified type of the original query
        reasoning:      Human-readable explanation of decomposition
        original_query: Input query (unchanged)
        used_llm:       Whether Groq was called (True = COMPOSITE path)
        duration_ms:    Total decomposer latency
        is_decomposed:  True if multiple sub-queries were produced
    """
    sub_queries:    list[SubQuery]
    strategy:       DecomposeStrategy
    query_type:     QueryType
    reasoning:      str
    original_query: str
    used_llm:       bool  = False
    duration_ms:    float = 0.0

    @property
    def is_decomposed(self) -> bool:
        return len(self.sub_queries) > 1

    @property
    def query_texts(self) -> list[str]:
        """Returns just the text strings — convenience for retrieval agent."""
        return [sq.text for sq in self.sub_queries]

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "strategy":       self.strategy.value,
            "query_type":     self.query_type.value,
            "is_decomposed":  self.is_decomposed,
            "sub_query_count": len(self.sub_queries),
            "sub_queries":    [sq.to_dict() for sq in self.sub_queries],
            "reasoning":      self.reasoning,
            "used_llm":       self.used_llm,
            "duration_ms":    round(self.duration_ms, 1),
        }

    def summary(self) -> str:
        lines = [
            f"[Agent2] {self.strategy.value} | "
            f"type={self.query_type.value} | "
            f"sub_queries={len(self.sub_queries)} | "
            f"llm={'yes' if self.used_llm else 'no'} | "
            f"{self.duration_ms:.0f}ms",
        ]
        for i, sq in enumerate(self.sub_queries, 1):
            lines.append(f"  [{i}] {sq.text[:80]}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# KNOWLEDGE BASES FOR RULE-BASED PATH
# ─────────────────────────────────────────────

# Known Indian financial companies / entities
# Format: canonical_name → set of aliases (all lowercase)
_ENTITY_ALIASES: dict[str, set[str]] = {
    "HDFC Bank":         {"hdfc bank", "hdfc", "hdfcbank"},
    "ICICI Bank":        {"icici bank", "icici", "icicibank"},
    "SBI":               {"sbi", "state bank", "state bank of india"},
    "Axis Bank":         {"axis bank", "axis"},
    "Kotak Mahindra":    {"kotak mahindra", "kotak", "kotak bank"},
    "IndusInd Bank":     {"indusind bank", "indusind"},
    "Bandhan Bank":      {"bandhan bank", "bandhan"},
    "Yes Bank":          {"yes bank", "yesbank"},
    "Federal Bank":      {"federal bank", "federal"},
    "PNB":               {"pnb", "punjab national bank"},
    "Bank of Baroda":    {"bank of baroda", "bob"},
    "Canara Bank":       {"canara bank", "canara"},
    "Union Bank":        {"union bank"},
    "RBL Bank":          {"rbl bank", "rbl"},
    "TCS":               {"tcs", "tata consultancy", "tata consultancy services"},
    "Infosys":           {"infosys", "infy"},
    "Wipro":             {"wipro"},
    "HCL Tech":          {"hcl tech", "hcl technologies", "hcl"},
    "Tech Mahindra":     {"tech mahindra", "techm"},
    "Reliance":          {"reliance", "ril", "reliance industries"},
    "Tata Motors":       {"tata motors", "tatamotors"},
    "Bajaj Finance":     {"bajaj finance", "bajajfin"},
    "Bajaj Finserv":     {"bajaj finserv", "bajajfinsv"},
    "HDFC Life":         {"hdfc life"},
    "SBI Life":          {"sbi life"},
    "LIC":               {"lic", "life insurance corporation"},
    "Sun Pharma":        {"sun pharma", "sunpharma"},
    "Cipla":             {"cipla"},
    "Divis Lab":         {"divis", "divi's"},
    "Bharti Airtel":     {"bharti airtel", "airtel"},
    "Adani Ports":       {"adani ports"},
    "Adani Green":       {"adani green"},
    "ONGC":              {"ongc"},
    "NTPC":              {"ntpc"},
    "PowerGrid":         {"powergrid", "power grid"},
    "Nifty 50":          {"nifty", "nifty50", "nifty 50"},
    "Sensex":            {"sensex"},
    "RBI":               {"rbi", "reserve bank"},
    "SEBI":              {"sebi"},
    "MCA":               {"mca"},
}

# Flat lookup: alias → canonical_name
_ENTITY_LOOKUP: dict[str, str] = {
    alias: canonical
    for canonical, aliases in _ENTITY_ALIASES.items()
    for alias in aliases
}

# Known financial metrics for rule-based multi-metric detection
_KNOWN_METRICS: frozenset[str] = frozenset({
    # Banking
    "nim", "nii", "npa", "gnpa", "nnpa", "casa", "crar", "car",
    "pcr", "lcr", "slr", "crr", "net interest margin",
    "non performing", "capital adequacy",
    # Returns
    "roe", "roa", "roce", "roic", "eps", "pat", "pbt", "bv",
    "return on equity", "return on assets", "earnings per share",
    # Valuation
    "pe", "pb", "ev", "ebitda", "ebitda margin", "p/e", "p/b",
    "price to earnings", "enterprise value",
    # Growth
    "revenue", "sales", "growth", "cagr", "yoy", "qoq",
    # Balance sheet
    "debt", "leverage", "fcf", "ocf", "capex", "working capital",
    # Insurance
    "vnb", "nbm", "aum", "gwp",
    # IT-specific
    "attrition", "utilization", "headcount", "deal wins", "tcv",
    "operating margin", "revenue growth",
    # Macro
    "gdp", "inflation", "repo rate", "cpi", "wpi",
})

# Comparison trigger words
_COMPARISON_WORDS: frozenset[str] = frozenset({
    "vs", "versus", "compare", "comparison", "compared to",
    "against", "relative to", "better than", "worse than",
    "difference between", "differentiate",
})

# Temporal trigger words / patterns
_TEMPORAL_WORDS: frozenset[str] = frozenset({
    "trend", "trends", "trending", "trajectory", "over time",
    "historical", "history", "over the years", "across quarters",
    "over last", "past quarters", "past years", "since",
    "year on year", "yoy", "quarter on quarter", "qoq",
    "fy24", "fy25", "fy26", "fy27",
    "q1", "q2", "q3", "q4",
})

# Fiscal period patterns (for extraction)
_FY_PATTERN    = re.compile(r'\bFY\s*(\d{2,4})\b', re.IGNORECASE)
_QUARTER_PATTERN = re.compile(r'\bQ([1-4])\s*(?:FY\s*(\d{2,4}))?\b', re.IGNORECASE)

# Classification signals
_REGULATORY_WORDS: frozenset[str] = frozenset({
    "sebi", "rbi", "irdai", "mca", "regulation", "circular", "compliance",
    "guideline", "rule", "act", "section", "policy", "directive",
    "notification", "mandate", "regulatory", "framework", "law",
    "penalty", "fine", "enforcement", "amendment",
})
_THESIS_WORDS: frozenset[str] = frozenset({
    "outperform", "underperform", "overvalued", "undervalued",
    "bullish", "bearish", "buy", "sell", "hold", "thesis",
    "investment case", "upside", "downside", "target price",
    "will rise", "will fall", "expect growth",
})
_ANALYTICAL_WORDS: frozenset[str] = frozenset({
    "why", "how", "reason", "because", "cause", "explain",
    "analysis", "impact", "effect", "influence", "driven by",
    "due to", "factors", "breakdown", "deep dive",
})

# Source routing hints per query type
_SOURCE_HINTS: dict[QueryType, str] = {
    QueryType.REGULATORY: "rbi,sebi,mca",
    QueryType.FACTUAL:    "annual_report,earnings_transcript",
    QueryType.COMPARATIVE: "annual_report,broker_research",
    QueryType.ANALYTICAL:  "annual_report,earnings_transcript,broker_research",
    QueryType.THESIS:      "annual_report,broker_research,news",
}

# Groq prompt template for COMPOSITE decomposition
_GROQ_DECOMPOSE_SYSTEM = """You are a financial query decomposer for an Indian markets RAG system.
Your task: break complex financial queries into atomic sub-queries for parallel retrieval.

Rules:
- Each sub-query must be self-contained and answerable independently
- Maximum 6 sub-queries; minimum 2 (never return 1 for a complex query)
- Each sub-query must retain time context (Q3 FY26, FY25 etc.) from original
- Keep entity names explicit in every sub-query (no pronouns)
- Focus on what information needs to be RETRIEVED, not what needs to be ANALYZED
- Sub-queries should be search-friendly phrases, not full sentences

Output ONLY valid JSON. No markdown. No explanation.

JSON format:
{
  "strategy": "TEMPORAL|MULTI_METRIC|COMPOSITE",
  "reasoning": "One sentence explaining decomposition approach",
  "sub_queries": [
    {
      "text": "HDFC Bank NIM Q3 FY26",
      "focus_entity": "HDFC Bank",
      "metric_hint": "NIM",
      "time_range": "Q3 FY26",
      "priority": 1
    }
  ]
}"""


# ─────────────────────────────────────────────
# QUERY DECOMPOSER CLASS
# ─────────────────────────────────────────────

class QueryDecomposer:
    """
    Agent 2: Decomposes complex financial queries into atomic sub-queries.

    Usage:
        from backend.agents.agent2_query_decomposer import query_decomposer

        result = await query_decomposer.decompose(
            "Compare HDFC vs ICICI NIM and NPA for Q3 FY26"
        )
        print(result.summary())
        # → [Agent2] COMPARISON | type=COMPARATIVE | sub_queries=4 | llm=no | 3ms
        #   [1] HDFC Bank NIM Q3 FY26
        #   [2] ICICI Bank NIM Q3 FY26
        #   [3] HDFC Bank NPA Q3 FY26
        #   [4] ICICI Bank NPA Q3 FY26

        for sq in result.sub_queries:
            chunks = await retriever.retrieve(sq.text, ...)
    """

    def __init__(self):
        # Runtime stats
        self._total_decomposed: int   = 0
        self._total_llm_calls:  int   = 0
        self._total_simple:     int   = 0
        self._total_ms:         float = 0.0

        logger.info(
            f"[DECOMPOSER] QueryDecomposer ready | "
            f"max_sub_queries={MAX_SUB_QUERIES}"
        )

    # ─────────────────────────────────────────
    # PUBLIC: decompose
    # ─────────────────────────────────────────

    async def decompose(
        self,
        query:       str,
        query_type:  Optional[QueryType] = None,
        use_llm:     bool                = True,
    ) -> DecomposeResult:
        """
        Main entry point. Decomposes a query into atomic sub-queries.

        Routing priority (first match wins):
            1. Not complex           → SIMPLE (returns original, no LLM)
            2. Comparison pattern    → COMPARISON (rule-based)
            3. Multi-entity pattern  → MULTI_ENTITY (rule-based)
            4. Multi-metric pattern  → MULTI_METRIC (rule-based)
            5. Temporal pattern      → TEMPORAL (rule-based)
            6. Fallthrough           → COMPOSITE (Groq LLM)

        Args:
            query:      Raw financial query string (pre-expanded by Agent 1)
            query_type: Pre-classified type (auto-detected if None)
            use_llm:    Allow Groq for COMPOSITE strategy (default True)

        Returns:
            DecomposeResult with sub_queries, strategy, query_type, reasoning

        Latency:
            Rule-based paths:  2-5ms
            LLM (COMPOSITE):   ~180ms

        Examples:
            # SIMPLE
            await decomposer.decompose("What is HDFC Bank NIM?")
            → sub_queries=["What is HDFC Bank NIM?"]  strategy=SIMPLE

            # COMPARISON
            await decomposer.decompose("Compare HDFC vs ICICI NIM and GNPA Q3 FY26")
            → sub_queries=["HDFC Bank NIM Q3 FY26", "ICICI Bank NIM Q3 FY26",
                           "HDFC Bank GNPA Q3 FY26", "ICICI Bank GNPA Q3 FY26"]

            # MULTI_METRIC
            await decomposer.decompose("HDFC Bank NIM GNPA CASA and CRAR for Q3 FY26")
            → sub_queries=["HDFC Bank NIM Q3 FY26", "HDFC Bank GNPA Q3 FY26",
                           "HDFC Bank CASA Q3 FY26", "HDFC Bank CRAR Q3 FY26"]

            # TEMPORAL
            await decomposer.decompose("HDFC Bank NIM trend over FY24 FY25 FY26")
            → sub_queries=["HDFC Bank NIM FY24", "HDFC Bank NIM FY25",
                           "HDFC Bank NIM FY26"]
        """
        if not query or not query.strip():
            return self._make_simple(query, QueryType.UNKNOWN, "Empty query")

        query       = query.strip()
        start_ms    = time.perf_counter()

        # Step 1: Classify query type
        detected_type = query_type or self.classify_query_type(query)

        # Step 2: Fast-path check — skip decomposition for simple queries
        if not self.is_complex_query(query):
            result = self._make_simple(
                query,
                detected_type,
                "Single entity and metric — no decomposition required",
            )
            result.duration_ms = (time.perf_counter() - start_ms) * 1000
            self._total_simple += 1
            self._log_result(result)
            return result

        # Step 3: Rule-based strategies (ordered by specificity)
        rule_result = self._try_rule_based(query, detected_type)

        if rule_result is not None:
            rule_result.duration_ms = (time.perf_counter() - start_ms) * 1000
            self._total_decomposed += 1
            self._log_result(rule_result)
            return rule_result

        # Step 4: Fallback — LLM-based COMPOSITE decomposition
        if use_llm:
            try:
                llm_result = await self._decompose_with_llm(
                    query       = query,
                    query_type  = detected_type,
                )
                llm_result.duration_ms = (time.perf_counter() - start_ms) * 1000
                self._total_decomposed += 1
                self._total_llm_calls  += 1
                self._log_result(llm_result)
                return llm_result

            except Exception as e:
                logger.warning(
                    f"[DECOMPOSER] LLM decomposition failed — falling back to SIMPLE | "
                    f"error={type(e).__name__}: {e}"
                )

        # Step 5: Ultimate fallback — return original query unchanged
        fallback = self._make_simple(
            query,
            detected_type,
            f"Fallback: LLM unavailable or failed — returning original query",
        )
        fallback.duration_ms = (time.perf_counter() - start_ms) * 1000
        self._log_result(fallback)
        return fallback

    # ─────────────────────────────────────────
    # PUBLIC: is_complex_query
    # ─────────────────────────────────────────

    def is_complex_query(self, query: str) -> bool:
        """
        Determines if a query needs decomposition.
        Returns False for simple single-entity, single-metric queries.

        Complexity signals (any one = complex):
            - Contains comparison trigger words (vs, compare, etc.)
            - Contains 2+ detected entities
            - Contains 3+ detected metrics
            - Contains 2+ fiscal year references
            - Contains 2+ quarter references
            - Word count > 18

        Args:
            query: Financial query string

        Returns:
            True if query is complex and should be decomposed

        Examples:
            is_complex_query("What is HDFC Bank NIM?")         → False
            is_complex_query("Compare HDFC vs ICICI NIM")      → True
            is_complex_query("HDFC NIM GNPA CASA Q3 FY26")     → True
            is_complex_query("RBI repo rate policy 2026")      → False
        """
        query_lower = query.lower()
        words       = set(query_lower.split())

        # Comparison words
        if words & _COMPARISON_WORDS:
            return True

        # Word count heuristic
        if len(query.split()) > 18:
            return True

        # Multiple entities
        entities = self._detect_entities(query)
        if len(entities) >= 2:
            return True

        # Multiple fiscal years
        fy_matches = _FY_PATTERN.findall(query)
        if len(fy_matches) >= 2:
            return True

        # Multiple quarters
        q_matches = _QUARTER_PATTERN.findall(query)
        if len(q_matches) >= 2:
            return True

        # Multiple known metrics
        metrics = self._detect_metrics(query)
        if len(metrics) >= 3:
            return True

        return False

    # ─────────────────────────────────────────
    # PUBLIC: classify_query_type
    # ─────────────────────────────────────────

    def classify_query_type(self, query: str) -> QueryType:
        """
        Classifies a query into one of the QueryType categories.
        Rule-based, no LLM required. Runs in <1ms.

        Classification priority (first match):
            1. COMPARATIVE  — has comparison words
            2. REGULATORY   — mentions regulatory bodies or frameworks
            3. THESIS       — contains investment thesis language
            4. ANALYTICAL   — contains "why/how/explain" language
            5. FACTUAL      — single entity + metric lookup (default)

        Args:
            query: Financial query string

        Returns:
            QueryType enum value

        Examples:
            classify("Compare HDFC vs ICICI NIM")          → COMPARATIVE
            classify("SEBI circular on F&O margins")        → REGULATORY
            classify("HDFC will outperform peers because")  → THESIS
            classify("Why did HDFC NIM compress in Q3?")    → ANALYTICAL
            classify("What is HDFC Bank NIM for Q3 FY26?")  → FACTUAL
        """
        query_lower = query.lower()
        words       = set(query_lower.split())

        if words & _COMPARISON_WORDS:
            return QueryType.COMPARATIVE

        # Check regulatory phrases (multi-word patterns first)
        regulatory_phrases = [
            "sebi circular", "rbi circular", "rbi policy", "rbi regulation",
            "sebi regulation", "rbi notification", "sebi notification",
        ]
        if any(p in query_lower for p in regulatory_phrases):
            return QueryType.REGULATORY
        if words & _REGULATORY_WORDS:
            return QueryType.REGULATORY

        # Thesis language
        thesis_phrases = [
            "will outperform", "will underperform", "is overvalued",
            "is undervalued", "investment case", "thesis",
        ]
        if any(p in query_lower for p in thesis_phrases):
            return QueryType.THESIS
        if words & _THESIS_WORDS:
            return QueryType.THESIS

        # Analytical language
        analytical_phrases = [
            "why did", "how did", "what caused", "explain why",
            "reason for", "due to", "driven by", "what led to",
        ]
        if any(p in query_lower for p in analytical_phrases):
            return QueryType.ANALYTICAL
        if words & _ANALYTICAL_WORDS:
            return QueryType.ANALYTICAL

        return QueryType.FACTUAL

    # ─────────────────────────────────────────
    # PRIVATE: try_rule_based
    # ─────────────────────────────────────────

    def _try_rule_based(
        self,
        query:      str,
        query_type: QueryType,
    ) -> Optional[DecomposeResult]:
        """
        Attempts rule-based decomposition in strategy priority order.
        Returns None if no rule-based strategy matches.
        """
        query_lower = query.lower()
        words       = set(query_lower.split())

        entities = self._detect_entities(query)
        metrics  = self._detect_metrics(query)
        fy_years = self._detect_fiscal_years(query)
        quarters = self._detect_quarters(query)
        time_ref = self._extract_time_context(query)

        # ── Strategy 1: COMPARISON ─────────────────────────────────────────
        if words & _COMPARISON_WORDS and len(entities) >= 2:
            return self._decompose_comparison(
                query, query_type, entities, metrics, time_ref
            )

        # ── Strategy 2: MULTI_ENTITY ───────────────────────────────────────
        if len(entities) >= 3:
            return self._decompose_multi_entity(
                query, query_type, entities, metrics, time_ref
            )

        # ── Strategy 3: MULTI_METRIC ───────────────────────────────────────
        if len(metrics) >= 3 and len(entities) >= 1:
            return self._decompose_multi_metric(
                query, query_type,
                primary_entity = entities[0] if entities else "",
                metrics        = metrics,
                time_ref       = time_ref,
            )

        # ── Strategy 4: TEMPORAL ───────────────────────────────────────────
        time_periods = fy_years or quarters
        if len(time_periods) >= 2:
            entity  = entities[0] if entities else ""
            metric  = metrics[0]  if metrics  else ""
            if entity or metric:
                return self._decompose_temporal(
                    query, query_type, entity, metric, time_periods
                )

        return None   # No rule matched → caller uses LLM

    # ─────────────────────────────────────────
    # PRIVATE: decompose_comparison
    # ─────────────────────────────────────────

    def _decompose_comparison(
        self,
        query:      str,
        query_type: QueryType,
        entities:   list[str],
        metrics:    list[str],
        time_ref:   str,
    ) -> DecomposeResult:
        """
        Handles "HDFC vs ICICI NIM and GNPA Q3 FY26"
        Cross-product: entity × metric (capped at MAX_SUB_QUERIES)

        Examples:
            entities=["HDFC Bank", "ICICI Bank"], metrics=["NIM", "GNPA"], time="Q3 FY26"
            → ["HDFC Bank NIM Q3 FY26", "ICICI Bank NIM Q3 FY26",
               "HDFC Bank GNPA Q3 FY26", "ICICI Bank GNPA Q3 FY26"]
        """
        cap_entities = entities[:3]   # Max 3 entities for cross-product
        cap_metrics  = metrics[:2]    # Max 2 metrics for cross-product

        sub_queries: list[SubQuery] = []
        time_suffix = f" {time_ref}" if time_ref else ""

        # If no metrics found, generate one sub-query per entity
        if not cap_metrics:
            for i, entity in enumerate(cap_entities):
                sub_queries.append(SubQuery(
                    text         = f"{entity}{time_suffix}",
                    focus_entity = entity,
                    metric_hint  = "",
                    time_range   = time_ref,
                    priority     = i + 1,
                    source_hint  = _SOURCE_HINTS.get(query_type),
                ))
        else:
            # Cross-product: entity × metric
            priority = 1
            for metric in cap_metrics:
                for entity in cap_entities:
                    if priority > MAX_SUB_QUERIES:
                        break
                    sub_queries.append(SubQuery(
                        text         = f"{entity} {metric}{time_suffix}",
                        focus_entity = entity,
                        metric_hint  = metric,
                        time_range   = time_ref,
                        priority     = priority,
                        source_hint  = _SOURCE_HINTS.get(query_type),
                    ))
                    priority += 1

        entity_list = " vs ".join(cap_entities)
        metric_list = ", ".join(cap_metrics) if cap_metrics else "general"

        return DecomposeResult(
            sub_queries    = sub_queries[:MAX_SUB_QUERIES],
            strategy       = DecomposeStrategy.COMPARISON,
            query_type     = query_type,
            reasoning      = (
                f"Comparison query: {entity_list} | "
                f"metrics: {metric_list} | "
                f"Generated {len(sub_queries)} cross-product sub-queries"
            ),
            original_query = query,
            used_llm       = False,
        )

    # ─────────────────────────────────────────
    # PRIVATE: decompose_multi_entity
    # ─────────────────────────────────────────

    def _decompose_multi_entity(
        self,
        query:      str,
        query_type: QueryType,
        entities:   list[str],
        metrics:    list[str],
        time_ref:   str,
    ) -> DecomposeResult:
        """
        Handles "HDFC, ICICI, Axis NIM Q3 FY26" (3+ entities, same metric)

        Example:
            entities=["HDFC Bank", "ICICI Bank", "Axis Bank"], metrics=["NIM"]
            → ["HDFC Bank NIM Q3 FY26", "ICICI Bank NIM Q3 FY26",
               "Axis Bank NIM Q3 FY26"]
        """
        primary_metric = metrics[0] if metrics else ""
        time_suffix    = f" {time_ref}" if time_ref else ""

        sub_queries: list[SubQuery] = []
        for i, entity in enumerate(entities[:MAX_SUB_QUERIES]):
            metric_part = f" {primary_metric}" if primary_metric else ""
            sub_queries.append(SubQuery(
                text         = f"{entity}{metric_part}{time_suffix}",
                focus_entity = entity,
                metric_hint  = primary_metric,
                time_range   = time_ref,
                priority     = i + 1,
                source_hint  = _SOURCE_HINTS.get(query_type),
            ))

        return DecomposeResult(
            sub_queries    = sub_queries,
            strategy       = DecomposeStrategy.MULTI_ENTITY,
            query_type     = query_type,
            reasoning      = (
                f"Multi-entity query: {len(entities)} entities detected | "
                f"metric: {primary_metric or 'general'} | "
                f"Generated {len(sub_queries)} entity-specific sub-queries"
            ),
            original_query = query,
            used_llm       = False,
        )

    # ─────────────────────────────────────────
    # PRIVATE: decompose_multi_metric
    # ─────────────────────────────────────────

    def _decompose_multi_metric(
        self,
        query:          str,
        query_type:     QueryType,
        primary_entity: str,
        metrics:        list[str],
        time_ref:       str,
    ) -> DecomposeResult:
        """
        Handles "HDFC Bank NIM GNPA CASA CRAR Q3 FY26" (1 entity, 4+ metrics)

        Example:
            entity="HDFC Bank", metrics=["NIM", "GNPA", "CASA", "CRAR"]
            → ["HDFC Bank NIM Q3 FY26", "HDFC Bank GNPA Q3 FY26",
               "HDFC Bank CASA Q3 FY26", "HDFC Bank CRAR Q3 FY26"]
        """
        time_suffix = f" {time_ref}" if time_ref else ""

        sub_queries: list[SubQuery] = []
        for i, metric in enumerate(metrics[:MAX_SUB_QUERIES]):
            entity_part = f"{primary_entity} " if primary_entity else ""
            sub_queries.append(SubQuery(
                text         = f"{entity_part}{metric}{time_suffix}",
                focus_entity = primary_entity,
                metric_hint  = metric,
                time_range   = time_ref,
                priority     = i + 1,
                source_hint  = _SOURCE_HINTS.get(query_type),
            ))

        return DecomposeResult(
            sub_queries    = sub_queries,
            strategy       = DecomposeStrategy.MULTI_METRIC,
            query_type     = query_type,
            reasoning      = (
                f"Multi-metric query: entity={primary_entity or 'general'} | "
                f"{len(metrics)} metrics detected: {', '.join(metrics[:5])} | "
                f"Generated {len(sub_queries)} metric-specific sub-queries"
            ),
            original_query = query,
            used_llm       = False,
        )

    # ─────────────────────────────────────────
    # PRIVATE: decompose_temporal
    # ─────────────────────────────────────────

    def _decompose_temporal(
        self,
        query:        str,
        query_type:   QueryType,
        entity:       str,
        metric:       str,
        time_periods: list[str],
    ) -> DecomposeResult:
        """
        Handles "HDFC NIM over FY24 FY25 FY26" (same entity+metric, multiple periods)

        Example:
            entity="HDFC Bank", metric="NIM", periods=["FY24", "FY25", "FY26"]
            → ["HDFC Bank NIM FY24", "HDFC Bank NIM FY25",
               "HDFC Bank NIM FY26"]
        """
        sub_queries: list[SubQuery] = []
        for i, period in enumerate(time_periods[:MAX_SUB_QUERIES]):
            entity_part = f"{entity} " if entity else ""
            metric_part = f"{metric} " if metric else ""
            sub_queries.append(SubQuery(
                text         = f"{entity_part}{metric_part}{period}",
                focus_entity = entity,
                metric_hint  = metric,
                time_range   = period,
                priority     = i + 1,
                source_hint  = _SOURCE_HINTS.get(query_type),
            ))

        return DecomposeResult(
            sub_queries    = sub_queries,
            strategy       = DecomposeStrategy.TEMPORAL,
            query_type     = query_type,
            reasoning      = (
                f"Temporal trend query: entity={entity or 'general'} | "
                f"metric={metric or 'general'} | "
                f"periods={time_periods[:6]} | "
                f"Generated {len(sub_queries)} time-period sub-queries"
            ),
            original_query = query,
            used_llm       = False,
        )

    # ─────────────────────────────────────────
    # PRIVATE: decompose_with_llm (COMPOSITE)
    # ─────────────────────────────────────────

    async def _decompose_with_llm(
        self,
        query:      str,
        query_type: QueryType,
    ) -> DecomposeResult:
        """
        Uses Groq (llama-3.3-70b) to decompose complex queries
        that don't fit clean rule-based patterns.

        Called only when all rule-based strategies fail to match.
        Parses structured JSON output from LLM.
        Falls back to SIMPLE if JSON parse fails.
        """
        context_hint = (
            f"Query type: {query_type.value}\n"
            f"Max sub-queries: {MAX_SUB_QUERIES}\n"
            f"Query to decompose: {query}"
        )

        logger.debug(
            f"[DECOMPOSER] Using LLM (COMPOSITE) | "
            f"query='{query[:60]}'"
        )

        raw: dict = await llm_client.call_llm_json(
            prompt        = context_hint,
            system_prompt = _GROQ_DECOMPOSE_SYSTEM,
            temperature   = 0.1,
            max_tokens    = 600,
            agent_name    = AGENT_NAME,
        )

        # Parse LLM response into SubQuery objects
        llm_sub_queries_raw: list[dict] = raw.get("sub_queries", [])
        strategy_str:        str        = raw.get("strategy", "COMPOSITE")
        reasoning:           str        = raw.get("reasoning", "LLM decomposition")

        if not llm_sub_queries_raw:
            raise ValueError("LLM returned empty sub_queries array")

        sub_queries: list[SubQuery] = []
        for i, sq_dict in enumerate(llm_sub_queries_raw[:MAX_SUB_QUERIES]):
            text = sq_dict.get("text", "").strip()
            if not text or len(text) < MIN_SUBQUERY_LEN:
                continue
            sub_queries.append(SubQuery(
                text         = text,
                focus_entity = sq_dict.get("focus_entity", ""),
                metric_hint  = sq_dict.get("metric_hint", ""),
                time_range   = sq_dict.get("time_range", ""),
                priority     = int(sq_dict.get("priority", i + 1)),
                source_hint  = _SOURCE_HINTS.get(query_type),
            ))

        if not sub_queries:
            raise ValueError("No valid sub-queries parsed from LLM response")

        # Map strategy string to enum
        try:
            strategy = DecomposeStrategy(strategy_str.upper())
        except ValueError:
            strategy = DecomposeStrategy.COMPOSITE

        return DecomposeResult(
            sub_queries    = sub_queries,
            strategy       = strategy,
            query_type     = query_type,
            reasoning      = reasoning,
            original_query = query,
            used_llm       = True,
        )

    # ─────────────────────────────────────────
    # PRIVATE: entity/metric/time detectors
    # ─────────────────────────────────────────

    def _detect_entities(self, query: str) -> list[str]:
        """
        Detects known financial entities in query.
        Returns ordered list of canonical entity names (order of appearance).
        Deduplicates: only first occurrence counted.

        Examples:
            _detect_entities("Compare HDFC vs ICICI NIM")
            → ["HDFC Bank", "ICICI Bank"]
        """
        query_lower   = query.lower()
        found:   list[str] = []
        seen:    set[str]  = set()

        # Sort by alias length descending (match "hdfc bank" before "hdfc")
        sorted_aliases = sorted(
            _ENTITY_LOOKUP.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for alias, canonical in sorted_aliases:
            # Word boundary match for short aliases (2-4 chars)
            if len(alias) <= 4:
                pattern = rf'\b{re.escape(alias)}\b'
            else:
                pattern = re.escape(alias)

            if re.search(pattern, query_lower):
                if canonical not in seen:
                    seen.add(canonical)
                    found.append(canonical)

        # Preserve order of appearance by position
        def _first_pos(canonical: str) -> int:
            for alias, can in sorted_aliases:
                if can == canonical:
                    m = re.search(re.escape(alias), query_lower)
                    if m:
                        return m.start()
            return 9999

        found.sort(key=_first_pos)
        return found

    def _detect_metrics(self, query: str) -> list[str]:
        """
        Detects known financial metrics in query.
        Returns list of uppercase metric names in order of appearance.

        Examples:
            _detect_metrics("HDFC Bank NIM GNPA CASA Q3 FY26")
            → ["NIM", "GNPA", "CASA"]
        """
        query_lower = query.lower()
        found: list[tuple[int, str]] = []   # (position, metric_upper)
        seen:  set[str] = set()

        # Sort by length descending (match "ebitda margin" before "ebitda")
        sorted_metrics = sorted(_KNOWN_METRICS, key=len, reverse=True)

        for metric in sorted_metrics:
            if len(metric) <= 4:
                pattern = rf'\b{re.escape(metric)}\b'
            else:
                pattern = re.escape(metric)

            m = re.search(pattern, query_lower)
            if m:
                canonical_upper = metric.upper().replace(" ", "_")
                # For short acronyms, keep as-is
                if " " not in metric:
                    canonical_upper = metric.upper()
                if canonical_upper not in seen:
                    seen.add(canonical_upper)
                    found.append((m.start(), canonical_upper))

        found.sort(key=lambda x: x[0])
        return [m for _, m in found]

    def _detect_fiscal_years(self, query: str) -> list[str]:
        """
        Extracts fiscal year references from query.
        Returns ordered list like ["FY24", "FY25", "FY26"].
        """
        matches = _FY_PATTERN.findall(query)
        seen: set[str] = set()
        result: list[str] = []
        for yr in matches:
            fy = f"FY{yr}"
            if fy not in seen:
                seen.add(fy)
                result.append(fy)
        return result

    def _detect_quarters(self, query: str) -> list[str]:
        """
        Extracts quarter references like "Q3 FY26", "Q1", "Q4 FY25".
        Returns ordered list.
        """
        matches = _QUARTER_PATTERN.findall(query)
        seen: set[str] = set()
        result: list[str] = []
        for q_num, fy_num in matches:
            period = f"Q{q_num}"
            if fy_num:
                period = f"Q{q_num} FY{fy_num}"
            if period not in seen:
                seen.add(period)
                result.append(period)
        return result

    def _extract_time_context(self, query: str) -> str:
        """
        Extracts the most specific time context from query.
        Priority: Q+FY > standalone Q > standalone FY > ""

        Examples:
            "HDFC NIM Q3 FY26"     → "Q3 FY26"
            "HDFC NIM FY26"        → "FY26"
            "HDFC NIM last quarter" → ""
        """
        # Q+FY combined: "Q3 FY26"
        qfy = re.search(r'\bQ([1-4])\s+FY\s*(\d{2,4})\b', query, re.IGNORECASE)
        if qfy:
            return f"Q{qfy.group(1)} FY{qfy.group(2)}"

        # Standalone quarter
        q = re.search(r'\bQ([1-4])\b', query, re.IGNORECASE)
        if q:
            return f"Q{q.group(1)}"

        # Standalone FY
        fy = re.search(r'\bFY\s*(\d{2,4})\b', query, re.IGNORECASE)
        if fy:
            return f"FY{fy.group(1)}"

        return ""

    # ─────────────────────────────────────────
    # PRIVATE: helpers
    # ─────────────────────────────────────────

    def _make_simple(
        self,
        query:      str,
        query_type: QueryType,
        reasoning:  str,
    ) -> DecomposeResult:
        """Creates a SIMPLE DecomposeResult with the original query as-is."""
        time_ref = self._extract_time_context(query)
        entities = self._detect_entities(query)
        metrics  = self._detect_metrics(query)

        return DecomposeResult(
            sub_queries = [SubQuery(
                text         = query,
                focus_entity = entities[0] if entities else "",
                metric_hint  = metrics[0]  if metrics  else "",
                time_range   = time_ref,
                priority     = 1,
                source_hint  = _SOURCE_HINTS.get(query_type),
            )],
            strategy       = DecomposeStrategy.SIMPLE,
            query_type     = query_type,
            reasoning      = reasoning,
            original_query = query,
            used_llm       = False,
        )

    def _log_result(self, result: DecomposeResult) -> None:
        """Logs decomposition result and updates stats."""
        self._total_ms += result.duration_ms
        logger.info(
            f"[DECOMPOSER] {result.strategy.value} | "
            f"type={result.query_type.value} | "
            f"sub_queries={len(result.sub_queries)} | "
            f"llm={'yes' if result.used_llm else 'no'} | "
            f"{result.duration_ms:.1f}ms"
        )
        for sq in result.sub_queries:
            logger.debug(f"[DECOMPOSER]   → '{sq.text}'")

        log_metric("decomposer_duration_ms",    result.duration_ms,         unit="ms")
        log_metric("decomposer_sub_query_count", len(result.sub_queries),   unit="count")
        log_metric("decomposer_llm_used",        int(result.used_llm),      unit="bool")

    # ─────────────────────────────────────────
    # PUBLIC: get_stats
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns runtime statistics for monitoring."""
        total = self._total_simple + self._total_decomposed
        avg_ms = round(self._total_ms / total, 1) if total > 0 else 0.0
        return {
            "total_queries":    total,
            "total_simple":     self._total_simple,
            "total_decomposed": self._total_decomposed,
            "total_llm_calls":  self._total_llm_calls,
            "total_ms":         round(self._total_ms, 1),
            "avg_ms_per_query": avg_ms,
        }


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

query_decomposer = QueryDecomposer()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "query_decomposer",
    "QueryDecomposer",
    "DecomposeResult",
    "DecomposeStrategy",
    "SubQuery",
]
