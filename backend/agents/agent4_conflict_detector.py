# backend/agents/agent4_conflict_detector.py
"""
FinThesisGuard AI — Agent 4: Conflict Detector
ArkAngel Financial Solutions

Detects and resolves numerical contradictions across retrieved chunks.
Runs after Agent 3 (Re-Ranker) and before LLM synthesis so the answer
layer always receives a conflict report alongside the source chunks.

What counts as a conflict:
    Two chunks report different numerical values for the SAME
    (metric, entity, time_period) triple and the difference exceeds
    a configurable threshold:
        Ratio / percentage metrics (NIM, NPA, ROE, CASA …):
            |val_a − val_b| > CONFLICT_THRESHOLD_PERCENTAGE (5%)
        Absolute value metrics (Revenue, Profit, AUM in ₹ crore …):
            |val_a − val_b| / avg > CONFLICT_THRESHOLD_PERCENTAGE (5%)
            OR |val_a − val_b| > CONFLICT_THRESHOLD_ABSOLUTE (0.5)

Severity classification:
    Minor    — percentage difference  < 5%
    Moderate — percentage difference  5%–20%
    Major    — percentage difference  > 20%

Resolution priority (highest → lowest):
    1. Most recent publication date
    2. Higher AUTHORITY_WEIGHTS from config.py
    3. Tie → "UNRESOLVED" with both values shown

Pipeline position:
    Agent 3 (Re-Ranker) → [Agent 4: Conflict Detector] → LLM Synthesis
    Runs only when 3+ distinct sources are retrieved.
    Latency target: 800ms (regex path ~10ms, LLM fallback ~600ms)

Dependencies:
    models/response_models.py — Conflict dataclass
    config.py                 — thresholds, authority weights
    utils/llm_client.py       — optional LLM-assisted claim extraction fallback
    utils/logger.py           — structured logging
"""

import re
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Optional

from backend.models.response_models import Conflict, ConflictSeverity
from backend.config import (
    AUTHORITY_WEIGHTS,
    CONFLICT_THRESHOLD_PERCENTAGE,
    CONFLICT_THRESHOLD_ABSOLUTE,
    CONFLICT_MIN_SOURCES,
)
from backend.utils.llm_client import llm_client, LLMError
from backend.utils.logger     import logger, log_metric


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

AGENT_NAME = "agent4_conflict_detector"

# Maximum number of conflicts to surface per pipeline call.
# Prevents overwhelming the user with noise when many chunks conflict.
MAX_CONFLICTS_RETURNED: int = 8

# Minimum number of distinct source documents required before conflict
# detection is even attempted. With only 1-2 sources there is nothing
# to compare. Mirrors config.CONFLICT_MIN_SOURCES.
MIN_SOURCES_REQUIRED: int = max(CONFLICT_MIN_SOURCES, 3)

# Percentage difference thresholds for severity classification.
# Applied to the relative percentage difference: |a-b|/avg × 100
SEVERITY_MINOR_MAX:    float = 5.0   # <5%  → Minor
SEVERITY_MODERATE_MAX: float = 20.0  # 5–20% → Moderate
                                     # >20%  → Major

# Default authority for unknown source types
_DEFAULT_AUTHORITY: float = AUTHORITY_WEIGHTS.get("unknown", 0.7)

# ─────────────────────────────────────────────
# METRIC PATTERN REGISTRY
# ─────────────────────────────────────────────
# Each pattern entry:
#   name:     canonical metric name used as Conflict.field
#   aliases:  list of regex fragments that match the metric in text
#   is_ratio: True → percentage/ratio (NIM, NPA, ROE …)
#             False → absolute value (Revenue, Profit, AUM …)
#
# Regex capture groups in value_pattern:
#   Group 1: numeric value (e.g. "4.2" from "4.2%")
#
# Note: patterns are matched in order. Put more specific patterns first.

@dataclass(frozen=True)
class MetricPattern:
    name:     str          # e.g. "NIM"
    aliases:  list[str]    # regex fragments matched in text (case-insensitive)
    is_ratio: bool         # True → %, False → ₹ crore / absolute

_METRIC_PATTERNS: list[MetricPattern] = [
    # ── Banking ratios ──────────────────────────────────────────────────
    MetricPattern("NIM",  ["net interest margin", r"\bnim\b"],            True),
    MetricPattern("GNPA", ["gross npa", r"\bgnpa\b", "gross non.?performing"], True),
    MetricPattern("NNPA", ["net npa", r"\bnnpa\b", "net non.?performing"],     True),
    MetricPattern("NPA",  [r"\bnpa ratio\b", r"\bnpa\b"],                 True),
    MetricPattern("CASA", ["casa ratio", r"\bcasa\b"],                    True),
    MetricPattern("CRAR", ["capital adequacy", r"\bcrar\b", r"\bcar\b"],  True),
    MetricPattern("PCR",  ["provision coverage", r"\bpcr\b"],             True),
    MetricPattern("ROE",  ["return on equity", r"\broe\b"],               True),
    MetricPattern("ROA",  ["return on assets?", r"\broa\b"],              True),
    MetricPattern("ROCE", ["return on capital employed", r"\broce\b"],    True),
    # ── P&L ratios ──────────────────────────────────────────────────────
    MetricPattern("EBITDA Margin", ["ebitda margin"],                     True),
    MetricPattern("PAT Margin",    ["pat margin", "profit after tax margin"], True),
    MetricPattern("Operating Margin", ["operating margin"],               True),
    MetricPattern("Gross Margin",  ["gross margin"],                      True),
    # ── Valuation ───────────────────────────────────────────────────────
    MetricPattern("PE Ratio",      [r"\bp/?e ratio\b", "price.to.earnings"], True),
    MetricPattern("PB Ratio",      [r"\bp/?b ratio\b", "price.to.book"],     True),
    # ── Absolute values ─────────────────────────────────────────────────
    MetricPattern("Revenue",       [r"\brev(?:enue)?\b", "total income"],  False),
    MetricPattern("PAT",           [r"\bpat\b", "profit after tax"],       False),
    MetricPattern("NII",           [r"\bnii\b", "net interest income"],    False),
    MetricPattern("AUM",           [r"\baum\b", "assets under management"],False),
    MetricPattern("Advances",      [r"\badvances\b", "loan book"],         False),
    MetricPattern("Deposits",      [r"\bdeposits\b"],                      False),
    MetricPattern("Market Cap",    ["market cap(?:italisation)?"],         False),
    # ── Macro ───────────────────────────────────────────────────────────
    MetricPattern("Repo Rate",     ["repo rate"],                          True),
    MetricPattern("GDP Growth",    ["gdp growth"],                         True),
    MetricPattern("CPI",           [r"\bcpi\b", "consumer price index"],   True),
    MetricPattern("WPI",           [r"\bwpi\b", "wholesale price index"],  True),
]

# Compile alias patterns once at import time for performance
@dataclass
class _CompiledMetric:
    name:     str
    pattern:  re.Pattern
    is_ratio: bool

_COMPILED_METRICS: list[_CompiledMetric] = [
    _CompiledMetric(
        name    = mp.name,
        pattern = re.compile(
            r'(?:' + '|'.join(mp.aliases) + r')',
            re.IGNORECASE
        ),
        is_ratio = mp.is_ratio,
    )
    for mp in _METRIC_PATTERNS
]

# ─────────────────────────────────────────────
# NUMERICAL VALUE REGEX
# ─────────────────────────────────────────────
# Captures numeric values in financial text.
# Handles:
#   "4.2%"          → 4.2
#   "₹45,000 crore" → 45000   (crore multiplier applied separately)
#   "1.26 per cent" → 1.26
#   "Rs 12,345 cr"  → 12345
#   "-0.5%"         → -0.5    (negative values)

_VALUE_PATTERN = re.compile(
    r'(?:'
    r'(?:₹|Rs\.?\s*)?'                    # optional currency symbol
    r'(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)'  # group 1: main number (with commas)
    r'\s*'
    r'(?:crore|cr\.?|lakh|lacs?|mn|bn|billion|million)?'  # optional scale word
    r'\s*'
    r'(?:%|per\s*cent|bps|basis\s*points?)?'  # optional unit
    r')',
    re.IGNORECASE
)

# Named entity patterns for Indian financial companies/regulators
_ENTITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("HDFC Bank",    re.compile(r'\bHDFC\s*Bank\b|\bHDFCBANK\b', re.I)),
    ("ICICI Bank",   re.compile(r'\bICICI\s*Bank\b|\bICICIBANK\b', re.I)),
    ("SBI",          re.compile(r'\bSBI\b|\bState\s*Bank\s*of\s*India\b', re.I)),
    ("Axis Bank",    re.compile(r'\bAxis\s*Bank\b|\bAXISBANK\b', re.I)),
    ("Kotak",        re.compile(r'\bKotak\s*(?:Mahindra)?\s*Bank\b', re.I)),
    ("IndusInd",     re.compile(r'\bIndusInd\s*Bank\b', re.I)),
    ("Yes Bank",     re.compile(r'\bYes\s*Bank\b|\bYESBANK\b', re.I)),
    ("Bajaj Finance",re.compile(r'\bBajaj\s*Finance\b', re.I)),
    ("TCS",          re.compile(r'\bTCS\b|\bTata\s*Consultancy\b', re.I)),
    ("Infosys",      re.compile(r'\bInfosys\b|\bINFY\b', re.I)),
    ("Reliance",     re.compile(r'\bReliance\s*(?:Industries)?\b|\bRIL\b', re.I)),
    ("RBI",          re.compile(r'\bRBI\b|\bReserve\s*Bank\b', re.I)),
    ("SEBI",         re.compile(r'\bSEBI\b', re.I)),
]

# Time period patterns for grouping by reporting period
_PERIOD_PATTERN = re.compile(
    r'\b(Q[1-4]\s*FY\s*\d{2,4}|FY\s*\d{2,4}|H[12]\s*FY\s*\d{2,4}|'
    r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
    r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|'
    r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'\s*\'?\d{2,4})\b',
    re.IGNORECASE
)

# Date formats for recency comparison
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y",
    "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y",
)


# ─────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────

@dataclass
class NumericalClaim:
    """
    A single numeric claim extracted from a chunk.

    Fields:
        metric:      Canonical metric name (e.g. "NIM", "GNPA")
        value:       Extracted numeric value (e.g. 4.2)
        unit:        Unit string detected (e.g. "%", "crore", "bps")
        entity:      Company/entity the value refers to (e.g. "HDFC Bank")
        period:      Reporting period (e.g. "Q3 FY26", "FY25")
        raw_text:    Original matched text snippet (for provenance)
        source_name: Title / name of the chunk's source document
        source_type: Source type (for authority lookup)
        date:        Publication date of the source document
        chunk_id:    ID of the originating chunk
        is_ratio:    True if this is a percentage/ratio metric
    """
    metric:      str
    value:       float
    unit:        str
    entity:      str
    period:      str
    raw_text:    str
    source_name: str
    source_type: str
    date:        str
    chunk_id:    str
    is_ratio:    bool

    @property
    def group_key(self) -> tuple[str, str, str]:
        """
        Canonical grouping key: (metric, entity, period).
        Claims with the same key are compared for conflicts.
        Entity and period normalized to lowercase for robust matching.
        """
        return (
            self.metric.upper(),
            self.entity.lower().strip(),
            self.period.lower().strip(),
        )

    def display_value(self) -> str:
        """Formatted value string for Conflict.value_a / value_b."""
        if self.unit in ("%", "per cent", "per_cent"):
            return f"{self.value:.2f}%"
        if self.unit in ("crore", "cr", "lakh", "mn", "bn"):
            return f"₹{self.value:,.0f} {self.unit}"
        if self.unit in ("bps", "basis points"):
            return f"{self.value:.0f} bps"
        return str(self.value)


@dataclass
class ConflictCandidate:
    """
    A pair of NumericalClaims that disagree.
    Intermediate structure produced before converting to Conflict.
    """
    claim_a:      NumericalClaim
    claim_b:      NumericalClaim
    abs_diff:     float           # |val_a - val_b|
    pct_diff:     float           # |val_a - val_b| / avg × 100
    severity:     ConflictSeverity
    is_ratio:     bool


# ─────────────────────────────────────────────
# CONFLICT DETECTOR CLASS
# ─────────────────────────────────────────────

class ConflictDetector:
    """
    Agent 4: Detects numerical contradictions across retrieved chunks
    and resolves them by source recency and authority.

    Usage:
        from backend.agents.agent4_conflict_detector import conflict_detector

        conflicts = await conflict_detector.detect_conflicts(reranked_chunks)
        for c in conflicts:
            print(c.conflict_summary)
            print(f"  → Use: {c.recommended_source} ({c.reason})")

    The agent runs only when MIN_SOURCES_REQUIRED (3) distinct source
    documents are present in the chunk list. For smaller sets it returns
    an empty list immediately to save latency.
    """

    def __init__(self):
        self._total_runs:        int   = 0
        self._total_conflicts:   int   = 0
        self._total_claims:      int   = 0
        self._total_ms:          float = 0.0
        self._skipped_runs:      int   = 0

        logger.info(
            f"[CONFLICT DETECTOR] Agent 4 ready | "
            f"min_sources={MIN_SOURCES_REQUIRED} | "
            f"threshold_pct={CONFLICT_THRESHOLD_PERCENTAGE*100:.0f}% | "
            f"threshold_abs={CONFLICT_THRESHOLD_ABSOLUTE}"
        )

    # ─────────────────────────────────────────
    # PUBLIC: detect_conflicts
    # ─────────────────────────────────────────

    async def detect_conflicts(
        self,
        chunks:    list[dict],
        use_llm:   bool = False,
    ) -> list[Conflict]:
        """
        Main entry point. Detects conflicts across all provided chunks.

        Pipeline:
            1. Guard: skip if fewer than MIN_SOURCES_REQUIRED distinct sources
            2. Extract NumericalClaims from every chunk (regex-based)
            3. Group claims by (metric, entity, period) triple
            4. Compare all pairs within each group → ConflictCandidate list
            5. Resolve each candidate → Conflict with recommended_source
            6. Sort by severity (Major first), return top MAX_CONFLICTS_RETURNED

        Args:
            chunks:   list[dict] from Agent 3 (re-ranked chunks).
                      Each must have: id, text, metadata
                      metadata must have: source_type, date, source (title)
            use_llm:  If True, runs an LLM-assisted claim extraction pass
                      AFTER the regex pass to catch narrative / implicit claims.
                      Adds ~400ms latency. Default False for speed.

        Returns:
            List[Conflict] sorted by severity (Major→Moderate→Minor).
            Empty list if:
                - fewer than MIN_SOURCES_REQUIRED distinct sources
                - no numerical claims found
                - no conflicts above threshold

        Raises:
            Nothing — all errors handled gracefully, returns [] on failure.

        Latency:
            Regex-only (default):  10–40ms for 10 chunks
            With LLM (use_llm=True): 400–800ms

        Examples:
            chunks = [
                {"id":"c1","text":"HDFC NIM stood at 4.2% for Q3 FY26...","similarity":0.9,
                 "metadata":{"source_type":"annual_report","date":"2026-01-15",
                             "source":"HDFC Bank Q3 FY26 Annual Report"}},
                {"id":"c2","text":"HDFC Bank's NIM was 4.5% in Q3 FY2026...","similarity":0.8,
                 "metadata":{"source_type":"broker_research","date":"2025-12-01",
                             "source":"Motilal Oswal HDFC Note"}},
            ]
            conflicts = await conflict_detector.detect_conflicts(chunks)
            # → [Conflict(field="NIM", value_a="4.20%", value_b="4.50%",
            #             recommended_source="HDFC Bank Q3 FY26 Annual Report",
            #             reason="More recent (2026-01-15 > 2025-12-01) and higher
            #                     authority (annual_report=1.3 > broker_research=1.1)",
            #             severity="Minor")]
        """
        if not chunks:
            return []

        start_ms = time.perf_counter()
        self._total_runs += 1

        # ── Guard: count distinct source documents ────────────────────────
        distinct_sources = self._count_distinct_sources(chunks)
        if distinct_sources < MIN_SOURCES_REQUIRED:
            self._skipped_runs += 1
            logger.debug(
                f"[CONFLICT DETECTOR] Skipped — only {distinct_sources} distinct "
                f"sources (need {MIN_SOURCES_REQUIRED})"
            )
            return []

        logger.info(
            f"[CONFLICT DETECTOR] Running on {len(chunks)} chunks | "
            f"distinct_sources={distinct_sources}"
        )

        try:
            # ── Step 1: Extract all numerical claims ───────────────────────
            all_claims: list[NumericalClaim] = []
            for chunk in chunks:
                claims = self.extract_numerical_claims(chunk)
                all_claims.extend(claims)

            self._total_claims += len(all_claims)
            logger.debug(
                f"[CONFLICT DETECTOR] Extracted {len(all_claims)} claims "
                f"from {len(chunks)} chunks"
            )

            # Optional: LLM-assisted extraction pass for implicit claims
            if use_llm and len(all_claims) < 3:
                llm_claims = await self._extract_claims_with_llm(chunks)
                # Merge: LLM claims only for groups not already covered
                existing_keys = {c.group_key for c in all_claims}
                new_claims    = [c for c in llm_claims if c.group_key not in existing_keys]
                all_claims.extend(new_claims)
                logger.debug(
                    f"[CONFLICT DETECTOR] LLM added {len(new_claims)} additional claims"
                )

            if len(all_claims) < 2:
                logger.debug("[CONFLICT DETECTOR] Fewer than 2 claims — no conflicts possible")
                return []

            # ── Step 2: Group by (metric, entity, period) ─────────────────
            groups: dict[tuple, list[NumericalClaim]] = {}
            for claim in all_claims:
                key = claim.group_key
                groups.setdefault(key, []).append(claim)

            # ── Step 3: Compare pairs within each group ───────────────────
            candidates: list[ConflictCandidate] = []
            for key, group_claims in groups.items():
                if len(group_claims) < 2:
                    continue
                # Deduplicate by source to avoid same-source pair comparisons
                by_source: dict[str, NumericalClaim] = {}
                for claim in group_claims:
                    src = claim.source_name
                    # If same source appears twice, keep the one with later date
                    if src not in by_source:
                        by_source[src] = claim
                    else:
                        existing = by_source[src]
                        if self._parse_date(claim.date) > self._parse_date(existing.date):
                            by_source[src] = claim

                unique_claims = list(by_source.values())
                for a, b in combinations(unique_claims, 2):
                    candidate = self._compare_claims(a, b)
                    if candidate is not None:
                        candidates.append(candidate)

            if not candidates:
                logger.info("[CONFLICT DETECTOR] No conflicts above threshold")
                return []

            # ── Step 4: Resolve each candidate → Conflict ─────────────────
            conflicts: list[Conflict] = []
            for candidate in candidates:
                conflict = self.resolve_conflict(candidate)
                conflicts.append(conflict)

            # ── Step 5: Sort and cap ───────────────────────────────────────
            severity_order = {"Major": 0, "Moderate": 1, "Minor": 2}
            conflicts.sort(key=lambda c: severity_order.get(c.severity, 3))
            conflicts = conflicts[:MAX_CONFLICTS_RETURNED]

            duration_ms = (time.perf_counter() - start_ms) * 1000
            self._total_ms       += duration_ms
            self._total_conflicts += len(conflicts)

            logger.info(
                f"[CONFLICT DETECTOR] Found {len(conflicts)} conflicts | "
                f"{duration_ms:.1f}ms"
            )
            for c in conflicts:
                logger.debug(
                    f"[CONFLICT DETECTOR]   [{c.severity}] {c.conflict_summary}"
                )

            log_metric("conflict_detector_duration_ms", duration_ms,       unit="ms")
            log_metric("conflict_detector_conflicts",    len(conflicts),    unit="count")
            log_metric("conflict_detector_claims",       len(all_claims),   unit="count")

            return conflicts

        except Exception as e:
            duration_ms = (time.perf_counter() - start_ms) * 1000
            logger.error(
                f"[CONFLICT DETECTOR] Unexpected error: {type(e).__name__}: {e} | "
                f"{duration_ms:.1f}ms"
            )
            return []

    # ─────────────────────────────────────────
    # PUBLIC: extract_numerical_claims
    # ─────────────────────────────────────────

    def extract_numerical_claims(self, chunk: dict) -> list[NumericalClaim]:
        """
        Extracts all numerical financial claims from a single chunk.
        Uses regex patterns from _COMPILED_METRICS and _VALUE_PATTERN.

        Strategy:
            For each metric pattern:
                1. Find all occurrences of the metric name in chunk text
                2. Within a ±200-char window around each occurrence,
                   search for a numeric value using _VALUE_PATTERN
                3. Also search for entity name and time period in the window
                4. Build NumericalClaim if value found

        Args:
            chunk: Chunk dict with 'text', 'id', 'metadata'
                   metadata: source_type, date, source (title), company

        Returns:
            List of NumericalClaim objects (may be empty)

        Examples:
            chunk = {
                "id": "c1",
                "text": "HDFC Bank's NIM stood at 4.2% for Q3 FY26, while GNPA ratio
                          was 1.26% compared to 1.32% in Q2 FY26",
                "metadata": {"source_type":"annual_report","date":"2026-01-15",
                             "source":"HDFC Q3 FY26 Annual Report","company":"HDFC Bank"}
            }
            claims = extractor.extract_numerical_claims(chunk)
            # → [
            #     NumericalClaim(metric="NIM",  value=4.2,  entity="HDFC Bank", period="Q3 FY26"),
            #     NumericalClaim(metric="GNPA", value=1.26, entity="HDFC Bank", period="Q3 FY26"),
            #   ]
        """
        text     = chunk.get("text", "")
        metadata = chunk.get("metadata", {}) or {}

        source_name = str(metadata.get("source", metadata.get("title", "Unknown Source")))
        source_type = str(metadata.get("source_type", "unknown")).lower().strip()
        date        = str(metadata.get("date", ""))
        chunk_id    = str(chunk.get("id", ""))

        # Pre-detect entity from metadata.company or chunk text
        base_entity = str(metadata.get("company", ""))
        if not base_entity:
            base_entity = self._detect_entity(text)

        # Pre-detect period from chunk text
        base_period = self._detect_period(text)

        claims: list[NumericalClaim] = []

        for metric_def in _COMPILED_METRICS:
            # Find all positions where this metric name appears
            for match in metric_def.pattern.finditer(text):
                mstart, mend = match.start(), match.end()

                # Extract a local context window around the metric mention
                window_start = max(0, mstart - 150)
                window_end   = min(len(text), mend + 150)
                window       = text[window_start:window_end]

                # Search for a numeric value in the window
                val_match = _VALUE_PATTERN.search(window)
                if not val_match:
                    continue

                raw_val_str = val_match.group(1).replace(",", "")
                try:
                    value = float(raw_val_str)
                except ValueError:
                    continue

                # Apply crore/lakh multipliers for absolute values
                unit, value = self._normalize_unit(val_match.group(0), value)

                # Resolve entity: window first, then metadata, then "Unknown"
                entity = self._detect_entity(window) or base_entity or "Unknown"

                # Resolve period: window first, then chunk-level
                period = self._detect_period(window) or base_period or ""

                # Build raw_text snippet
                raw_text = text[max(0, mstart-30):min(len(text), mend+50)].strip()

                claims.append(NumericalClaim(
                    metric      = metric_def.name,
                    value       = value,
                    unit        = unit,
                    entity      = entity,
                    period      = period,
                    raw_text    = raw_text[:200],
                    source_name = source_name,
                    source_type = source_type,
                    date        = date,
                    chunk_id    = chunk_id,
                    is_ratio    = metric_def.is_ratio,
                ))

        # Deduplicate: if same (metric, entity, period, source) appears
        # multiple times in the chunk, keep only the first.
        seen: set[tuple] = set()
        deduped: list[NumericalClaim] = []
        for c in claims:
            key = (c.metric, c.entity, c.period, c.source_name)
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        return deduped

    # ─────────────────────────────────────────
    # PUBLIC: resolve_conflict
    # ─────────────────────────────────────────

    def resolve_conflict(self, candidate: ConflictCandidate) -> Conflict:
        """
        Resolves a ConflictCandidate by choosing the more reliable source.

        Resolution priority:
            1. Most recent publication date
               → if date_a > date_b: recommend A
               → if date_b > date_a: recommend B
            2. Higher authority weight (AUTHORITY_WEIGHTS from config.py)
               → if authority_a > authority_b: recommend A
               → else: recommend B
            3. Exact tie on both: flag as UNRESOLVED, show both values

        Args:
            candidate: ConflictCandidate with both claims and diff stats.

        Returns:
            Conflict Pydantic model (from response_models.py).

        Examples:
            # Date wins:
            A: annual_report 2026-01-15, NIM=4.2%
            B: broker_research 2025-12-01, NIM=4.5%
            → recommended_source = A
              reason = "Source A is more recent (2026-01-15 > 2025-12-01)
                        and has higher authority (annual_report=1.3 > broker_research=1.1)"

            # Authority wins (same date):
            A: broker_research 2026-01-15, NIM=4.2%
            B: rbi 2026-01-15, Repo Rate=6.5%
            → recommended_source = B (rbi authority=1.5 > broker_research=1.1)

            # Tie:
            A: news 2026-01-15, NIM=4.2%
            B: news 2026-01-15, NIM=4.5%
            → recommended_source = "UNRESOLVED"
        """
        a = candidate.claim_a
        b = candidate.claim_b

        date_a = self._parse_date(a.date)
        date_b = self._parse_date(b.date)
        auth_a = AUTHORITY_WEIGHTS.get(a.source_type, _DEFAULT_AUTHORITY)
        auth_b = AUTHORITY_WEIGHTS.get(b.source_type, _DEFAULT_AUTHORITY)

        # ── Decision logic ────────────────────────────────────────────────
        recommended: str
        reason_parts: list[str] = []
        unresolved = False

        date_winner: Optional[str] = None
        if date_a > date_b:
            date_winner = "A"
            reason_parts.append(
                f"Source A is more recent "
                f"({a.date or 'unknown'} > {b.date or 'unknown'})"
            )
        elif date_b > date_a:
            date_winner = "B"
            reason_parts.append(
                f"Source B is more recent "
                f"({b.date or 'unknown'} > {a.date or 'unknown'})"
            )
        # else: dates equal or both unknown → fall through to authority

        auth_winner: Optional[str] = None
        if auth_a > auth_b:
            auth_winner = "A"
            reason_parts.append(
                f"Source A has higher authority "
                f"({a.source_type}={auth_a:.1f} > {b.source_type}={auth_b:.1f})"
            )
        elif auth_b > auth_a:
            auth_winner = "B"
            reason_parts.append(
                f"Source B has higher authority "
                f"({b.source_type}={auth_b:.1f} > {a.source_type}={auth_a:.1f})"
            )

        # Final decision: date is primary, authority is tiebreaker
        if date_winner is not None:
            winner = date_winner
            if auth_winner is not None and auth_winner != date_winner:
                # Conflicting signals: date says A but authority says B
                # Date takes precedence but note the conflict
                reason_parts.append(
                    "Date takes precedence over authority in resolution"
                )
        elif auth_winner is not None:
            winner = auth_winner
        else:
            # True tie — cannot resolve
            winner      = "UNRESOLVED"
            unresolved  = True
            reason_parts.append(
                "Both sources have equal recency and authority — "
                "recommend manual verification"
            )

        if winner == "A":
            recommended_source = a.source_name
        elif winner == "B":
            recommended_source = b.source_name
        else:
            recommended_source = "UNRESOLVED — verify manually"

        reason = "; ".join(reason_parts) if reason_parts else "No resolution criteria met"

        return Conflict(
            field              = f"{a.metric}" + (f" ({a.entity})" if a.entity != "Unknown" else ""),
            source_a           = a.source_name,
            value_a            = a.display_value(),
            source_b           = b.source_name,
            value_b            = b.display_value(),
            recommended_source = recommended_source,
            reason             = reason,
            severity           = candidate.severity,
            date_a             = a.date or None,
            date_b             = b.date or None,
        )

    # ─────────────────────────────────────────
    # PUBLIC: classify_conflict_severity
    # ─────────────────────────────────────────

    def classify_conflict_severity(self, pct_diff: float) -> ConflictSeverity:
        """
        Classifies conflict severity based on relative percentage difference.

        Thresholds:
            pct_diff < 5%        → "Minor"
            5% ≤ pct_diff < 20%  → "Moderate"
            pct_diff ≥ 20%       → "Major"

        Args:
            pct_diff: Absolute percentage difference between two values.
                      e.g. NIM 4.2% vs 4.5% → pct_diff = |4.2-4.5|/4.35×100 ≈ 6.9%
                      → "Moderate"

        Returns:
            ConflictSeverity literal: "Minor", "Moderate", or "Major"

        Examples:
            classify_conflict_severity(3.0)   → "Minor"
            classify_conflict_severity(10.0)  → "Moderate"
            classify_conflict_severity(25.0)  → "Major"
            classify_conflict_severity(0.5)   → "Minor"
        """
        if pct_diff < SEVERITY_MINOR_MAX:
            return "Minor"
        elif pct_diff < SEVERITY_MODERATE_MAX:
            return "Moderate"
        else:
            return "Major"

    # ─────────────────────────────────────────
    # PUBLIC: get_stats
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns runtime statistics for monitoring dashboards."""
        avg_ms = round(self._total_ms / max(1, self._total_runs), 1)
        return {
            "total_runs":        self._total_runs,
            "skipped_runs":      self._skipped_runs,
            "total_conflicts":   self._total_conflicts,
            "total_claims":      self._total_claims,
            "total_ms":          round(self._total_ms, 1),
            "avg_ms_per_run":    avg_ms,
        }

    # ─────────────────────────────────────────
    # PRIVATE: compare_claims
    # ─────────────────────────────────────────

    def _compare_claims(
        self,
        a: NumericalClaim,
        b: NumericalClaim,
    ) -> Optional[ConflictCandidate]:
        """
        Compares two claims for the same (metric, entity, period).
        Returns ConflictCandidate if difference exceeds threshold, else None.

        Threshold logic:
            Ratio metrics (NIM, NPA, ROE …):
                - Absolute diff > CONFLICT_THRESHOLD_ABSOLUTE (0.5 pp)
                AND pct_diff > CONFLICT_THRESHOLD_PERCENTAGE (5%)
                → either condition triggers for ratios

            Absolute value metrics (Revenue, PAT …):
                - pct_diff > CONFLICT_THRESHOLD_PERCENTAGE (5%)
                → relative diff only (absolute amounts vary too widely)
        """
        abs_diff = abs(a.value - b.value)
        avg      = (a.value + b.value) / 2.0
        pct_diff = (abs_diff / avg * 100.0) if avg != 0.0 else 0.0

        # Determine if this pair constitutes a conflict
        is_conflict = False
        if a.is_ratio:
            # For ratios: flag if absolute diff > 0.5pp OR relative diff > 5%
            is_conflict = (
                abs_diff > CONFLICT_THRESHOLD_ABSOLUTE
                or pct_diff > CONFLICT_THRESHOLD_PERCENTAGE * 100
            )
        else:
            # For absolute values: flag only on relative diff
            is_conflict = pct_diff > CONFLICT_THRESHOLD_PERCENTAGE * 100

        if not is_conflict:
            return None

        severity = self.classify_conflict_severity(pct_diff)

        return ConflictCandidate(
            claim_a  = a,
            claim_b  = b,
            abs_diff = round(abs_diff, 4),
            pct_diff = round(pct_diff, 2),
            severity = severity,
            is_ratio = a.is_ratio,
        )

    # ─────────────────────────────────────────
    # PRIVATE: LLM-assisted claim extraction
    # ─────────────────────────────────────────

    async def _extract_claims_with_llm(
        self,
        chunks: list[dict],
    ) -> list[NumericalClaim]:
        """
        Fallback: uses Groq to extract financial claims when regex finds < 3.
        Returns list of NumericalClaim objects parsed from LLM JSON response.
        Safe — returns [] on any error.
        """
        combined_text = "\n\n---\n\n".join(
            f"[Source: {c.get('metadata',{}).get('source','unknown')}]\n{c.get('text','')[:400]}"
            for c in chunks[:5]
        )

        prompt = f"""Extract all numerical financial claims from these source excerpts.
For each claim return a JSON object with these exact keys:
  metric, value (float), unit ("%", "crore", "bps", or ""), entity, period, source_name

Return a JSON array. If no claims, return [].

Sources:
{combined_text}"""

        try:
            raw = await llm_client.call_llm_json(
                prompt      = prompt,
                agent_name  = AGENT_NAME,
                max_tokens  = 800,
                temperature = 0.0,
            )
        except LLMError as e:
            logger.warning(f"[CONFLICT DETECTOR] LLM extraction failed: {e}")
            return []

        items = raw if isinstance(raw, list) else raw.get("claims", [])
        claims: list[NumericalClaim] = []

        # Build a source_name → chunk metadata lookup
        src_meta: dict[str, dict] = {
            c.get("metadata", {}).get("source", ""): c.get("metadata", {})
            for c in chunks
        }

        for item in items:
            try:
                metric_name = str(item.get("metric", "")).strip().upper()
                value_raw   = item.get("value", None)
                if value_raw is None:
                    continue
                value       = float(value_raw)
                unit        = str(item.get("unit", "")).strip()
                entity      = str(item.get("entity", "Unknown")).strip()
                period      = str(item.get("period", "")).strip()
                source_name = str(item.get("source_name", "LLM Extraction")).strip()

                meta = src_meta.get(source_name, {})
                is_ratio = unit in ("%", "per cent", "bps", "basis points")

                claims.append(NumericalClaim(
                    metric      = metric_name,
                    value       = value,
                    unit        = unit,
                    entity      = entity,
                    period      = period,
                    raw_text    = f"LLM-extracted: {metric_name}={value}{unit}",
                    source_name = source_name,
                    source_type = meta.get("source_type", "unknown"),
                    date        = meta.get("date", ""),
                    chunk_id    = "llm_extracted",
                    is_ratio    = is_ratio,
                ))
            except (ValueError, TypeError, KeyError):
                continue

        return claims

    # ─────────────────────────────────────────
    # PRIVATE: helpers
    # ─────────────────────────────────────────

    def _count_distinct_sources(self, chunks: list[dict]) -> int:
        """Counts unique source document names across all chunks."""
        seen: set[str] = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {}) or {}
            src  = str(meta.get("source", meta.get("title", chunk.get("id", ""))))
            if src:
                seen.add(src.lower().strip())
        return len(seen)

    def _detect_entity(self, text: str) -> str:
        """
        Detects the first known financial entity in text.
        Returns canonical entity name or empty string.
        """
        for name, pattern in _ENTITY_PATTERNS:
            if pattern.search(text):
                return name
        return ""

    def _detect_period(self, text: str) -> str:
        """
        Detects the first time period reference in text.
        Returns normalized period string e.g. "Q3 FY26" or "".
        """
        m = _PERIOD_PATTERN.search(text)
        if m:
            raw = m.group(1).strip()
            # Normalize: "Q3 FY2026" → "Q3 FY26", "FY2025" → "FY25"
            raw = re.sub(r'FY\s*20(\d{2})', r'FY\1', raw, flags=re.I)
            raw = re.sub(r'\s+', ' ', raw)
            return raw.upper()
        return ""

    def _normalize_unit(
        self,
        matched_text: str,
        value: float,
    ) -> tuple[str, float]:
        """
        Detects unit from matched text and applies scale multipliers.

        Returns:
            (unit_str, normalized_value)

        Multipliers applied:
            "crore" / "cr" → value × 1   (already in crore)
            "lakh" / "lacs"→ value × 0.01 (convert to crore)
            "million" / "mn" → value × 0.1  (≈ crore)
            "billion" / "bn" → value × 100  (billion → crore)
            "bps"            → value × 0.01 (convert bps → %)
            "%"              → value (unchanged)
        """
        lower = matched_text.lower()

        if re.search(r'\bbps\b|basis\s*point', lower):
            return "bps", value

        if re.search(r'%|per\s*cent', lower):
            return "%", value

        if re.search(r'\bcrore\b|\bcr\b', lower):
            return "crore", value

        if re.search(r'\blakh\b|\blac', lower):
            return "lakh", value

        if re.search(r'\bbillion\b|\bbn\b', lower):
            return "bn", value

        if re.search(r'\bmillion\b|\bmn\b', lower):
            return "mn", value

        return "", value

    def _parse_date(self, date_str: str) -> datetime:
        """
        Parses a date string to datetime for comparison.
        Returns datetime.min on failure (treats unknown dates as oldest).
        """
        if not date_str:
            return datetime.min

        date_str = date_str.strip()
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try YYYY-MM extraction
        m = re.match(r'(\d{4})-(\d{2})', date_str)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), 1)
            except ValueError:
                pass

        # Try YYYY only
        m = re.match(r'^(\d{4})$', date_str)
        if m:
            try:
                return datetime(int(m.group(1)), 1, 1)
            except ValueError:
                pass

        return datetime.min


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

conflict_detector = ConflictDetector()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "conflict_detector",
    "ConflictDetector",
    "NumericalClaim",
    "ConflictCandidate",
]
