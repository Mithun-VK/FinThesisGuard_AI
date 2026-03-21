# backend/agents/agent5_quant_validator.py
"""
FinThesisGuard AI — Agent 5: Quantitative Validator
ArkAngel Financial Solutions

Cross-checks numerical financial data extracted from retrieved chunks for
mathematical consistency, unit coherence, and statistical plausibility.
Runs in parallel with Agent 4 (Conflict Detector) before LLM synthesis.

What this agent validates:
    1. Ratio Consistency  — fundamental accounting identity checks
           NIM  = NII / Avg Interest-Earning Assets     (tolerance ±15%)
           ROE  = PAT / Avg Equity × 100                (tolerance ±15%)
           ROA  = PAT / Avg Assets × 100                (tolerance ±15%)
           GNPA ≥ NNPA  always (mathematical identity)
           CRAR ≥ 0%    always (capital cannot be negative)
           PCR  ∈ [0%, 100%]  (coverage ratio bounded)
           CASA ∈ [0%, 100%]  (deposit ratio bounded)
           NIM  ∈ [0%, 15%]   (implausible beyond this)
           PE   ∈ [0, 1000]   (flag extreme values)

    2. Unit Mismatch Detection
           ₹ crore vs ₹ lakh vs ₹ million confusion
           % vs absolute value confusion (e.g., "NIM: 420" should be 4.2%)
           bps vs percentage confusion (e.g., "NIM increased 30bps" vs "30%")

    3. Statistical Outlier Detection
           Z-score > 3.0 (configurable) → flagged as outlier
           Requires ≥ 4 data points for meaningful Z-score
           Reports which values are outliers with z-score and context

Pipeline position:
    Agent 3 (Re-Ranker) → Agent 4 (Conflict) ─┬─ Agent 5 (Quant)
                                               └─ LLM Synthesis ← both outputs

Latency target: 300ms (pure computation, no LLM calls in default path)
Depends on: config.py, utils/llm_client.py, utils/logger.py,
            models/response_models.py

Output format:
    List[dict] — each dict is a ValidationWarning:
    {
        "code":       str   — machine-readable code e.g. "RATIO_INCONSISTENCY"
        "severity":   str   — "critical" | "warning" | "info"
        "metric":     str   — affected metric name
        "message":    str   — human-readable explanation
        "source":     str   — source document name
        "value":      float | None — offending value
        "expected":   str   — what the validator expected
        "chunk_id":   str   — originating chunk ID
    }
"""

import re
import math
import time
import asyncio
from dataclasses import dataclass, asdict, field
from typing import Optional

from pytest import warns

from backend.config import (
    AUTHORITY_WEIGHTS,
    AGENT_LATENCY_TARGETS,
)
from backend.utils.llm_client import llm_client, LLMError
from backend.utils.logger     import logger, log_metric


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

AGENT_NAME = "agent5_quant_validator"
LATENCY_TARGET_MS: int = AGENT_LATENCY_TARGETS.get(AGENT_NAME, 300)

# Z-score threshold for outlier detection
Z_SCORE_THRESHOLD: float = 3.0

# Minimum data points required for a meaningful Z-score
MIN_POINTS_FOR_ZSCORE: int = 4

# Tolerance for ratio consistency checks (15% relative deviation allowed)
# Accounts for: different averaging conventions, rounding in reports, etc.
RATIO_TOLERANCE: float = 0.15

# Unit confusion detection: if a claimed "percentage" value is suspiciously
# large (e.g., NIM=420 instead of 4.2%), flag it.
PERCENTAGE_UPPER_SANITY: float = 100.0   # Values > 100 are never valid %
NIM_MAX_SANE:     float = 15.0           # NIM > 15% is almost impossible
NIM_MIN_SANE:     float = 0.0
NPA_MAX_SANE:     float = 50.0           # GNPA > 50% → almost certainly error
PCR_MAX_SANE:     float = 100.0
PCR_MIN_SANE:     float = 0.0
CASA_MAX_SANE:    float = 100.0
CASA_MIN_SANE:    float = 0.0
CRAR_MIN_SANE:    float = 0.0
CRAR_MAX_SANE:    float = 60.0           # Above 60% → flag
ROE_MAX_SANE:     float = 100.0          # ROE > 100% extremely rare
ROE_MIN_SANE:     float = -100.0
PE_MAX_SANE:      float = 1000.0
PE_MIN_SANE:      float = 0.0

# Scale ratios — multiplier to convert to crore:
#   1 lakh     = 0.01 crore
#   1 million  = 0.1  crore
#   1 billion  = 100  crore
# Used in unit mismatch detection to detect cross-scale comparisons.
_UNIT_TO_CRORE: dict[str, float] = {
    "crore": 1.0,
    "cr":    1.0,
    "lakh":  0.01,
    "lacs":  0.01,
    "mn":    0.1,
    "million": 0.1,
    "bn":    100.0,
    "billion": 100.0,
}

# Minimum scale ratio to flag as unit mismatch (100× difference)
UNIT_MISMATCH_RATIO_THRESHOLD: float = 100.0


# ─────────────────────────────────────────────
# WARNING CODES
# ─────────────────────────────────────────────

class WarnCode:
    RATIO_INCONSISTENCY    = "RATIO_INCONSISTENCY"
    GNPA_BELOW_NNPA        = "GNPA_BELOW_NNPA"
    IMPOSSIBLE_VALUE       = "IMPOSSIBLE_VALUE"
    OUT_OF_RANGE           = "OUT_OF_RANGE"
    UNIT_MISMATCH          = "UNIT_MISMATCH"
    PERCENTAGE_AS_ABSOLUTE = "PERCENTAGE_AS_ABSOLUTE"
    BPS_VS_PERCENT         = "BPS_VS_PERCENT"
    OUTLIER_ZSCORE         = "OUTLIER_ZSCORE"
    NEGATIVE_METRIC        = "NEGATIVE_METRIC"
    TABLE_EXTRACTION_ERROR = "TABLE_EXTRACTION_ERROR"


class Severity:
    CRITICAL = "critical"   # Mathematically impossible — discard the value
    WARNING  = "warning"    # Likely error — flag for review
    INFO     = "info"       # Suspicious but may be legitimate


# ─────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────

@dataclass
class ValidationWarning:
    """
    A single validation issue raised by Agent 5.

    Fields:
        code:       Machine-readable warning code from WarnCode
        severity:   "critical" | "warning" | "info"
        metric:     Financial metric affected (e.g. "NIM", "GNPA")
        message:    Human-readable explanation of the issue
        source:     Source document name where the value was found
        value:      The offending numeric value (None if not applicable)
        expected:   String describing what the validator expected
        chunk_id:   ID of the chunk that contained this value
        entity:     Company / entity the value refers to
        period:     Reporting period (e.g. "Q3 FY26")
    """
    code:     str
    severity: str
    metric:   str
    message:  str
    source:   str   = ""
    value:    Optional[float] = None
    expected: str   = ""
    chunk_id: str   = ""
    entity:   str   = ""
    period:   str   = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        val_str = f" (value={self.value})" if self.value is not None else ""
        return f"[{self.severity.upper()}][{self.code}] {self.metric}{val_str}: {self.message}"


@dataclass
class FinancialRecord:
    """
    A structured financial data record extracted from one chunk.
    Aggregates all numeric metrics from a single source document
    into a flat dict for ratio-consistency checks.

    Fields:
        metrics:     {metric_name: float}  e.g. {"NIM": 4.2, "GNPA": 1.26}
        units:       {metric_name: str}    e.g. {"NIM": "%", "Revenue": "crore"}
        source_name: Document title
        source_type: Source type for authority lookup
        chunk_id:    Originating chunk ID
        entity:      Company / entity
        period:      Reporting period
    """
    metrics:     dict[str, float]  = field(default_factory=dict)
    units:       dict[str, str]    = field(default_factory=dict)
    source_name: str               = ""
    source_type: str               = "unknown"
    chunk_id:    str               = ""
    entity:      str               = ""
    period:      str               = ""


# ─────────────────────────────────────────────
# REGEX PATTERNS (compiled at import)
# ─────────────────────────────────────────────

# Numeric value extractor — handles: 4.2%, ₹45,000 crore, 1.26 per cent
_VALUE_RE = re.compile(
    r'(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)'    # number (with optional commas)
    r'\s*'
    r'(%|per\s*cent|bps|basis\s*points?'    # unit
    r'|crore|cr\.?|lakh|lacs?'
    r'|mn|million|bn|billion)?',
    re.IGNORECASE
)

# Metric name patterns: maps canonical name → list of regex fragments
_METRIC_ALIASES: dict[str, list[str]] = {
    "NIM":      [r"\bnim\b", r"net\s*interest\s*margin"],
    "NII":      [r"\bnii\b", r"net\s*interest\s*income"],
    "GNPA":     [r"\bgnpa\b", r"gross\s*npa", r"gross\s*non.?performing"],
    "NNPA":     [r"\bnnpa\b", r"net\s*npa",   r"net\s*non.?performing"],
    "NPA":      [r"\bnpa\s*ratio\b"],
    "CASA":     [r"\bcasa\b", r"casa\s*ratio"],
    "CRAR":     [r"\bcrar\b", r"capital\s*adequacy", r"\bcar\b"],
    "PCR":      [r"\bpcr\b", r"provision\s*coverage"],
    "ROE":      [r"\broe\b", r"return\s*on\s*equity"],
    "ROA":      [r"\broa\b", r"return\s*on\s*assets?"],
    "ROCE":     [r"\broce\b", r"return\s*on\s*capital\s*employed"],
    "PAT":      [r"\bpat\b", r"profit\s*after\s*tax"],
    "Revenue":  [r"\brevenue\b", r"total\s*income", r"net\s*revenue"],
    "Equity":   [r"\bequity\b", r"shareholders?\s*equity", r"net\s*worth"],
    "Assets":   [r"\btotal\s*assets?\b"],
    "PE":       [r"\bp/?e\s*ratio\b", r"price.to.earnings"],
    "EPS":      [r"\beps\b", r"earnings\s*per\s*share"],
    "Advances": [r"\badvances\b", r"loan\s*book", r"gross\s*advances"],
    "Deposits": [r"\bdeposits\b"],
    "AUM":      [r"\baum\b", r"assets\s*under\s*management"],
    "EBITDA":   [r"\bebitda\b"],
    "Dividend": [r"\bdividend\b"],
}

_COMPILED_METRICS: dict[str, re.Pattern] = {
    name: re.compile(r'(?:' + '|'.join(aliases) + r')', re.IGNORECASE)
    for name, aliases in _METRIC_ALIASES.items()
}

# Period detection
_PERIOD_RE = re.compile(
    r'\b(Q[1-4]\s*FY\s*\d{2,4}|FY\s*\d{2,4}|H[12]\s*FY\s*\d{2,4}|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\'?\d{2,4})\b',
    re.IGNORECASE,
)

# Entity detection (top 10 Indian financial names)
_ENTITY_RE = re.compile(
    r'\b(HDFC\s*Bank|ICICI\s*Bank|SBI|Axis\s*Bank|Kotak|'
    r'IndusInd|Yes\s*Bank|Bajaj\s*Finance|TCS|Infosys|Reliance|RBI|SEBI)\b',
    re.IGNORECASE,
)

# BPS context detector — "increased 30 bps" near a metric
_BPS_CONTEXT_RE = re.compile(
    r'(?:increased?|decreased?|changed?|up|down|rose|fell)\s*'
    r'(?:by\s*)?(\d+(?:\.\d+)?)\s*(?:bps|basis\s*points?)',
    re.IGNORECASE,
)

# Absolute-value-used-as-percent detector
# If NIM/NPA/CASA etc. appear with values > 20, likely % confusion
_HIGH_VALUE_RE = re.compile(r'\b(\d{2,}(?:\.\d+)?)\s*%?', re.IGNORECASE)


# ─────────────────────────────────────────────
# QUANT VALIDATOR CLASS
# ─────────────────────────────────────────────

class QuantValidator:
    """
    Agent 5: Cross-checks financial numbers for mathematical consistency,
    unit correctness, and statistical plausibility.

    Usage:
        from backend.agents.agent5_quant_validator import quant_validator

        warnings = await quant_validator.validate_quantities(reranked_chunks)
        for w in warnings:
            print(w["severity"], w["code"], w["message"])

    The agent is non-blocking — all validation is synchronous regex/math
    wrapped in an async interface for pipeline compatibility.
    """

    def __init__(self):
        self._total_runs:     int   = 0
        self._total_warnings: int   = 0
        self._total_chunks:   int   = 0
        self._total_ms:       float = 0.0

        logger.info(
            f"[QUANT VALIDATOR] Agent 5 ready | "
            f"z_threshold={Z_SCORE_THRESHOLD} | "
            f"ratio_tolerance={RATIO_TOLERANCE*100:.0f}% | "
            f"latency_target={LATENCY_TARGET_MS}ms"
        )

    # ─────────────────────────────────────────
    # PUBLIC: validate_quantities
    # ─────────────────────────────────────────

    async def validate_quantities(
        self,
        chunks: list[dict],
    ) -> list[dict]:
        """
        Main entry point. Validates all financial quantities across chunks.

        Pipeline:
            1. Extract FinancialRecords from all chunks
            2. validate_ratio_consistency() per record
            3. detect_unit_mismatches() across records
            4. flag_outliers() per metric across all records
            5. Collect, deduplicate, sort by severity, return as list[dict]

        Args:
            chunks: list[dict] from Agent 3 (re-ranked chunks).
                    Each must have: id, text, metadata
                    metadata: source_type, date, source, company

        Returns:
            list[dict] — each dict is a ValidationWarning.to_dict().
            Sorted: critical first, then warning, then info.
            Returns [] if no issues found or on empty input.

        Raises:
            Nothing — all errors handled gracefully, returns [] on failure.

        Latency:
            Regex-only path: ~10-40ms for 10 chunks.
            Target: 300ms (AGENT_LATENCY_TARGETS["agent5_quant_validator"]).

        Example return:
            [
                {
                    "code":     "GNPA_BELOW_NNPA",
                    "severity": "critical",
                    "metric":   "GNPA",
                    "message":  "GNPA (1.1%) < NNPA (1.5%) — mathematically impossible",
                    "source":   "HDFC Q3 FY26 Annual Report",
                    "value":    1.1,
                    "expected": "GNPA ≥ NNPA (1.5%)",
                    "chunk_id": "c1",
                    "entity":   "HDFC Bank",
                    "period":   "Q3 FY26",
                },
                ...
            ]
        """
        if not chunks:
            return []

        start_ms = time.perf_counter()
        self._total_runs  += 1
        self._total_chunks += len(chunks)

        warnings: list[ValidationWarning] = []

        try:
            # ── Step 1: Extract structured data from all chunks ───────────
            records: list[FinancialRecord] = []
            for chunk in chunks:
                recs = self.extract_financial_tables(chunk)
                records.extend(recs)

            logger.debug(
                f"[QUANT VALIDATOR] Extracted {len(records)} financial records "
                f"from {len(chunks)} chunks"
            )

            if not records:
                logger.debug("[QUANT VALIDATOR] No financial records found — skipping")
                return []

            # ── Step 2: Ratio consistency checks per record ───────────────
            for record in records:
                ratio_warns = self.validate_ratio_consistency(record)
                warnings.extend(ratio_warns)

            # ── Step 3: Unit mismatch detection across all records ────────
            unit_warns = self.detect_unit_mismatches(records)
            warnings.extend(unit_warns)

            # ── Step 4: Outlier detection per metric ──────────────────────
            metric_values: dict[str, list[tuple[float, FinancialRecord]]] = {}
            for record in records:
                for metric, value in record.metrics.items():
                    metric_values.setdefault(metric, []).append((value, record))

            for metric, value_record_pairs in metric_values.items():
                values  = [v for v, _ in value_record_pairs]
                outlier_warns = self.flag_outliers(
                    metric          = metric,
                    values          = values,
                    value_records   = value_record_pairs,
                )
                warnings.extend(outlier_warns)

            # ── Step 5: Sort by severity and deduplicate ──────────────────
            severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
            warnings.sort(key=lambda w: severity_order.get(w.severity, 3))
            warnings = _deduplicate_warnings(warnings)

            duration_ms        = (time.perf_counter() - start_ms) * 1000
            self._total_ms    += duration_ms
            self._total_warnings += len(warnings)

            logger.info(
                f"[QUANT VALIDATOR] Complete | "
                f"records={len(records)} | warnings={len(warnings)} | "
                f"{duration_ms:.1f}ms"
            )
            for w in warnings[:5]:      # log first 5 for debug
                logger.debug(f"[QUANT VALIDATOR]   {w}")

            log_metric("quant_validator_duration_ms", duration_ms,    unit="ms")
            log_metric("quant_validator_warnings",     len(warnings),  unit="count")
            log_metric("quant_validator_records",      len(records),   unit="count")

            # Warn if latency target breached
            if duration_ms > LATENCY_TARGET_MS:
                logger.warning(
                    f"[QUANT VALIDATOR] Latency target breached: "
                    f"{duration_ms:.0f}ms > {LATENCY_TARGET_MS}ms"
                )

            return [w.to_dict() for w in warnings]

        except Exception as e:
            duration_ms = (time.perf_counter() - start_ms) * 1000
            logger.error(
                f"[QUANT VALIDATOR] Unexpected error: {type(e).__name__}: {e} | "
                f"{duration_ms:.1f}ms"
            )
            return []

    # ─────────────────────────────────────────
    # PUBLIC: extract_financial_tables
    # ─────────────────────────────────────────

    def extract_financial_tables(self, chunk: dict) -> list[FinancialRecord]:
        """
        Extracts structured financial data from a single chunk.

        Strategy:
            1. Scan chunk text for each known metric pattern.
            2. Within a ±180-char window around each metric name match,
               extract the first numeric value with _VALUE_RE.
            3. Detect entity and period from the same window.
            4. Aggregate into FinancialRecord — one record per
               (entity, period, source) combination.
               Multiple metrics from the same source+entity+period
               are merged into one record.

        Args:
            chunk: dict with 'id', 'text', 'metadata'
                   metadata: source, source_type, company, date

        Returns:
            list[FinancialRecord] — usually 1 per chunk, occasionally
            multiple if the chunk discusses multiple entities/periods.

        Examples:
            chunk text: "HDFC NIM at 4.2% Q3 FY26; GNPA 1.26%; CASA 42%"
            →  [FinancialRecord(
                    metrics={"NIM": 4.2, "GNPA": 1.26, "CASA": 42.0},
                    units={"NIM": "%", "GNPA": "%", "CASA": "%"},
                    entity="HDFC Bank", period="Q3 FY26",
                    source_name="HDFC Annual Report"
                )]
        """
        text     = chunk.get("text", "")
        metadata = chunk.get("metadata", {}) or {}

        source_name = str(metadata.get("source", metadata.get("title", "Unknown")))
        source_type = str(metadata.get("source_type", "unknown")).lower().strip()
        chunk_id    = str(chunk.get("id", ""))

        base_entity = str(metadata.get("company", ""))
        if not base_entity:
            em = _ENTITY_RE.search(text)
            base_entity = em.group(0).strip() if em else "Unknown"

        base_period = ""
        pm = _PERIOD_RE.search(text)
        if pm:
            base_period = _normalise_period(pm.group(1))

        # records keyed by (entity, period) — merge metrics into same record
        records_map: dict[tuple[str, str], FinancialRecord] = {}

        for metric_name, pattern in _COMPILED_METRICS.items():
            for match in pattern.finditer(text):
                mstart, mend = match.start(), match.end()
                window_start = max(0, mstart - 150)
                window_end   = min(len(text), mend + 180)
                window       = text[window_start:window_end]

                search_from = mend - window_start          # always ≥ 0
                val_match   = _VALUE_RE.search(window, search_from)                
                if val_match is None:
                    # Try entire window
                    val_match = _VALUE_RE.search(window)
                if val_match is None:
                    continue

                raw_num = val_match.group(1).replace(",", "")
                try:
                    value = float(raw_num)
                except ValueError:
                    continue

                # Apply unit normalization
                raw_unit = (val_match.group(2) or "").lower().strip()
                unit, value = _normalise_unit(raw_unit, value)

                # Entity and period resolution
                em = _ENTITY_RE.search(window)
                entity = em.group(0).strip() if em else base_entity or "Unknown"

                pm = _PERIOD_RE.search(window)
                period = _normalise_period(pm.group(1)) if pm else base_period

                key = (entity.lower(), period.lower())
                if key not in records_map:
                    records_map[key] = FinancialRecord(
                        source_name = source_name,
                        source_type = source_type,
                        chunk_id    = chunk_id,
                        entity      = entity,
                        period      = period,
                    )

                rec = records_map[key]
                # Keep the first extracted value for each metric in this record
                if metric_name not in rec.metrics:
                    rec.metrics[metric_name] = value
                    rec.units[metric_name]   = unit

        return list(records_map.values())

    # ─────────────────────────────────────────
    # PUBLIC: validate_ratio_consistency
    # ─────────────────────────────────────────

    def validate_ratio_consistency(
        self,
        record: FinancialRecord,
    ) -> list[ValidationWarning]:
        """
        Checks fundamental accounting identity constraints on a FinancialRecord.

        Checks performed:
            A. GNPA ≥ NNPA         (hard mathematical identity)
            B. CRAR ≥ 0%           (capital adequacy cannot be negative)
            C. PCR  ∈ [0%, 100%]   (provision coverage ratio bounded)
            D. CASA ∈ [0%, 100%]   (deposit composition bounded)
            E. NIM  ∈ [0%, 15%]    (implausible outside this range)
            F. GNPA ∈ [0%, 50%]    (above 50% → almost certainly unit error)
            G. ROE  ∈ [-100%, 100%] (flag extreme returns)
            H. PE   ∈ [0, 1000]    (flag negative or extreme PE)
            I. NIM ≈ NII / Assets  if both NII and Assets available (±15%)
            J. ROE ≈ PAT / Equity  if both PAT and Equity available  (±15%)
            K. ROA ≈ PAT / Assets  if both PAT and Assets available  (±15%)

        Args:
            record: FinancialRecord with metrics dict

        Returns:
            list[ValidationWarning] — empty if no issues

        Examples:
            record.metrics = {"GNPA": 1.1, "NNPA": 1.5}
            → [ValidationWarning(code="GNPA_BELOW_NNPA", severity="critical", ...)]

            record.metrics = {"NIM": 4.2, "NII": 10000, "Assets": 25000}
            computed_NIM = 10000/25000*100 = 40% ≠ 4.2% → ratio_inconsistency
        """
        m       = record.metrics
        warns:  list[ValidationWarning] = []
        src     = record.source_name
        cid     = record.chunk_id
        entity  = record.entity
        period  = record.period

        def _warn(code, severity, metric, message, value=None, expected=""):
            warns.append(ValidationWarning(
                code=code, severity=severity, metric=metric,
                message=message, source=src, value=value,
                expected=expected, chunk_id=cid,
                entity=entity, period=period,
            ))

        # ── A: GNPA ≥ NNPA ───────────────────────────────────────────────
        gnpa = m.get("GNPA")
        nnpa = m.get("NNPA")
        if gnpa is not None and nnpa is not None:
            if gnpa < nnpa:
                _warn(
                    WarnCode.GNPA_BELOW_NNPA, Severity.CRITICAL, "GNPA",
                    f"GNPA ({gnpa:.2f}%) < NNPA ({nnpa:.2f}%) — "
                    f"mathematically impossible: gross NPA must always ≥ net NPA",
                    value=gnpa,
                    expected=f"GNPA ≥ NNPA ({nnpa:.2f}%)",
                )

        # ── B: CRAR ≥ 0 ──────────────────────────────────────────────────
        crar = m.get("CRAR")
        if crar is not None:
            if crar < CRAR_MIN_SANE:
                _warn(
                    WarnCode.IMPOSSIBLE_VALUE, Severity.CRITICAL, "CRAR",
                    f"CRAR ({crar:.2f}%) is negative — impossible for a solvent bank",
                    value=crar, expected="CRAR ≥ 0%",
                )
            elif crar > CRAR_MAX_SANE:
                _warn(
                    WarnCode.OUT_OF_RANGE, Severity.WARNING, "CRAR",
                    f"CRAR ({crar:.2f}%) exceeds {CRAR_MAX_SANE}% — "
                    f"possible unit confusion (bps reported as %?)",
                    value=crar, expected=f"CRAR ∈ [0%, {CRAR_MAX_SANE}%]",
                )

        # ── C: PCR ∈ [0%, 100%] ──────────────────────────────────────────
        pcr = m.get("PCR")
        if pcr is not None:
            if not (PCR_MIN_SANE <= pcr <= PCR_MAX_SANE):
                _warn(
                    WarnCode.OUT_OF_RANGE, Severity.CRITICAL, "PCR",
                    f"PCR ({pcr:.2f}%) is outside [0%, 100%] — "
                    f"provision coverage ratio must be bounded",
                    value=pcr, expected="PCR ∈ [0%, 100%]",
                )

        # ── D: CASA ∈ [0%, 100%] ─────────────────────────────────────────
        casa = m.get("CASA")
        if casa is not None:
            if not (CASA_MIN_SANE <= casa <= CASA_MAX_SANE):
                _warn(
                    WarnCode.OUT_OF_RANGE, Severity.CRITICAL, "CASA",
                    f"CASA ratio ({casa:.2f}%) is outside [0%, 100%] — impossible",
                    value=casa, expected="CASA ∈ [0%, 100%]",
                )

        # ── E: NIM ∈ [0%, 15%] ───────────────────────────────────────────
        nim = m.get("NIM")
        if nim is not None:
            if nim < NIM_MIN_SANE:
                _warn(
                    WarnCode.IMPOSSIBLE_VALUE, Severity.CRITICAL, "NIM",
                    f"NIM ({nim:.2f}%) is negative — net interest margin cannot be negative",
                    value=nim, expected="NIM ≥ 0%",
                )
            elif nim > NIM_MAX_SANE:
                _warn(
                    WarnCode.OUT_OF_RANGE, Severity.WARNING, "NIM",
                    f"NIM ({nim:.2f}%) exceeds {NIM_MAX_SANE}% — "
                    f"likely unit error (bps? or absolute NII reported as NIM?)",
                    value=nim, expected=f"NIM ∈ [0%, {NIM_MAX_SANE}%]",
                )

        # ── F: GNPA ∈ [0%, 50%] ──────────────────────────────────────────
        if gnpa is not None:
            if gnpa < 0:
                _warn(
                    WarnCode.IMPOSSIBLE_VALUE, Severity.CRITICAL, "GNPA",
                    f"GNPA ({gnpa:.2f}%) is negative — NPA ratio cannot be negative",
                    value=gnpa, expected="GNPA ≥ 0%",
                )
            elif gnpa > NPA_MAX_SANE:
                _warn(
                    WarnCode.OUT_OF_RANGE, Severity.WARNING, "GNPA",
                    f"GNPA ({gnpa:.2f}%) exceeds {NPA_MAX_SANE}% — "
                    f"possible absolute NPA amount (₹ crore) reported as ratio (%)",
                    value=gnpa, expected=f"GNPA ∈ [0%, {NPA_MAX_SANE}%]",
                )

        # ── G: ROE ∈ [-100%, 100%] ───────────────────────────────────────
        roe = m.get("ROE")
        if roe is not None and not (ROE_MIN_SANE <= roe <= ROE_MAX_SANE):
            _warn(
                WarnCode.OUT_OF_RANGE, Severity.WARNING, "ROE",
                f"ROE ({roe:.2f}%) is outside [{ROE_MIN_SANE}%, {ROE_MAX_SANE}%] — "
                f"verify units (PAT in crore vs equity in different scale?)",
                value=roe, expected=f"ROE ∈ [{ROE_MIN_SANE}%, {ROE_MAX_SANE}%]",
            )

        # ── H: PE ∈ [0, 1000] ────────────────────────────────────────────
        pe = m.get("PE")
        if pe is not None:
            if pe < PE_MIN_SANE:
                _warn(
                    WarnCode.IMPOSSIBLE_VALUE, Severity.WARNING, "PE",
                    f"PE Ratio ({pe:.1f}) is negative — check EPS sign",
                    value=pe, expected=f"PE ≥ 0",
                )
            elif pe > PE_MAX_SANE:
                _warn(
                    WarnCode.OUT_OF_RANGE, Severity.INFO, "PE",
                    f"PE Ratio ({pe:.1f}) exceeds {PE_MAX_SANE} — "
                    f"extremely high; possible data error",
                    value=pe, expected=f"PE ∈ [0, {PE_MAX_SANE}]",
                )

        # ── I: NIM ≈ NII / Assets × 100 ──────────────────────────────────
        nii    = m.get("NII")
        assets = m.get("Assets")
        if nim is not None and nii is not None and assets is not None:
            if assets > 0:
                computed_nim = (nii / assets) * 100.0
                if not _within_tolerance(nim, computed_nim, RATIO_TOLERANCE):
                    _warn(
                        WarnCode.RATIO_INCONSISTENCY, Severity.WARNING, "NIM",
                        f"NIM ({nim:.2f}%) inconsistent with NII/Assets: "
                        f"NII={nii:,.0f} / Assets={assets:,.0f} × 100 = {computed_nim:.2f}% "
                        f"(difference: {abs(nim - computed_nim):.2f}pp, "
                        f"tolerance: ±{RATIO_TOLERANCE*100:.0f}%)",
                        value=nim,
                        expected=f"≈ {computed_nim:.2f}% (NII/Assets×100)",
                    )

        # ── J: ROE ≈ PAT / Equity × 100 ──────────────────────────────────
        pat    = m.get("PAT")
        equity = m.get("Equity")
        if roe is not None and pat is not None and equity is not None:
            if equity > 0:
                computed_roe = (pat / equity) * 100.0
                if not _within_tolerance(roe, computed_roe, RATIO_TOLERANCE):
                    _warn(
                        WarnCode.RATIO_INCONSISTENCY, Severity.WARNING, "ROE",
                        f"ROE ({roe:.2f}%) inconsistent with PAT/Equity: "
                        f"PAT={pat:,.0f} / Equity={equity:,.0f} × 100 = {computed_roe:.2f}% "
                        f"(difference: {abs(roe - computed_roe):.2f}pp, "
                        f"tolerance: ±{RATIO_TOLERANCE*100:.0f}%)",
                        value=roe,
                        expected=f"≈ {computed_roe:.2f}% (PAT/Equity×100)",
                    )

        # ── K: ROA ≈ PAT / Assets × 100 ──────────────────────────────────
        roa = m.get("ROA")
        if roa is not None and pat is not None and assets is not None:
            if assets > 0:
                computed_roa = (pat / assets) * 100.0
                if not _within_tolerance(roa, computed_roa, RATIO_TOLERANCE):
                    _warn(
                        WarnCode.RATIO_INCONSISTENCY, Severity.WARNING, "ROA",
                        f"ROA ({roa:.2f}%) inconsistent with PAT/Assets: "
                        f"PAT={pat:,.0f} / Assets={assets:,.0f} × 100 = {computed_roa:.4f}% "
                        f"(difference: {abs(roa - computed_roa):.4f}pp, "
                        f"tolerance: ±{RATIO_TOLERANCE*100:.0f}%)",
                        value=roa,
                        expected=f"≈ {computed_roa:.4f}% (PAT/Assets×100)",
                    )

        return warns

    # ─────────────────────────────────────────
    # PUBLIC: detect_unit_mismatches
    # ─────────────────────────────────────────

    def detect_unit_mismatches(
        self,
        records: list[FinancialRecord],
    ) -> list[ValidationWarning]:
        """
        Detects unit inconsistencies across FinancialRecords:
            1. Scale confusion — same metric reported in crore vs lakh vs million
               across different chunks for the same entity + period.
            2. Percentage-as-absolute — metric expected as % but value is in
               hundreds or thousands (e.g., NIM=420 instead of 4.2%).
            3. BPS vs percent — metric reported as X% when context says Xbps,
               or vice versa.

        Args:
            records: list[FinancialRecord] from extract_financial_tables()

        Returns:
            list[ValidationWarning]

        Examples:
            # Record A: Revenue = 50000 crore
            # Record B: Revenue = 50000 lakh  (should be 500 crore)
            # Scale ratio = 100× → UNIT_MISMATCH warning

            # Record A: NIM = 420% (should be 4.20%)
            # → PERCENTAGE_AS_ABSOLUTE warning

            # Record A: NIM = 42 bps  (should be 0.42%)
            # → BPS_VS_PERCENT warning
        """
        warns: list[ValidationWarning] = []

        # ── Check 1: Scale confusion for absolute-value metrics ───────────
        # Group absolute-value metrics by (metric, entity, period)
        # and compare units across records.
        abs_metrics = {"Revenue", "PAT", "NII", "AUM", "Advances", "Deposits", "EBITDA"}

        groups: dict[tuple[str, str, str], list[FinancialRecord]] = {}
        for record in records:
            for metric in abs_metrics:
                if metric in record.metrics:
                    key = (metric, record.entity.lower(), record.period.lower())
                    groups.setdefault(key, []).append(record)

        for (metric, entity, period), recs in groups.items():
            if len(recs) < 2:
                continue
            # Check if any two records use different scale units
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    rec_a = recs[i]
                    rec_b = recs[j]
                    unit_a = rec_a.units.get(metric, "")
                    unit_b = rec_b.units.get(metric, "")
                    val_a  = rec_a.metrics[metric]
                    val_b  = rec_b.metrics[metric]

                    scale_a = _UNIT_TO_CRORE.get(unit_a, 1.0)
                    scale_b = _UNIT_TO_CRORE.get(unit_b, 1.0)

                    if scale_a == scale_b:
                        continue   # same unit family → no mismatch

                    # Normalise both to crore and check ratio
                    norm_a = val_a * scale_a
                    norm_b = val_b * scale_b
                    if norm_b == 0:
                        continue
                    ratio = norm_a / norm_b
                    if ratio >= UNIT_MISMATCH_RATIO_THRESHOLD or ratio <= 1 / UNIT_MISMATCH_RATIO_THRESHOLD:
                        warns.append(ValidationWarning(
                            code     = WarnCode.UNIT_MISMATCH,
                            severity = Severity.WARNING,
                            metric   = metric,
                            message  = (
                                f"{metric} unit mismatch: "
                                f"'{rec_a.source_name}' reports {val_a:,.0f} {unit_a} "
                                f"vs '{rec_b.source_name}' reports {val_b:,.0f} {unit_b} — "
                                f"normalised ratio = {ratio:.0f}× (expected ~1×). "
                                f"Likely {unit_a} vs {unit_b} confusion."
                            ),
                            source   = rec_a.source_name,
                            value    = val_a,
                            expected = f"Same unit as {rec_b.source_name}: {val_b:,.0f} {unit_b}",
                            chunk_id = rec_a.chunk_id,
                            entity   = rec_a.entity,
                            period   = rec_a.period,
                        ))

        # ── Check 2: Percentage-as-absolute (ratio metrics) ───────────────
        # Ratio metrics that should always be in [0%, 100%]
        ratio_metrics_bounds: dict[str, tuple[float, float]] = {
            "NIM":  (0.0, NIM_MAX_SANE),
            "GNPA": (0.0, NPA_MAX_SANE),
            "NNPA": (0.0, NPA_MAX_SANE),
            "NPA":  (0.0, NPA_MAX_SANE),
            "CASA": (0.0, 100.0),
            "CRAR": (0.0, CRAR_MAX_SANE),
            "PCR":  (0.0, 100.0),
            "ROE":  (-100.0, 100.0),
            "ROA":  (-10.0,  10.0),
        }

        for record in records:
            for metric, (lo, hi) in ratio_metrics_bounds.items():
                val = record.metrics.get(metric)
                if val is None:
                    continue
                unit = record.units.get(metric, "")

                # If unit is not %, check if value is unreasonably large
                if unit not in ("%", "per cent", "") and val > hi * 10:
                    warns.append(ValidationWarning(
                        code     = WarnCode.PERCENTAGE_AS_ABSOLUTE,
                        severity = Severity.WARNING,
                        metric   = metric,
                        message  = (
                            f"{metric} ({val} {unit}) appears to be an absolute amount "
                            f"where a percentage is expected. "
                            f"Expected range: [{lo}%, {hi}%]. "
                            f"Possible confusion: {val} {unit} should be {val/100:.2f}%?"
                        ),
                        source   = record.source_name,
                        value    = val,
                        expected = f"Percentage in [{lo}%, {hi}%]",
                        chunk_id = record.chunk_id,
                        entity   = record.entity,
                        period   = record.period,
                    ))

        # ── Check 3: BPS vs percent confusion ─────────────────────────────
        # If a ratio metric is reported in "bps" but the raw value > 100,
        # it may have been incorrectly left as bps (not divided by 100).
        for record in records:
            for metric in ("NIM", "GNPA", "NNPA", "CRAR", "CASA", "PCR"):
                val  = record.metrics.get(metric)
                unit = record.units.get(metric, "")
                if val is None:
                    continue
                if unit == "bps" and val > 100:
                    warns.append(ValidationWarning(
                        code     = WarnCode.BPS_VS_PERCENT,
                        severity = Severity.INFO,
                        metric   = metric,
                        message  = (
                            f"{metric} reported as {val:.0f} bps — "
                            f"if this is the ratio value, it should be "
                            f"{val/100:.2f}% (divide by 100). "
                            f"Verify whether bps is a change or a level."
                        ),
                        source   = record.source_name,
                        value    = val,
                        expected = f"If ratio: {val/100:.2f}%",
                        chunk_id = record.chunk_id,
                        entity   = record.entity,
                        period   = record.period,
                    ))

        return warns

    # ─────────────────────────────────────────
    # PUBLIC: flag_outliers
    # ─────────────────────────────────────────
    def flag_outliers(
        self,
        metric:        str,
        values:        list[float],
        value_records: Optional[list[tuple[float, FinancialRecord]]] = None,    
    ) -> list[dict]:
        """
    Detects statistical outliers using the Modified Z-score method.
    Requires MIN_POINTS_FOR_ZSCORE (4) data points minimum.

    WHY MODIFIED Z-SCORE (not classical Z-score):
        Classical z = (x − μ) / σ fails when outliers are present because
        they inflate both μ and σ, masking themselves.

        Example: values = [4.2, 4.1, 4.3, 4.2, 4.15, 4.25, 420.0]
            Classical: μ=63.0, σ=145.5 → z(420.0) = 2.45 < 3.0 → MISSED ✗
            Modified:  median=4.2, MAD=0.05 → M(420.0) = 5609 >> 3.5 → CAUGHT ✓

    Modified Z-score formula (Iglewicz & Hoaglin, 1993):
        median   = median(values)
        MAD      = median(|x_i − median|)
        M_i      = 0.6745 × |x_i − median| / MAD

        The consistency constant 0.6745 ensures that for a normal
        distribution, MAD ≈ 0.6745 × σ (so M_i ≈ classical z_i).

        Flag if M_i > MODIFIED_Z_THRESHOLD (3.5 — standard per literature)

    MAD = 0 fallback:
        When all values equal the median (MAD=0), use the mean absolute
        deviation (MAD_mean) as substitute. If that is also 0, no outliers
        are possible (all values identical).

    Args:
        metric:        Metric name (e.g. "NIM")
        values:        List of observed values across all records.
        value_records: Parallel list of (value, FinancialRecord) tuples for
                       attaching source context to warnings.
                       If None, warnings have empty source fields.

    Returns:
        list[dict] — each dict is a ValidationWarning.to_dict().
        Returns [] when:
            - len(values) < MIN_POINTS_FOR_ZSCORE (4)
            - all values are identical
            - no M_i exceeds MODIFIED_Z_THRESHOLD

    Examples:
        # Outlier present — caught ✓
        flag_outliers("NIM", [4.2, 4.1, 4.3, 4.2, 4.15, 4.25, 420.0])
        # median=4.2, MAD=0.05, M(420.0)=5609 >> 3.5 → flagged

        # Tight cluster — no outliers ✓
        flag_outliers("NIM", [4.1, 4.2, 4.3, 4.15])
        # All M_i < 3.5 → []

        # Too few points — skipped ✓
        flag_outliers("NIM", [4.2, 420.0])
        # n=2 < 4 → []
        """
        warns: list[ValidationWarning] = []

        if len(values) < MIN_POINTS_FOR_ZSCORE:
            return []

        n = len(values)

    # ── Compute median ─────────────────────────────────────────────────────
        sorted_vals = sorted(values)
        if n % 2 == 1:
            median = sorted_vals[n // 2]
        else:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    # ── Compute MAD (Median Absolute Deviation) ───────────────────────────
        abs_devs = sorted(abs(v - median) for v in values)
        if n % 2 == 1:
            mad = abs_devs[n // 2]
        else:
            mad = (abs_devs[n // 2 - 1] + abs_devs[n // 2]) / 2.0

    # ── MAD = 0 fallback: use mean absolute deviation ─────────────────────
        if mad == 0.0:
            mad_mean = sum(abs(v - median) for v in values) / n
            if mad_mean == 0.0:
                return []   # All values identical — no outliers possible
        # Use mean absolute deviation with adjusted constant
        # (0.7979 ≈ E[|Z|] for standard normal, gives consistent scaling)
            effective_mad = mad_mean / 0.7979
        else:
            effective_mad = mad

    # ── Modified Z-score threshold ────────────────────────────────────────
    # 3.5 is the standard recommendation (Iglewicz & Hoaglin, 1993).
    # Slightly higher than classical z=3.0 to compensate for the
    # consistency constant 0.6745 producing slightly larger scores.
        CONSISTENCY_CONSTANT   = 0.6745
        MODIFIED_Z_THRESHOLD   = 3.5

        for i, val in enumerate(values):
            modified_z = CONSISTENCY_CONSTANT * abs(val - median) / effective_mad

            if modified_z > MODIFIED_Z_THRESHOLD:
                rec = value_records[i][1] if value_records and i < len(value_records) else None
                warns.append(ValidationWarning(
                    code     = WarnCode.OUTLIER_ZSCORE,
                    severity = Severity.WARNING,
                    metric   = metric,
                    message  = (
                        f"{metric} value {val:.4g} is a statistical outlier "
                        f"(modified z-score={modified_z:.1f}, threshold={MODIFIED_Z_THRESHOLD}). "
                        f"Median across {n} sources = {median:.4g}, "
                        f"MAD = {mad:.4g}. "
                        f"Possible data entry error or unit confusion "
                        f"(e.g. bps reported as %, or absolute value as ratio)."
                    ),
                    source   = rec.source_name if rec else "",
                    value    = val,
                    expected = (
                        f"Within {MODIFIED_Z_THRESHOLD} modified-σ of median "
                        f"({median:.4g} ± {mad / CONSISTENCY_CONSTANT:.4g})"
                    ),
                    chunk_id = rec.chunk_id if rec else "",
                    entity   = rec.entity   if rec else "",
                    period   = rec.period   if rec else "",
                ))

    # Returns list[dict] so callers (tests + validate_quantities) get
    # consistent dict access: warns[0]["code"], warns[0]["severity"] etc.
        return [w.to_dict() for w in warns]

    # ─────────────────────────────────────────
    # PUBLIC: get_stats
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns runtime statistics for monitoring dashboards."""
        avg_ms = round(self._total_ms / max(1, self._total_runs), 1)
        return {
            "total_runs":           self._total_runs,
            "total_chunks_seen":    self._total_chunks,
            "total_warnings":       self._total_warnings,
            "total_ms":             round(self._total_ms, 1),
            "avg_ms_per_run":       avg_ms,
            "z_score_threshold":    Z_SCORE_THRESHOLD,
            "ratio_tolerance_pct":  RATIO_TOLERANCE * 100,
        }


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _within_tolerance(actual: float, expected: float, tolerance: float) -> bool:
    """
    Returns True if actual is within `tolerance` fraction of expected.
    Uses relative tolerance when expected != 0, absolute otherwise.

    Examples:
        _within_tolerance(4.2, 4.0, 0.15) → True   (5% diff < 15%)
        _within_tolerance(4.2, 8.0, 0.15) → False  (47.5% diff > 15%)
        _within_tolerance(0.0, 0.0, 0.15) → True   (both zero)
    """
    if expected == 0.0:
        return abs(actual) < 1e-9
    return abs(actual - expected) / abs(expected) <= tolerance


def _normalise_unit(raw_unit: str, value: float) -> tuple[str, float]:
    """
    Normalises unit string and adjusts value for scale.
    bps → converted but unit marked "bps" (caller decides interpretation).
    Returns (canonical_unit, adjusted_value).
    """
    u = raw_unit.lower().strip()
    if u in ("bps", "basis points", "basis point"):
        return "bps", value
    if u in ("%", "per cent", "per_cent", "percent"):
        return "%", value
    if u in ("crore", "cr"):
        return "crore", value
    if u in ("lakh", "lacs", "lac"):
        return "lakh", value
    if u in ("million", "mn"):
        return "mn", value
    if u in ("billion", "bn"):
        return "bn", value
    return u, value


def _normalise_period(raw: str) -> str:
    """Normalises period strings: 'Q3 FY2026' → 'Q3 FY26'."""
    s = re.sub(r'FY\s*20(\d{2})', r'FY\1', raw, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s)
    return s.upper().strip()


def _deduplicate_warnings(warns: list[ValidationWarning]) -> list[ValidationWarning]:
    """
    Removes duplicate warnings with identical (code, metric, chunk_id, value).
    Keeps first occurrence (already sorted by severity so highest stays).
    """
    seen: set[tuple] = set()
    result: list[ValidationWarning] = []
    for w in warns:
        key = (w.code, w.metric, w.chunk_id, w.value)
        if key not in seen:
            seen.add(key)
            result.append(w)
    return result


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

quant_validator = QuantValidator()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "quant_validator",
    "QuantValidator",
    "ValidationWarning",
    "FinancialRecord",
    "WarnCode",
    "Severity",
]
