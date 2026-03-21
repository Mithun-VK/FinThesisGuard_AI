# backend/models/query_models.py
"""
FinThesisGuard AI — Incoming Request Models
ArkAngel Financial Solutions

Pydantic v2 schemas for all API request validation.
Every FastAPI route uses these models for input parsing and validation.
Standalone — no internal dependencies.
"""

import re
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────
# CONSTANTS (mirrored from config for standalone use)
# ─────────────────────────────────────────────

QUERY_MIN_LENGTH: int = 5
QUERY_MAX_LENGTH: int = 2000
THESIS_MIN_LENGTH: int = 20
THESIS_MAX_LENGTH: int = 5000
CONTEXT_MAX_LENGTH: int = 1000
TOP_K_MIN: int = 1
TOP_K_MAX: int = 20
TOP_K_DEFAULT: int = 10

# Basic non-English detection: reject strings with majority non-ASCII
MIN_ASCII_RATIO: float = 0.6

# Common gibberish/spam patterns to reject
SPAM_PATTERNS: list[str] = [
    r"(.)\1{9,}",          # Same char repeated 10+ times: "aaaaaaaaaa"
    r"^[^a-zA-Z0-9]+$",    # Only special characters
    r"^\d+$",              # Only numbers
]


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class QueryType(str, Enum):
    """
    Classification of financial query types.
    Used by Agent 2 (Query Decomposer) for routing.
    """
    FACTUAL      = "FACTUAL"       # Single fact lookup: "What is HDFC NIM?"
    COMPARATIVE  = "COMPARATIVE"   # Multi-entity: "Compare HDFC vs ICICI NIM"
    REGULATORY   = "REGULATORY"    # SEBI/RBI: "What does SEBI say about F&O margins?"
    THESIS       = "THESIS"        # Investment thesis: "NVIDIA will outperform because..."
    ANALYTICAL   = "ANALYTICAL"    # Complex analysis: "Why did HDFC NIM compress in Q3?"
    UNKNOWN      = "UNKNOWN"       # Default before classification


class SourceType(str, Enum):
    """Valid document source types for ingestion."""
    RBI                 = "rbi"
    SEBI                = "sebi"
    ANNUAL_REPORT       = "annual_report"
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    BROKER_RESEARCH     = "broker_research"
    NEWS                = "news"
    BLOG                = "blog"
    TAX_LAW             = "tax_law"
    MCA                 = "mca"
    UNKNOWN             = "unknown"


class IngestStatus(str, Enum):
    """Status of a document ingestion job."""
    PENDING    = "pending"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"
    SKIPPED    = "skipped"


# ─────────────────────────────────────────────
# SHARED VALIDATORS
# ─────────────────────────────────────────────

def _normalize_text(value: str) -> str:
    """
    Strips leading/trailing whitespace and collapses internal whitespace.
    Applied to all text fields.
    """
    return re.sub(r'\s+', ' ', value.strip())


def _check_not_empty(value: str, field_name: str = "Field") -> str:
    """Rejects empty or whitespace-only strings after normalization."""
    if not value or not value.strip():
        raise ValueError(f"{field_name} cannot be empty or whitespace only.")
    return value


def _check_ascii_ratio(value: str, field_name: str = "Field") -> str:
    """
    Basic English language check via ASCII character ratio.
    Rejects strings where less than 60% of characters are ASCII.
    Financial terms are overwhelmingly ASCII-based.
    """
    if not value:
        return value
    ascii_count = sum(1 for c in value if ord(c) < 128)
    ratio = ascii_count / len(value)
    if ratio < MIN_ASCII_RATIO:
        raise ValueError(
            f"{field_name} appears to be in a non-English language. "
            f"FinThesisGuard currently supports English financial text only. "
            f"(ASCII ratio: {ratio:.0%}, minimum: {MIN_ASCII_RATIO:.0%})"
        )
    return value


def _check_spam_patterns(value: str, field_name: str = "Field") -> str:
    """
    Rejects obvious spam/gibberish patterns.
    Catches: 'aaaaaaaaaaaa', '!!!!!!!!!!!', '12345678'.
    """
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, value):
            raise ValueError(
                f"{field_name} appears to contain invalid or gibberish content."
            )
    return value


def _check_min_word_count(value: str, min_words: int, field_name: str) -> str:
    """Ensures text has a minimum number of words."""
    word_count = len(value.split())
    if word_count < min_words:
        raise ValueError(
            f"{field_name} must contain at least {min_words} word(s). "
            f"Got {word_count} word(s)."
        )
    return value


# ─────────────────────────────────────────────
# 1. QueryRequest
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Request schema for financial RAG queries.
    Used by POST /api/query endpoint.

    Examples:
        {"query": "What is HDFC Bank NIM for Q3 FY26?"}
        {"query": "Compare HDFC vs ICICI NIM and NPA", "top_k": 15}
        {"query": "SEBI circular on F&O margin requirements", "use_cache": false}
    """

    query: str = Field(
        ...,
        min_length=QUERY_MIN_LENGTH,
        max_length=QUERY_MAX_LENGTH,
        description="Financial question or query in English",
        examples=[
            "What is HDFC Bank NIM for Q3 FY26?",
            "Compare HDFC vs ICICI NPA ratios",
            "SEBI circular on F&O margin requirements 2025",
        ],
    )
    top_k: int = Field(
        default=TOP_K_DEFAULT,
        ge=TOP_K_MIN,
        le=TOP_K_MAX,
        description=f"Number of source documents to retrieve ({TOP_K_MIN}–{TOP_K_MAX})",
    )
    use_cache: bool = Field(
        default=True,
        description="Use Redis cache for this query (set False to force fresh retrieval)",
    )
    query_type: Optional[QueryType] = Field(
        default=None,
        description="Optional pre-classified query type (auto-detected if not provided)",
    )
    filters: Optional[dict] = Field(
        default=None,
        description="Optional metadata filters for retrieval e.g. {'source_type': 'rbi'}",
    )
    stream: bool = Field(
        default=False,
        description="If True, returns a streaming text response",
    )

    # ── Validators ───────────────────────────

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("query must be a string.")
        return _normalize_text(v)

    @field_validator("query")
    @classmethod
    def validate_query_content(cls, v: str) -> str:
        _check_not_empty(v, "Query")
        _check_ascii_ratio(v, "Query")
        _check_spam_patterns(v, "Query")
        _check_min_word_count(v, min_words=2, field_name="Query")
        return v

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, v: Optional[dict]) -> Optional[dict]:
        if v is None:
            return v
        allowed_filter_keys = {
            "source_type", "company", "sector",
            "date_from", "date_to", "authority",
        }
        invalid_keys = set(v.keys()) - allowed_filter_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid filter keys: {invalid_keys}. "
                f"Allowed: {allowed_filter_keys}"
            )
        return v

    # ── Computed Properties ───────────────────

    @property
    def word_count(self) -> int:
        return len(self.query.split())

    @property
    def is_complex(self) -> bool:
        """Heuristic: likely needs query decomposition."""
        comparison_words = {"vs", "versus", "compare", "comparison", "difference", "between"}
        query_words = set(self.query.lower().split())
        return bool(comparison_words & query_words) or self.word_count > 20

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is HDFC Bank Net Interest Margin for Q3 FY26?",
                    "top_k": 10,
                    "use_cache": True,
                },
                {
                    "query": "Compare HDFC vs ICICI NPA and NIM for FY25",
                    "top_k": 15,
                    "use_cache": False,
                    "filters": {"source_type": "annual_report"},
                },
            ]
        }
    }


# ─────────────────────────────────────────────
# 2. BatchQueryRequest
# ─────────────────────────────────────────────

class BatchQueryRequest(BaseModel):
    """
    Request schema for batch RAG queries.
    Used by POST /api/query/batch endpoint.
    Maximum 10 queries per batch.
    """

    queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of financial queries (max 10)",
    )
    top_k: int = Field(
        default=TOP_K_DEFAULT,
        ge=TOP_K_MIN,
        le=TOP_K_MAX,
    )
    use_cache: bool = Field(default=True)

    @field_validator("queries", mode="before")
    @classmethod
    def validate_queries(cls, v: list) -> list:
        if not v:
            raise ValueError("queries list cannot be empty.")
        cleaned = []
        for i, q in enumerate(v):
            if not isinstance(q, str):
                raise ValueError(f"Query at index {i} must be a string.")
            q = _normalize_text(q)
            if len(q) < QUERY_MIN_LENGTH:
                raise ValueError(
                    f"Query at index {i} is too short "
                    f"(min {QUERY_MIN_LENGTH} chars)."
                )
            if len(q) > QUERY_MAX_LENGTH:
                raise ValueError(
                    f"Query at index {i} is too long "
                    f"(max {QUERY_MAX_LENGTH} chars)."
                )
            cleaned.append(q)
        return cleaned


# ─────────────────────────────────────────────
# 3. ThesisRequest
# ─────────────────────────────────────────────

class ThesisRequest(BaseModel):
    """
    Request schema for investment thesis validation.
    Used by POST /api/validate-thesis endpoint.

    A valid thesis must:
    - Have a subject (company/asset)
    - Make a directional claim (will outperform, overvalued, etc.)
    - Provide at least one reason/assumption

    Examples:
        {"thesis": "HDFC Bank will outperform peers because NIM expansion..."}
        {"thesis": "NVIDIA is overvalued as AI capex cycle peaks in 2025..."}
    """

    thesis: str = Field(
        ...,
        min_length=THESIS_MIN_LENGTH,
        max_length=THESIS_MAX_LENGTH,
        description="Investment thesis statement in English (min 20 chars)",
        examples=[
            "HDFC Bank will outperform peers over the next 12 months because "
            "its NIM expansion cycle is still underway and CASA ratio remains "
            "above 40%, providing a structural cost of funds advantage.",

            "NVIDIA is at risk of significant multiple compression as AI capex "
            "growth decelerates in H2 2025 and hyperscaler ROI on GPU investments "
            "remains unproven at scale.",
        ],
    )
    context: Optional[str] = Field(
        default=None,
        max_length=CONTEXT_MAX_LENGTH,
        description="Optional additional context (sector, time horizon, portfolio position)",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Investment time horizon e.g. '12 months', '3 years'",
        examples=["6 months", "12 months", "3 years", "5 years"],
    )
    asset_class: Optional[Literal["equity", "debt", "commodity", "currency", "other"]] = Field(
        default="equity",
        description="Asset class the thesis refers to",
    )
    use_cache: bool = Field(
        default=True,
        description="Use cached results if same thesis was validated recently",
    )
    quick_mode: bool = Field(
        default=False,
        description="If True, runs faster analysis (skips Agent 5 quant validation)",
    )

    # ── Validators ───────────────────────────

    @field_validator("thesis", mode="before")
    @classmethod
    def normalize_thesis(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("thesis must be a string.")
        normalized = _normalize_text(v)
        # Check for pure questions BEFORE length validation fires
        # so the error message is meaningful
        if normalized.strip().endswith("?") and len(normalized.split()) < 15:
            raise ValueError(
                "This looks like a question, not an investment thesis. "
                "Use the RAG Query endpoint for questions. "
                "A thesis should make a claim, e.g., "
                "'HDFC will outperform because...'"
            )
        return normalized


    @field_validator("thesis")
    @classmethod
    def validate_thesis_content(cls, v: str) -> str:
        _check_not_empty(v, "Thesis")
        _check_ascii_ratio(v, "Thesis")
        _check_spam_patterns(v, "Thesis")
        _check_min_word_count(v, min_words=5, field_name="Thesis")
        _validate_thesis_structure(v)
        return v

    @field_validator("context", mode="before")
    @classmethod
    def normalize_context(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("context must be a string.")
        cleaned = _normalize_text(v)
        return cleaned if cleaned else None

    @field_validator("time_horizon")
    @classmethod
    def validate_time_horizon(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = _normalize_text(v)
        # Accept: "12 months", "3 years", "6M", "1Y", "short-term" etc.
        if len(v) > 50:
            raise ValueError("time_horizon is too long (max 50 chars).")
        return v

    # ── Computed Properties ───────────────────

    @property
    def word_count(self) -> int:
        return len(self.thesis.split())

    @property
    def is_detailed(self) -> bool:
        """Thesis with 50+ words is considered detailed."""
        return self.word_count >= 50

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "thesis": (
                        "HDFC Bank will outperform the Nifty Bank index over the next "
                        "12 months because its NIM expansion cycle is still underway, "
                        "CASA ratio above 40% provides structural cost advantage, and "
                        "credit growth guidance of 15-17% is achievable given corporate "
                        "capex recovery."
                    ),
                    "time_horizon": "12 months",
                    "asset_class": "equity",
                    "use_cache": True,
                }
            ]
        }
    }


# ─────────────────────────────────────────────
# 4. IngestRequest
# ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    """
    Request schema for ingesting a document by URL.
    Used by POST /api/ingest endpoint (admin/internal).

    For file uploads, use multipart/form-data with IngestFileRequest.
    """

    url: str = Field(
        ...,
        description="Public URL of the document to ingest (PDF or web page)",
        examples=[
            "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12345",
            "https://www.sebi.gov.in/legal/circulars/dec-2024/circular.pdf",
        ],
    )
    source_type: SourceType = Field(
        default=SourceType.UNKNOWN,
        description="Type of source document",
    )
    company: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Company name if document is company-specific",
        examples=["HDFC Bank", "TCS", "Reliance Industries"],
    )
    date: Optional[str] = Field(
        default=None,
        description="Publication date in YYYY-MM-DD format",
        examples=["2025-12-31", "2026-01-15"],
    )
    title: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Document title override (auto-extracted if not provided)",
    )
    force_reingest: bool = Field(
        default=False,
        description="If True, re-ingests even if document was already processed",
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Pinecone namespace to store vectors in (defaults to source_type)",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                "URL must start with http:// or https://"
            )
        if len(v) > 2000:
            raise ValueError("URL is too long (max 2000 chars).")
        # Block localhost/internal network URLs in production
        blocked_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "192.168.", "10.0."]
        for blocked in blocked_hosts:
            if blocked in v:
                raise ValueError(
                    f"Ingestion of internal/localhost URLs is not permitted."
                )
        return v

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        import datetime
        try:
            datetime.date.fromisoformat(v)
        except ValueError:
            raise ValueError(
                f"date must be in YYYY-MM-DD format. Got: '{v}'"
            )
        return v

    @field_validator("company", mode="before")
    @classmethod
    def normalize_company(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _normalize_text(v) or None


class IngestFileRequest(BaseModel):
    """
    Metadata for file upload ingestion.
    Sent as form fields alongside multipart file upload.
    """

    source_type: SourceType = Field(default=SourceType.UNKNOWN)
    company: Optional[str] = Field(default=None, max_length=100)
    date: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None, max_length=300)
    force_reingest: bool = Field(default=False)


# ─────────────────────────────────────────────
# 5. HealthCheckRequest
# ─────────────────────────────────────────────

class HealthCheckRequest(BaseModel):
    """
    Optional request body for detailed health checks.
    Used by POST /health/detailed endpoint.
    """

    check_groq: bool = Field(
        default=True,
        description="Include Groq API connectivity check",
    )
    check_pinecone: bool = Field(
        default=True,
        description="Include Pinecone connectivity check",
    )
    check_redis: bool = Field(
        default=True,
        description="Include Redis connectivity check",
    )


# ─────────────────────────────────────────────
# 6. ThesisCompareRequest
# ─────────────────────────────────────────────

class ThesisCompareRequest(BaseModel):
    """
    Request schema for comparing two investment theses side-by-side.
    Used by POST /api/thesis/compare endpoint.
    """

    thesis_a: str = Field(
        ...,
        min_length=THESIS_MIN_LENGTH,
        max_length=THESIS_MAX_LENGTH,
        description="First investment thesis",
    )
    thesis_b: str = Field(
        ...,
        min_length=THESIS_MIN_LENGTH,
        max_length=THESIS_MAX_LENGTH,
        description="Second investment thesis to compare against",
    )
    use_cache: bool = Field(default=True)

    @field_validator("thesis_a", "thesis_b", mode="before")
    @classmethod
    def normalize_theses(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Thesis must be a string.")
        return _normalize_text(v)

    @field_validator("thesis_a", "thesis_b")
    @classmethod
    def validate_theses(cls, v: str) -> str:
        _check_not_empty(v, "Thesis")
        _check_ascii_ratio(v, "Thesis")
        _check_min_word_count(v, min_words=5, field_name="Thesis")
        return v

    @model_validator(mode="after")
    def theses_must_differ(self) -> "ThesisCompareRequest":
        if self.thesis_a.lower().strip() == self.thesis_b.lower().strip():
            raise ValueError(
                "thesis_a and thesis_b cannot be identical. "
                "Provide two different theses to compare."
            )
        return self


# ─────────────────────────────────────────────
# PRIVATE VALIDATORS
# ─────────────────────────────────────────────

def _validate_thesis_structure(thesis: str) -> None:
    """
    Validates that a thesis has minimum structural validity.
    A valid thesis needs:
    1. A subject (company/asset keyword OR proper noun)
    2. Some directional or analytical language
    3. At least implied reasoning (more than just a statement)

    This is intentionally lenient — Agent 6 does deep validation.
    We only block the most obvious non-thesis inputs here.
    """
    thesis_lower = thesis.lower()

    # Must not be a pure question (questions go to RAG, not thesis validator)
    if thesis.strip().endswith("?") and len(thesis.split()) < 15:
        raise ValueError(
            "This looks like a question, not an investment thesis. "
            "Use the RAG Query endpoint for questions. "
            "A thesis should make a claim, e.g., "
            "'HDFC will outperform because...'"
        )

    # Must have at least some financial/directional language
    directional_words = {
        "will", "should", "expect", "believe", "think", "thesis",
        "outperform", "underperform", "overvalued", "undervalued",
        "bullish", "bearish", "growth", "decline", "risk", "opportunity",
        "because", "due to", "driven by", "given", "despite", "although",
        "increase", "decrease", "expand", "compress", "rise", "fall",
        "strong", "weak", "positive", "negative", "upside", "downside",
    }
    words_in_thesis = set(thesis_lower.split())
    if not words_in_thesis.intersection(directional_words):
        raise ValueError(
            "Thesis does not appear to contain directional or analytical language. "
            "A valid thesis should include claims like 'will outperform', "
            "'is overvalued', 'driven by', 'because', etc."
        )
