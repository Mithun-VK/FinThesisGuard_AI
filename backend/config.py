# backend/config.py
"""
FinThesisGuard AI — Central Configuration
ArkAngel Financial Solutions
Single source of truth for all settings, constants, and environment variables.
Every other module imports from this file.
"""

import os
from functools import lru_cache
from typing import Dict, Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────
# ENVIRONMENT & MODEL CONSTANTS
# ─────────────────────────────────────────────

GROQ_MODEL: str = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST: str = "llama-3.1-8b-instant"          # Fast fallback for simple tasks
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
PINECONE_INDEX: str = "finthesisguard"
PINECONE_METRIC: str = "cosine"
PINECONE_CLOUD: str = "aws"
PINECONE_REGION: str = "us-east-1"

# ─────────────────────────────────────────────
# EMBEDDING SETTINGS
# ─────────────────────────────────────────────

EMBEDDING_MODEL: str      = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int  = 384
EMBEDDING_BATCH_SIZE: int = 32


# ─────────────────────────────────────────────
# CACHE TTL SETTINGS (seconds)
# ─────────────────────────────────────────────

CACHE_TTL_THESIS: int     = 7200          # 2 hours — thesis validation results
CACHE_TTL_EMBEDDINGS: int = 86400         # 24 hours — encoded query vectors
CACHE_TTL_HEALTH: int     = 30            # 30 secs — health check pings


# ─────────────────────────────────────────────
# LLM SETTINGS
# ─────────────────────────────────────────────

LLM_TEMPERATURE_FACTUAL: float  = 0.1
LLM_TEMPERATURE_THESIS: float   = 0.3
LLM_TEMPERATURE_CREATIVE: float = 0.7

LLM_MAX_TOKENS_DEFAULT: int     = 1024
LLM_MAX_TOKENS_THESIS: int      = 2048
LLM_MAX_TOKENS_SYNTHESIS: int   = 4096

LLM_RETRY_ATTEMPTS: int         = 3
LLM_RETRY_BACKOFF_BASE: int     = 2       # seconds — exponential: 2^attempt
LLM_REQUEST_TIMEOUT: int        = 30      # seconds


# ─────────────────────────────────────────────
# CACHE NAMESPACES
# ─────────────────────────────────────────────

CACHE_NAMESPACES: dict[str, str] = {
    "rag":     "rag:",
    "thesis":  "thesis:",
    "embed":   "embed:",
    "health":  "health:",
    "agent":   "agent:",
    "ingest":  "ingest:",
    "query":   "query:",
}


# ─────────────────────────────────────────────
# THESIS VALIDATION SETTINGS
# ─────────────────────────────────────────────

THESIS_MIN_LENGTH: int  = 20
THESIS_MAX_LENGTH: int  = 5000

THESIS_MAX_ASSUMPTIONS: int      = 10
THESIS_MAX_BREAK_CONDITIONS: int = 8

THESIS_STRENGTH_THRESHOLDS: dict[str, float] = {
    "strong": 4.0,   # avg risk score < 4.0  → Strong
    "medium": 7.0,   # avg risk score < 7.0  → Medium
                     # avg risk score >= 7.0 → Weak
}

THESIS_RISK_DIMENSIONS: list[str] = [
    "demand_risk",
    "margin_risk",
    "valuation_risk",
    "regulatory_risk",
    "macro_risk",
    "competitive_risk",
]

# ─────────────────────────────────────────────
# CHUNKING CONSTANTS
# ─────────────────────────────────────────────

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
MIN_CHUNK_SIZE: int = 100                               # Discard chunks smaller than this
MAX_CHUNK_SIZE: int = 1500                              # Hard cap

CHUNK_SIZE_BY_SOURCE: Dict[str, int] = {
    "annual_report": 1200,
    "earnings_transcript": 1000,
    "rbi": 800,
    "sebi": 800,
    "broker_research": 1000,
    "news": 600,
    "blog": 500,
    "tax_law": 900,
}

# ─────────────────────────────────────────────
# AUTHORITY WEIGHTS
# Used by Agent 3 (Re-Ranker) for scoring
# ─────────────────────────────────────────────

AUTHORITY_WEIGHTS: Dict[str, float] = {
    "rbi": 1.5,
    "sebi": 1.5,
    "mca": 1.4,
    "annual_report": 1.3,
    "earnings_transcript": 1.2,
    "broker_research": 1.1,
    "tax_law": 1.3,
    "news": 1.0,
    "blog": 0.8,
    "unknown": 0.7,
}

# ─────────────────────────────────────────────
# RE-RANKING FORMULA WEIGHTS
# score = similarity × (1 + RECENCY_WEIGHT×recency)
#       × authority × (1 + OVERLAP_WEIGHT×overlap)
# ─────────────────────────────────────────────

RECENCY_WEIGHT: float = 0.3
OVERLAP_WEIGHT: float = 0.2
RECENCY_DECAY_DAYS: int = 365
MIN_SIMILARITY_THRESHOLD: float = 0.3                  # Discard below this score
TOP_K_RETRIEVAL: int = 10                              # Default chunks to retrieve
TOP_K_RERANKED: int = 6                                # After re-ranking, keep top N

# ─────────────────────────────────────────────
# LATENCY TARGETS (milliseconds)
# ─────────────────────────────────────────────

MAX_LATENCY_MS: int = 2500                             # Total pipeline target
AGENT_LATENCY_TARGETS: Dict[str, int] = {
    "agent1_acronym_resolver": 50,
    "agent2_query_decomposer": 200,
    "agent3_reranker": 100,
    "agent4_conflict_detector": 800,
    "agent5_quant_validator": 300,
    "agent6_thesis_analyzer": 600,
    "agent7_stress_synthesizer": 700,
    "retriever": 400,
}

# ─────────────────────────────────────────────
# LLM GENERATION SETTINGS
# ─────────────────────────────────────────────

LLM_TEMPERATURE_FACTUAL: float = 0.1
LLM_TEMPERATURE_THESIS: float = 0.3
LLM_TEMPERATURE_CREATIVE: float = 0.5
LLM_MAX_TOKENS_DEFAULT: int = 1024
LLM_MAX_TOKENS_THESIS: int = 2048
LLM_MAX_TOKENS_SYNTHESIS: int = 1500
LLM_RETRY_ATTEMPTS: int = 3
LLM_RETRY_BACKOFF_BASE: float = 2.0                   # Exponential backoff base (seconds)
LLM_REQUEST_TIMEOUT: int = 30                          # Seconds

# ─────────────────────────────────────────────
# CACHING
# ─────────────────────────────────────────────

CACHE_TTL: int = 3600                                  # 1 hour default
CACHE_TTL_THESIS: int = 1800                           # 30 min for thesis (more dynamic)
CACHE_TTL_EMBEDDINGS: int = 86400                      # 24 hours (embeddings rarely change)
CACHE_TTL_HEALTH: int = 30                             # 30 sec for health checks

CACHE_NAMESPACES = {
    "rag": "rag:",
    "thesis": "thesis:",
    "embed": "embed:",
    "health": "health:",
}

# ─────────────────────────────────────────────
# CONFLICT DETECTION THRESHOLDS
# Used by Agent 4
# ─────────────────────────────────────────────

CONFLICT_THRESHOLD_PERCENTAGE: float = 0.05           # 5% difference = conflict
CONFLICT_THRESHOLD_ABSOLUTE: float = 0.5              # 0.5 unit difference = conflict
CONFLICT_MIN_SOURCES: int = 2                          # Need at least 2 sources to detect

# ─────────────────────────────────────────────
# THESIS VALIDATION SETTINGS
# Used by Agents 6 & 7
# ─────────────────────────────────────────────

THESIS_STRENGTH_THRESHOLDS: Dict[str, float] = {
    "strong": 4.0,                                     # avg risk score < 4 = Strong
    "medium": 7.0,                                     # avg risk score 4-7 = Medium
    # above 7 = Weak
}

THESIS_RISK_DIMENSIONS: list = [
    "demand_risk",
    "margin_risk",
    "valuation_risk",
    "regulatory_risk",
    "macro_risk",
    "competitive_risk",
]

THESIS_MAX_ASSUMPTIONS: int = 8                        # Max assumptions to extract
THESIS_MAX_BREAK_CONDITIONS: int = 6                   # Max break conditions to flag
THESIS_MIN_LENGTH: int = 20                            # Reject thesis shorter than this
THESIS_MAX_LENGTH: int = 5000                          # Cap thesis input

# ─────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────

EMBEDDING_BATCH_SIZE: int = 32
PINECONE_UPSERT_BATCH_SIZE: int = 100
INGESTION_PARALLEL_FILES: int = 4
CRAWLER_RATE_LIMIT_SECONDS: float = 1.0               # 1 request/second

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_ROTATION_SIZE: str = "10 MB"
LOG_RETENTION_DAYS: int = 7
LOG_DIR: str = "logs"

# ─────────────────────────────────────────────
# API RATE LIMITING
# ─────────────────────────────────────────────

RATE_LIMIT_RAG_PER_MINUTE: int = 60
RATE_LIMIT_THESIS_PER_MINUTE: int = 20
RATE_LIMIT_BATCH_MAX: int = 10

# ─────────────────────────────────────────────
# FINANCIAL ENTITY PATTERNS
# Used by metadata tagger
# ─────────────────────────────────────────────

KNOWN_COMPANIES: list = [
    "HDFC", "ICICI", "SBI", "Axis", "Kotak",
    "TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra",
    "Reliance", "Adani", "Tata", "Bajaj", "Mahindra",
    "Maruti", "ONGC", "NTPC", "Power Grid", "Coal India",
    "Hindustan Unilever", "ITC", "Nestle", "Asian Paints",
    "Sun Pharma", "Dr Reddy", "Cipla", "Divi's",
]

KNOWN_SECTORS: Dict[str, list] = {
    "banking": ["bank", "NBFC", "lending", "credit", "deposits", "NIM", "NPA"],
    "technology": ["software", "IT", "tech", "digital", "cloud", "SaaS"],
    "energy": ["oil", "gas", "power", "solar", "renewable", "petroleum"],
    "pharma": ["pharma", "drug", "medicine", "healthcare", "API"],
    "fmcg": ["consumer", "FMCG", "foods", "beverages", "personal care"],
    "auto": ["automobile", "vehicle", "EV", "electric vehicle", "two-wheeler"],
}

# ─────────────────────────────────────────────
# SEBI / REGULATORY SOURCE PATTERNS
# ─────────────────────────────────────────────

REGULATORY_URL_PATTERNS: Dict[str, list] = {
    "rbi": ["rbi.org.in", "rbi.gov.in"],
    "sebi": ["sebi.gov.in"],
    "mca": ["mca.gov.in"],
    "tax_law": ["indiacode.nic.in", "incometaxindia.gov.in"],
    "bse": ["bseindia.com"],
    "nse": ["nseindia.com"],
}

NEWS_SOURCES: list = [
    "economictimes.indiatimes.com",
    "livemint.com",
    "moneycontrol.com",
    "businessstandard.com",
    "financialexpress.com",
]


# ─────────────────────────────────────────────
# PYDANTIC SETTINGS CLASS
# ─────────────────────────────────────────────

class Settings(BaseSettings):
    """
    FinThesisGuard AI — Environment Settings
    Loaded from .env file. All secrets live here.
    """
    CACHE_TTL: int            = 3600          # 1 hour  — RAG query responses
    MAX_LATENCY_MS: int = 2500                             # Total pipeline target

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",                                # Ignore unknown env vars
    )
    # ── Retrieval ─────────────────────────────────────────────────────────────
    RAG_TOP_K:    int = 10   # ← THIS was missing
    THESIS_TOP_K: int = 12   # ← THIS was missing
    MAX_SUB_QUERIES: int = 4  # ← add if used elsewhere

    # ── API Keys ──────────────────────────────
    groq_api_key: str = Field(..., description="Groq API key for Llama inference")
    pinecone_api_key: str = Field(..., description="Pinecone vector DB API key")

    # ── Optional / Defaulted ──────────────────
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    environment: str = Field(
        default="development",
        description="Environment: development | staging | production"
    )
    log_level: str = Field(
        default="INFO",
        description="Log level: DEBUG | INFO | WARNING | ERROR"
    )
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    app_version: str = Field(default="1.0.0")
    app_name: str = Field(default="FinThesisGuard AI")
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated list of allowed CORS origins"
    )

    # ── Model Overrides (optional) ────────────
    groq_model: str = Field(
        default=GROQ_MODEL,
        description="Primary Groq model name"
    )
    groq_model_fast: str = Field(
        default=GROQ_MODEL_FAST,
        description="Fast Groq model for lightweight tasks"
    )
    embedding_model: str = Field(
        default=EMBEDDING_MODEL,
        description="Sentence-transformers model name"
    )
    pinecone_index: str = Field(
        default=PINECONE_INDEX,
        description="Pinecone index name"
    )
    pinecone_cloud: str = Field(default=PINECONE_CLOUD)
    pinecone_region: str = Field(default=PINECONE_REGION)

    # ── Feature Flags ─────────────────────────
    enable_cache: bool = Field(
        default=True,
        description="Enable Redis caching"
    )
    enable_conflict_detection: bool = Field(
        default=True,
        description="Enable Agent 4 conflict detection"
    )
    enable_quant_validation: bool = Field(
        default=True,
        description="Enable Agent 5 quantitative validation"
    )
    enable_streaming: bool = Field(
        default=True,
        description="Enable streaming LLM responses"
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable verbose debug logging"
    )

    # ─────────────────────────────────────────
    # VALIDATORS
    # ─────────────────────────────────────────

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    @field_validator("app_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1024 <= v <= 65535):
            raise ValueError("app_port must be between 1024 and 65535")
        return v

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY is missing or still set to placeholder value. "
                "Set it in your .env file."
            )
        if not self.pinecone_api_key or self.pinecone_api_key == "your_pinecone_api_key_here":
            raise ValueError(
                "PINECONE_API_KEY is missing or still set to placeholder value. "
                "Set it in your .env file."
            )
        return self

    # ─────────────────────────────────────────
    # COMPUTED PROPERTIES
    # ─────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def cors_origins_list(self) -> list:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def redis_host(self) -> str:
        """Extract host from Redis URL for health checks."""
        return self.redis_url.replace("redis://", "").split(":")[0]

    @property
    def redis_port(self) -> int:
        """Extract port from Redis URL."""
        try:
            return int(self.redis_url.split(":")[-1])
        except (ValueError, IndexError):
            return 6379


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# Cached — only loaded once per process
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the cached Settings singleton.
    Use this in dependency injection contexts.
    """
    return Settings()


# Direct export for simple imports:
# from backend.config import settings
try:
    settings = get_settings()
except Exception as e:
    raise RuntimeError(
        f"Failed to load FinThesisGuard configuration: {e}\n"
        "Ensure your .env file exists and contains valid API keys."
    ) from e


# ─────────────────────────────────────────────
# CONFIGURATION SUMMARY (logged at startup)
# ─────────────────────────────────────────────

def get_config_summary() -> dict:
    """
    Returns a safe, non-sensitive summary of the current config.
    Used at startup to confirm configuration loaded correctly.
    """
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "groq_model": settings.groq_model,
        "embedding_model": settings.embedding_model,
        "pinecone_index": settings.pinecone_index,
        "pinecone_region": settings.pinecone_region,
        "redis_url": settings.redis_url,
        "cache_enabled": settings.enable_cache,
        "conflict_detection": settings.enable_conflict_detection,
        "quant_validation": settings.enable_quant_validation,
        "streaming_enabled": settings.enable_streaming,
        "debug_mode": settings.debug_mode,
        "log_level": settings.log_level,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_latency_ms": MAX_LATENCY_MS,
        "groq_api_key_set": bool(settings.groq_api_key),
        "pinecone_api_key_set": bool(settings.pinecone_api_key),
    }
