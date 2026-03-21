# backend/utils/logger.py
"""
FinThesisGuard AI — Structured Logging System
ArkAngel Financial Solutions

Centralized logging using loguru. Every module imports `logger` from here.
Supports: colored dev console, JSON production logs, agent timing, query tracking.
"""

import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any
from functools import wraps
import time

from loguru import logger as _loguru_logger

from backend.config import (
    settings,
    AGENT_LATENCY_TARGETS,
    LOG_ROTATION_SIZE,
    LOG_RETENTION_DAYS,
    LOG_DIR,
)


# ─────────────────────────────────────────────
# LOG DIRECTORY SETUP
# ─────────────────────────────────────────────

def _ensure_log_dir() -> Path:
    """Create log directory if it doesn't exist."""
    log_path = Path(LOG_DIR)
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


LOG_PATH = _ensure_log_dir()

# Log file paths
APP_LOG_FILE      = LOG_PATH / "app.log"
AGENTS_LOG_FILE   = LOG_PATH / "agents.log"
ERRORS_LOG_FILE   = LOG_PATH / "errors.log"
QUERIES_LOG_FILE  = LOG_PATH / "queries.log"
METRICS_LOG_FILE  = LOG_PATH / "metrics.log"


# ─────────────────────────────────────────────
# CUSTOM LOG FORMATS
# ─────────────────────────────────────────────

# Human-readable format for development console
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Structured format for file logs (development)
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)


# ─────────────────────────────────────────────
# JSON SERIALIZER FOR PRODUCTION
# ─────────────────────────────────────────────

def _json_formatter(record: dict) -> str:
    """
    Formats log record as a JSON string for production log aggregation.
    Compatible with: Datadog, CloudWatch, ELK Stack, GCP Logging.
    """
    log_entry = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
        "level": record["level"].name,
        "logger": record["name"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
        "process_id": record["process"].id,
        "thread_id": record["thread"].id,
        "environment": settings.environment,
        "app_version": settings.app_version,
    }

    # Attach any extra fields passed via logger.bind() or extra={}
    if record.get("extra"):
        for key, value in record["extra"].items():
            if key not in log_entry:
                log_entry[key] = value

    # Attach exception info if present
    if record["exception"]:
        exc_type, exc_value, exc_tb = record["exception"]
        log_entry["exception"] = {
            "type": exc_type.__name__ if exc_type else None,
            "message": str(exc_value) if exc_value else None,
            "traceback": traceback.format_tb(exc_tb) if exc_tb else None,
        }

    return json.dumps(log_entry, default=str) + "\n"


# ─────────────────────────────────────────────
# LOGGER CONFIGURATION
# ─────────────────────────────────────────────

def _configure_logger() -> None:
    """
    Removes default loguru handler and configures all custom sinks.
    Called once at module load time.
    """

    # Remove loguru default stderr handler
    _loguru_logger.remove()

    log_level = settings.log_level  # DEBUG | INFO | WARNING | ERROR

    # ── SINK 1: Console Output ────────────────────────────────────────────────
    if settings.is_production:
        # Production: JSON to stdout (captured by container log driver)
        _loguru_logger.add(
            sink=sys.stdout,
            level=log_level,
            format="{message}",
            serialize=False,
            filter=lambda record: True,
            colorize=False,
        )
    else:
        # Development: colored human-readable console
        _loguru_logger.add(
            sink=sys.stderr,
            level=log_level,
            format=CONSOLE_FORMAT,
            colorize=True,
            backtrace=True,
            diagnose=settings.debug_mode,
        )

    # ── SINK 2: app.log — All logs ────────────────────────────────────────────
    _loguru_logger.add(
        sink=str(APP_LOG_FILE),
        level=log_level,
        format=_json_formatter if settings.is_production else FILE_FORMAT,
        rotation=LOG_ROTATION_SIZE,
        retention=f"{LOG_RETENTION_DAYS} days",
        compression="zip",
        encoding="utf-8",
        backtrace=True,
        diagnose=False,
        enqueue=True,                                  # Thread-safe async writes
    )

    # ── SINK 3: agents.log — Agent-specific logs only ─────────────────────────
    _loguru_logger.add(
        sink=str(AGENTS_LOG_FILE),
        level="DEBUG",
        format=_json_formatter if settings.is_production else FILE_FORMAT,
        rotation=LOG_ROTATION_SIZE,
        retention=f"{LOG_RETENTION_DAYS} days",
        compression="zip",
        encoding="utf-8",
        filter=lambda record: record["extra"].get("log_type") in (
            "agent_timing", "agent_event", "agent_error"
        ),
        enqueue=True,
    )

    # ── SINK 4: errors.log — WARNING and above only ───────────────────────────
    _loguru_logger.add(
        sink=str(ERRORS_LOG_FILE),
        level="WARNING",
        format=_json_formatter if settings.is_production else FILE_FORMAT,
        rotation=LOG_ROTATION_SIZE,
        retention=f"{LOG_RETENTION_DAYS} days",
        compression="zip",
        encoding="utf-8",
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )

    # ── SINK 5: queries.log — Query tracking only ─────────────────────────────
    _loguru_logger.add(
        sink=str(QUERIES_LOG_FILE),
        level="INFO",
        format=_json_formatter if settings.is_production else FILE_FORMAT,
        rotation=LOG_ROTATION_SIZE,
        retention=f"{LOG_RETENTION_DAYS} days",
        compression="zip",
        encoding="utf-8",
        filter=lambda record: record["extra"].get("log_type") == "query",
        enqueue=True,
    )

    # ── SINK 6: metrics.log — Performance metrics only ────────────────────────
    _loguru_logger.add(
        sink=str(METRICS_LOG_FILE),
        level="INFO",
        format=_json_formatter if settings.is_production else FILE_FORMAT,
        rotation=LOG_ROTATION_SIZE,
        retention=f"{LOG_RETENTION_DAYS} days",
        compression="zip",
        encoding="utf-8",
        filter=lambda record: record["extra"].get("log_type") == "metric",
        enqueue=True,
    )


# Run configuration immediately on import
_configure_logger()


# ─────────────────────────────────────────────
# EXPORTED LOGGER INSTANCE
# All modules use: from backend.utils.logger import logger
# ─────────────────────────────────────────────

logger = _loguru_logger


# ─────────────────────────────────────────────
# HELPER: AGENT TIMING LOGGER
# ─────────────────────────────────────────────

def log_agent_timing(
    agent_name: str,
    duration_ms: float,
    success: bool = True,
    metadata: Optional[dict] = None,
) -> None:
    """
    Logs structured timing data for each agent execution.
    Compares against target latency from config.AGENT_LATENCY_TARGETS.

    Usage:
        log_agent_timing("agent1_acronym_resolver", 45.3)
        log_agent_timing("agent6_thesis_analyzer", 850.0, success=False)
    """
    target_ms = AGENT_LATENCY_TARGETS.get(agent_name, MAX_LATENCY_MS)
    within_target = duration_ms <= target_ms
    overage_pct = ((duration_ms - target_ms) / target_ms * 100) if not within_target else 0.0

    log_data = {
        "log_type": "agent_timing",
        "agent_name": agent_name,
        "duration_ms": round(duration_ms, 2),
        "target_ms": target_ms,
        "within_target": within_target,
        "overage_pct": round(overage_pct, 1),
        "success": success,
        **(metadata or {}),
    }

    bound_logger = logger.bind(**log_data)

    if not success:
        bound_logger.error(
            f"[AGENT FAILED] {agent_name} | {duration_ms:.1f}ms"
        )
    elif not within_target:
        bound_logger.warning(
            f"[AGENT SLOW] {agent_name} | {duration_ms:.1f}ms "
            f"(target: {target_ms}ms, +{overage_pct:.1f}% over)"
        )
    else:
        bound_logger.info(
            f"[AGENT OK] {agent_name} | {duration_ms:.1f}ms "
            f"(target: {target_ms}ms ✓)"
        )


# ─────────────────────────────────────────────
# HELPER: QUERY LOGGER
# ─────────────────────────────────────────────

def log_query(
    query: str,
    response_time_ms: float,
    confidence: str,
    query_type: str = "unknown",
    agents_used: Optional[list] = None,
    cache_hit: bool = False,
    conflict_detected: bool = False,
    citations_count: int = 0,
    user_ip: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """
    Logs structured query execution data to queries.log.
    Used for analytics, debugging, and performance monitoring.

    Usage:
        log_query(
            query="HDFC NIM Q3 FY26",
            response_time_ms=1450.0,
            confidence="High",
            query_type="COMPARATIVE",
            agents_used=["agent1", "agent2", "agent3", "agent4", "agent5"],
            cache_hit=False,
            conflict_detected=True,
            citations_count=6,
        )
    """
    # Sanitize: truncate long queries for logs
    query_preview = query[:200] + "..." if len(query) > 200 else query
    within_target = response_time_ms <= settings.__class__.__dict__.get(
        "max_latency_ms", 3000
    )

    log_data = {
        "log_type": "query",
        "query_preview": query_preview,
        "query_length": len(query),
        "query_type": query_type,
        "response_time_ms": round(response_time_ms, 2),
        "within_latency_target": response_time_ms <= 3000,
        "confidence": confidence,
        "agents_used": agents_used or [],
        "agents_count": len(agents_used) if agents_used else 0,
        "cache_hit": cache_hit,
        "conflict_detected": conflict_detected,
        "citations_count": citations_count,
        "user_ip_hash": _hash_ip(user_ip) if user_ip else None,
        "error": error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    bound_logger = logger.bind(**log_data)

    if error:
        bound_logger.error(
            f"[QUERY ERROR] {query_type} | {response_time_ms:.0f}ms | "
            f"confidence={confidence} | error={error}"
        )
    elif response_time_ms > 3000:
        bound_logger.warning(
            f"[QUERY SLOW] {query_type} | {response_time_ms:.0f}ms | "
            f"confidence={confidence} | citations={citations_count}"
        )
    else:
        bound_logger.info(
            f"[QUERY OK] {query_type} | {response_time_ms:.0f}ms | "
            f"confidence={confidence} | citations={citations_count} | "
            f"cache={'HIT' if cache_hit else 'MISS'}"
        )


# ─────────────────────────────────────────────
# HELPER: METRIC LOGGER
# ─────────────────────────────────────────────

def log_metric(
    metric_name: str,
    value: float,
    unit: str = "",
    tags: Optional[dict] = None,
) -> None:
    """
    Logs a scalar metric to metrics.log.
    Use for tracking: token counts, cache hit rates, accuracy scores, etc.

    Usage:
        log_metric("llm_tokens_used", 1024, unit="tokens")
        log_metric("cache_hit_rate", 0.72, unit="ratio", tags={"endpoint": "/api/query"})
    """
    log_data = {
        "log_type": "metric",
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
        "tags": tags or {},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    logger.bind(**log_data).info(
        f"[METRIC] {metric_name}={value}{' ' + unit if unit else ''}"
    )


# ─────────────────────────────────────────────
# HELPER: INGESTION LOGGER
# ─────────────────────────────────────────────

def log_ingestion(
    file_path: str,
    source_type: str,
    chunks_created: int,
    vectors_upserted: int,
    duration_ms: float,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """
    Logs document ingestion events.

    Usage:
        log_ingestion("data/raw/hdfc_ar_2025.pdf", "annual_report", 450, 450, 3200.0)
    """
    log_data = {
        "log_type": "agent_event",
        "event": "document_ingestion",
        "file_path": file_path,
        "file_name": Path(file_path).name,
        "source_type": source_type,
        "chunks_created": chunks_created,
        "vectors_upserted": vectors_upserted,
        "duration_ms": round(duration_ms, 2),
        "success": success,
        "error": error,
    }

    bound_logger = logger.bind(**log_data)

    if not success:
        bound_logger.error(
            f"[INGEST FAILED] {Path(file_path).name} | error={error}"
        )
    else:
        bound_logger.info(
            f"[INGEST OK] {Path(file_path).name} | "
            f"source={source_type} | chunks={chunks_created} | "
            f"vectors={vectors_upserted} | {duration_ms:.0f}ms"
        )


# ─────────────────────────────────────────────
# HELPER: LLM CALL LOGGER
# ─────────────────────────────────────────────

def log_llm_call(
    agent_name: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_ms: float,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """
    Logs every LLM API call for token usage tracking and cost analysis.

    Usage:
        log_llm_call("agent6_thesis_analyzer", "llama-3.3-70b-versatile", 512, 256, 320.5)
    """
    total_tokens = prompt_tokens + completion_tokens
    # Approximate cost: Groq Llama-3.3-70B ~$0.59/1M tokens
    estimated_cost_usd = total_tokens * 0.00000059

    log_data = {
        "log_type": "metric",
        "metric_name": "llm_call",
        "agent_name": agent_name,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "duration_ms": round(duration_ms, 2),
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "success": success,
        "error": error,
    }

    bound_logger = logger.bind(**log_data)

    if not success:
        bound_logger.error(
            f"[LLM FAILED] {agent_name} | model={model} | error={error}"
        )
    else:
        bound_logger.debug(
            f"[LLM OK] {agent_name} | model={model} | "
            f"tokens={total_tokens} (in={prompt_tokens}, out={completion_tokens}) | "
            f"{duration_ms:.0f}ms | ~${estimated_cost_usd:.6f}"
        )


# ─────────────────────────────────────────────
# DECORATOR: AUTOMATIC FUNCTION TIMING
# ─────────────────────────────────────────────

def timed(agent_name: Optional[str] = None):
    """
    Decorator that automatically times any async or sync function
    and calls log_agent_timing() on completion.

    Usage:
        @timed("agent1_acronym_resolver")
        async def resolve(query: str) -> str:
            ...

        @timed()  # Uses function name as agent_name
        def some_function():
            ...
    """
    def decorator(func):
        name = agent_name or func.__name__

        if __import__("asyncio").iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                success = True
                error_msg = None
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error_msg = str(e)
                    logger.exception(
                        f"[AGENT ERROR] {name} raised {type(e).__name__}: {e}"
                    )
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    log_agent_timing(
                        agent_name=name,
                        duration_ms=duration_ms,
                        success=success,
                        metadata={"error": error_msg} if error_msg else None,
                    )
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                success = True
                error_msg = None
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error_msg = str(e)
                    logger.exception(
                        f"[AGENT ERROR] {name} raised {type(e).__name__}: {e}"
                    )
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    log_agent_timing(
                        agent_name=name,
                        duration_ms=duration_ms,
                        success=success,
                        metadata={"error": error_msg} if error_msg else None,
                    )
            return sync_wrapper

    return decorator


# ─────────────────────────────────────────────
# CONTEXT LOGGER — For Request-Scoped Logging
# ─────────────────────────────────────────────

def get_request_logger(
    request_id: str,
    endpoint: str,
    user_ip: Optional[str] = None,
) -> Any:
    """
    Returns a logger instance bound to a specific request context.
    Use in FastAPI route handlers.

    Usage:
        req_logger = get_request_logger("abc-123", "/api/query", "192.168.1.1")
        req_logger.info("Processing query")
    """
    return logger.bind(
        request_id=request_id,
        endpoint=endpoint,
        user_ip_hash=_hash_ip(user_ip) if user_ip else None,
    )


# ─────────────────────────────────────────────
# STARTUP LOG
# ─────────────────────────────────────────────

def log_startup(config_summary: dict) -> None:
    """
    Logs application startup with full config summary.
    Called from main.py on app startup.
    """
    logger.info("=" * 60)
    logger.info(f"  {config_summary.get('app_name', 'FinThesisGuard AI')} Starting Up")
    logger.info(f"  Version  : {config_summary.get('version')}")
    logger.info(f"  Env      : {config_summary.get('environment')}")
    logger.info(f"  Model    : {config_summary.get('groq_model')}")
    logger.info(f"  Embedder : {config_summary.get('embedding_model')}")
    logger.info(f"  Pinecone : {config_summary.get('pinecone_index')} @ {config_summary.get('pinecone_region')}")
    logger.info(f"  Redis    : {config_summary.get('redis_url')}")
    logger.info(f"  Cache    : {'ENABLED' if config_summary.get('cache_enabled') else 'DISABLED'}")
    logger.info(f"  Debug    : {'ON' if config_summary.get('debug_mode') else 'OFF'}")
    logger.info(f"  Log Dir  : {LOG_PATH.resolve()}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────
# PRIVATE UTILITIES
# ─────────────────────────────────────────────

def _hash_ip(ip: str) -> str:
    """One-way hash IP address for privacy-safe logging."""
    import hashlib
    return hashlib.sha256(ip.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────
# MAX_LATENCY_MS local reference (from config)
# ─────────────────────────────────────────────
try:
    from backend.config import MAX_LATENCY_MS
except ImportError:
    MAX_LATENCY_MS = 3000

import logging
from tqdm import tqdm

class _TqdmLoggingHandler(logging.Handler):
    """Routes all logging output through tqdm.write() so bars don't break."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)

# For loguru, patch the sink:
from loguru import logger as _loguru_logger
import sys

_loguru_logger.remove()                        # Remove default stderr sink
_loguru_logger.add(
    lambda msg: tqdm.write(msg, end=""),       # Route through tqdm
    colorize  = True,
    format    = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> | "
        "<level>{message}</level>"
    ),
)
