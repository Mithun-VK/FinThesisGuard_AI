# api/routes_health.py
# ─────────────────────────────────────────────────────────────────────────────
# FinThesisGuard AI — Health & Monitoring Routes
# ArkAngel Financial Solutions
#
# Endpoints
#   GET  /health          → overall liveness + per-service status
#   GET  /metrics         → query throughput, latency percentiles, cache stats
#   GET  /ready           → deep readiness probe (Pinecone + Redis + Groq)
#   GET  /agents/status   → per-agent last-run time + rolling success rate
#
# KEY FIX: ServiceStatus is a Literal type alias (not an Enum).
# Never use ServiceStatus.ok / ServiceStatus.unavailable — use plain strings.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from backend.config import settings, AGENT_LATENCY_TARGETS, MAX_LATENCY_MS
from backend.models.response_models import (
    HealthResponse,
    MetricsResponse,
    ServiceHealth,
    ServiceStatus,
)
from backend.pipeline.orchestrator import orchestrator
from backend.utils.cache import cache
from backend.utils.logger import logger, log_metric

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(
    prefix="",
    tags=["Health & Monitoring"],
)

# ── Process start time ────────────────────────────────────────────────────────

_PROCESS_START:     float = time.monotonic()
_PROCESS_START_UTC: str   = datetime.now(timezone.utc).isoformat()

# ── ServiceStatus string constants ────────────────────────────────────────────
# ServiceStatus is Literal["ok", "degraded", "unavailable", "unknown"] — a type
# alias, NOT an Enum. Always use these string constants, never ServiceStatus.xxx.

_STATUS_OK          = "ok"
_STATUS_DEGRADED    = "degraded"
_STATUS_UNAVAILABLE = "unavailable"
_STATUS_UNKNOWN     = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _uptime_seconds() -> float:
    return round(time.monotonic() - _PROCESS_START, 2)


def _overall_status(services: Dict[str, ServiceHealth]) -> ServiceStatus:
    """
    Derive overall status from individual service statuses.
    unavailable > degraded > ok > unknown
    """
    statuses = {s.status for s in services.values()}
    if _STATUS_UNAVAILABLE in statuses:
        return _STATUS_UNAVAILABLE
    if _STATUS_DEGRADED in statuses:
        return _STATUS_DEGRADED
    if _STATUS_OK in statuses:
        return _STATUS_OK
    return _STATUS_UNKNOWN


def _make_error_health(name: str, exc: Exception, elapsed_ms: float) -> ServiceHealth:
    """Build a ServiceHealth for an unexpected probe exception."""
    return ServiceHealth(
        name=name,
        status=_STATUS_UNAVAILABLE,
        latency_ms=round(elapsed_ms, 2),
        error=f"{type(exc).__name__}: {exc}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROBE HELPERS — each returns ServiceHealth, NEVER raises
# ─────────────────────────────────────────────────────────────────────────────

async def _probe_redis(timeout: float = 2.0) -> ServiceHealth:
    """Write-then-read round-trip to Redis."""
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(
            cache.set_cache("health:probe", {"ping": "pong"}, ttl=10),
            timeout=timeout,
        )
        val = await asyncio.wait_for(
            cache.get_cached("health:probe"),
            timeout=timeout,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        if val is None:
            return ServiceHealth(
                name="redis", status=_STATUS_DEGRADED,
                latency_ms=round(latency_ms, 2),
                error="write-then-read returned None",
            )
        return ServiceHealth(
            name="redis", status=_STATUS_OK,
            latency_ms=round(latency_ms, 2),
        )
    except asyncio.TimeoutError:
        return ServiceHealth(
            name="redis", status=_STATUS_UNAVAILABLE,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            error=f"timeout after {timeout}s",
        )
    except Exception as exc:
        logger.warning(f"[PROBE] redis: {exc}")
        return _make_error_health("redis", exc, (time.monotonic() - t0) * 1000)


async def _probe_pinecone(timeout: float = 10.0) -> ServiceHealth:
    """
    initialize_index() (idempotent) then get_index_stats().
    Generous 10 s timeout — first call requires a control-plane handshake.
    """
    t0 = time.monotonic()
    try:
        from backend.retrieval.pinecone_client import pinecone_client  # noqa: PLC0415

        await asyncio.wait_for(pinecone_client.initialize_index(), timeout=timeout)
        stats = await asyncio.wait_for(pinecone_client.get_index_stats(), timeout=timeout)

        latency_ms    = (time.monotonic() - t0) * 1000
        total_vectors = stats.get("total_vector_count", 0)
        return ServiceHealth(
            name="pinecone", status=_STATUS_OK,
            latency_ms=round(latency_ms, 2),
            version=f"{total_vectors:,} vectors",
        )
    except asyncio.TimeoutError:
        return ServiceHealth(
            name="pinecone", status=_STATUS_UNAVAILABLE,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            error=f"timeout after {timeout}s",
        )
    except Exception as exc:
        logger.warning(f"[PROBE] pinecone: {exc}")
        return _make_error_health("pinecone", exc, (time.monotonic() - t0) * 1000)


async def _probe_groq(timeout: float = 8.0) -> ServiceHealth:
    """1-token probe to Groq — verifies API key and reachability."""
    t0 = time.monotonic()
    try:
        from backend.utils.llm_client import llm_client  # noqa: PLC0415

        response = await asyncio.wait_for(
            llm_client.call_llm(
                prompt        = "Reply with the single word: ok",
                system_prompt = "You are a health check bot. Reply with exactly one word.",
                max_tokens    = 5,
            ),
            timeout=timeout,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        reply      = (response or "").strip().lower()

        if "ok" not in reply:
            return ServiceHealth(
                name="groq", status=_STATUS_DEGRADED,
                latency_ms=round(latency_ms, 2),
                version=settings.groq_model,
                error=f"unexpected probe reply: {reply!r}",
            )
        return ServiceHealth(
            name="groq", status=_STATUS_OK,
            latency_ms=round(latency_ms, 2),
            version=settings.groq_model,
        )
    except asyncio.TimeoutError:
        return ServiceHealth(
            name="groq", status=_STATUS_UNAVAILABLE,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            version=settings.groq_model,
            error=f"timeout after {timeout}s",
        )
    except Exception as exc:
        logger.warning(f"[PROBE] groq: {exc}")
        return _make_error_health("groq", exc, (time.monotonic() - t0) * 1000)


# ─────────────────────────────────────────────────────────────────────────────
# GET /health  — lightweight liveness (reads CB state, no network I/O)
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    responses={
        200: {"description": "System is live"},
        503: {"description": "One or more critical services unavailable"},
    },
)
async def get_health() -> JSONResponse:
    t0 = time.monotonic()

    try:
        orch_health: Dict[str, Any] = await orchestrator.health()
    except Exception as exc:
        logger.warning(f"[HEALTH] orchestrator.health() error: {exc}")
        orch_health = {}

    # Map orchestrator CB strings → ServiceHealth objects using plain string comparisons
    services: Dict[str, ServiceHealth] = {}
    for svc_name, svc_status_str in orch_health.get("services", {}).items():
        s = str(svc_status_str)
        if s.startswith("ok") or s == "closed":
            svc_enum = _STATUS_OK
        elif s == "open":
            svc_enum = _STATUS_UNAVAILABLE
        elif s in ("half_open", "degraded"):
            svc_enum = _STATUS_DEGRADED
        else:
            svc_enum = _STATUS_UNKNOWN
        services[svc_name] = ServiceHealth(name=svc_name, status=svc_enum)

    if not services:
        for svc in ("redis", "pinecone", "groq"):
            services[svc] = ServiceHealth(
                name=svc, status=_STATUS_UNKNOWN,
                error="Orchestrator not initialized",
            )

    overall     = _overall_status(services)
    http_code   = status.HTTP_200_OK if overall != _STATUS_UNAVAILABLE else 503
    metrics_snap = orchestrator.get_metrics()

    payload = HealthResponse(
        status         = overall,
        app_name       = settings.app_name,
        version        = settings.app_version,
        environment    = settings.environment,
        uptime_seconds = _uptime_seconds(),
        services       = services,
        cache_stats    = {
            "total_queries"  : metrics_snap["total_queries"],
            "cache_hits"     : metrics_snap["cache_hits"],
            "cache_hit_rate" : round(metrics_snap["cache_hit_rate"] * 100, 1),
            "errors"         : metrics_snap["errors"],
        },
    )

    logger.info(
        f"[HEALTH] status={overall} uptime={_uptime_seconds():.0f}s "
        f"latency={(time.monotonic() - t0)*1000:.1f}ms"
    )
    return JSONResponse(content=payload.model_dump(mode="json"), status_code=http_code)


# ─────────────────────────────────────────────────────────────────────────────
# GET /ready  — deep readiness probe (active I/O to all services)
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness probe — deep connectivity check",
    responses={
        200: {"description": "All services reachable — pod is ready"},
        503: {"description": "One or more services unreachable"},
    },
)
async def get_ready() -> JSONResponse:
    t0 = time.monotonic()
    logger.info("[READY] starting deep connectivity probe")

    try:
        # return_exceptions=True so a probe crash doesn't propagate to global handler
        results = await asyncio.gather(
            _probe_redis(),
            _probe_pinecone(),
            _probe_groq(),
            return_exceptions=True,
        )

        now_ms          = (time.monotonic() - t0) * 1000
        redis_health    = results[0] if isinstance(results[0], ServiceHealth) else _make_error_health("redis",    results[0], now_ms)
        pinecone_health = results[1] if isinstance(results[1], ServiceHealth) else _make_error_health("pinecone", results[1], now_ms)
        groq_health     = results[2] if isinstance(results[2], ServiceHealth) else _make_error_health("groq",     results[2], now_ms)

    except Exception as exc:
        logger.error(f"[READY] gather failed: {exc}\n{traceback.format_exc()}")
        now_ms          = (time.monotonic() - t0) * 1000
        redis_health    = _make_error_health("redis",    exc, now_ms)
        pinecone_health = _make_error_health("pinecone", exc, now_ms)
        groq_health     = _make_error_health("groq",     exc, now_ms)

    services: Dict[str, ServiceHealth] = {
        "redis":    redis_health,
        "pinecone": pinecone_health,
        "groq":     groq_health,
    }
    overall   = _overall_status(services)
    http_code = status.HTTP_200_OK if overall == _STATUS_OK else 503

    # Corpus stats — only when Pinecone probe succeeded
    corpus_stats: Optional[Dict] = None
    if pinecone_health.status == _STATUS_OK:
        try:
            from backend.retrieval.pinecone_client import pinecone_client  # noqa: PLC0415
            raw_stats    = await pinecone_client.get_index_stats()
            corpus_stats = {
                "total_vectors":  raw_stats.get("total_vector_count", 0),
                "index_name":     settings.pinecone_index,
                "index_fullness": raw_stats.get("index_fullness", 0),
                "namespaces":     list(raw_stats.get("namespaces", {}).keys()),
            }
        except Exception as exc:
            logger.warning(f"[READY] corpus stats fetch failed: {exc}")

    probe_latency_ms = (time.monotonic() - t0) * 1000

    payload = HealthResponse(
        status         = overall,
        app_name       = settings.app_name,
        version        = settings.app_version,
        environment    = settings.environment,
        uptime_seconds = _uptime_seconds(),
        services       = services,
        corpus_stats   = corpus_stats,
    )

    # Tags must be plain strings — no enum objects
    log_metric(
        "readiness_probe",
        value = probe_latency_ms,
        unit  = "ms",
        tags  = {
            "overall":  str(overall),
            "redis":    str(redis_health.status),
            "pinecone": str(pinecone_health.status),
            "groq":     str(groq_health.status),
        },
    )

    logger.info(
        f"[READY] overall={overall} "
        f"redis={redis_health.status}({redis_health.latency_ms}ms) "
        f"pinecone={pinecone_health.status}({pinecone_health.latency_ms}ms) "
        f"groq={groq_health.status}({groq_health.latency_ms}ms) "
        f"probe_total={probe_latency_ms:.0f}ms"
    )

    return JSONResponse(content=payload.model_dump(mode="json"), status_code=http_code)


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics  — in-memory query stats snapshot (no I/O)
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Operational metrics snapshot",
)
async def get_metrics() -> JSONResponse:
    t0   = time.monotonic()
    snap = orchestrator.get_metrics()

    cache_hit_pct = round(snap["cache_hit_rate"] * 100, 2)
    error_pct     = round(snap["error_rate"]     * 100, 2)

    base = MetricsResponse(
        total_queries            = snap["total_queries"],
        total_thesis_validations = snap["by_query_type"].get("THESIS", 0),
        avg_query_latency_ms     = snap.get("p50_latency_ms", 0.0),
        avg_thesis_latency_ms    = 0.0,
        cache_hit_rate_pct       = cache_hit_pct,
        error_rate_pct           = error_pct,
        uptime_seconds           = _uptime_seconds(),
    )

    extra: Dict[str, Any] = {
        "cache_hits"         : snap["cache_hits"],
        "errors"             : snap["errors"],
        "by_query_type"      : snap["by_query_type"],
        "latency_percentiles": {
            "p50_ms": snap["p50_latency_ms"],
            "p95_ms": snap["p95_latency_ms"],
            "p99_ms": snap["p99_latency_ms"],
        },
        "latency_budget_ms"  : MAX_LATENCY_MS,
        "circuit_breakers"   : {
            "rag_pipeline"   : orchestrator.cb_rag._state.value,
            "thesis_pipeline": orchestrator.cb_thesis._state.value,
            "redis_cache"    : orchestrator.cb_cache._state.value,
        },
        "process_started_utc": _PROCESS_START_UTC,
        "snapshot_utc"       : datetime.now(timezone.utc).isoformat(),
        "snapshot_latency_ms": round((time.monotonic() - t0) * 1000, 2),
    }

    logger.debug(
        f"[METRICS] total={snap['total_queries']} "
        f"cache={cache_hit_pct}% err={error_pct}% "
        f"p50={snap['p50_latency_ms']}ms"
    )
    return JSONResponse(content={**base.model_dump(mode="json"), **extra})


# ─────────────────────────────────────────────────────────────────────────────
# GET /agents/status
# ─────────────────────────────────────────────────────────────────────────────

_agent_registry: Dict[str, Dict[str, Any]] = {
    name: {
        "last_run_utc"          : None,
        "last_duration_ms"      : None,
        "last_success"          : None,
        "success_count"         : 0,
        "failure_count"         : 0,
        "consecutive_failures"  : 0,
        "success_rate_pct"      : 100.0,
        "avg_duration_ms"       : 0.0,
        "latency_target_ms"     : target_ms,
        "within_target_last_run": None,
    }
    for name, target_ms in AGENT_LATENCY_TARGETS.items()
}


def record_agent_run(agent_name: str, duration_ms: float, success: bool) -> None:
    """Called by the @timed decorator after every agent execution."""
    if agent_name not in _agent_registry:
        _agent_registry[agent_name] = {
            "last_run_utc"          : None,
            "last_duration_ms"      : None,
            "last_success"          : None,
            "success_count"         : 0,
            "failure_count"         : 0,
            "consecutive_failures"  : 0,
            "success_rate_pct"      : 100.0,
            "avg_duration_ms"       : 0.0,
            "latency_target_ms"     : AGENT_LATENCY_TARGETS.get(agent_name, MAX_LATENCY_MS),
            "within_target_last_run": None,
        }

    entry    = _agent_registry[agent_name]
    prev_avg = entry["avg_duration_ms"] or duration_ms

    if success:
        entry["success_count"] += 1
        entry["consecutive_failures"] = 0
    else:
        entry["failure_count"] += 1
        entry["consecutive_failures"] += 1

    total_runs = entry["success_count"] + entry["failure_count"]
    entry["last_run_utc"]           = datetime.now(timezone.utc).isoformat()
    entry["last_duration_ms"]       = round(duration_ms, 2)
    entry["last_success"]           = success
    entry["success_rate_pct"]       = round(entry["success_count"] / total_runs * 100, 1)
    entry["avg_duration_ms"]        = round((prev_avg * (total_runs - 1) + duration_ms) / total_runs, 2)
    entry["within_target_last_run"] = duration_ms <= entry["latency_target_ms"]


@router.get("/agents/status", summary="Per-agent health and performance")
async def get_agents_status() -> JSONResponse:
    t0 = time.monotonic()

    agents_out: Dict[str, Any] = {}
    unhealthy_count = 0

    for agent_name, entry in sorted(_agent_registry.items()):
        total_runs = entry["success_count"] + entry["failure_count"]
        is_healthy = (
            entry["consecutive_failures"] < 3
            and entry["success_rate_pct"] >= 80.0
        )
        if not is_healthy:
            unhealthy_count += 1
        agents_out[agent_name] = {
            "status"               : "healthy" if is_healthy else "degraded",
            "last_run_utc"         : entry["last_run_utc"],
            "last_duration_ms"     : entry["last_duration_ms"],
            "last_success"         : entry["last_success"],
            "within_latency_target": entry["within_target_last_run"],
            "latency_target_ms"    : entry["latency_target_ms"],
            "avg_duration_ms"      : entry["avg_duration_ms"],
            "success_rate_pct"     : entry["success_rate_pct"],
            "total_runs"           : total_runs,
            "success_count"        : entry["success_count"],
            "failure_count"        : entry["failure_count"],
            "consecutive_failures" : entry["consecutive_failures"],
        }

    http_code = status.HTTP_200_OK if unhealthy_count == 0 else 207
    return JSONResponse(
        content={
            "status"             : "healthy" if unhealthy_count == 0 else "degraded",
            "total_agents"       : len(agents_out),
            "healthy_agents"     : len(agents_out) - unhealthy_count,
            "degraded_agents"    : unhealthy_count,
            "snapshot_utc"       : datetime.now(timezone.utc).isoformat(),
            "snapshot_latency_ms": round((time.monotonic() - t0) * 1000, 2),
            "agents"             : agents_out,
        },
        status_code=http_code,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Kubernetes-style alias routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health/live", include_in_schema=False)
async def k8s_liveness() -> JSONResponse:
    return JSONResponse(content={
        "status"        : "alive",
        "uptime_seconds": _uptime_seconds(),
        "version"       : settings.app_version,
        "timestamp"     : datetime.now(timezone.utc).isoformat(),
    })


@router.get("/health/ready", include_in_schema=False)
async def k8s_readiness() -> JSONResponse:
    return await get_ready()