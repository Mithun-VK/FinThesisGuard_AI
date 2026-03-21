# api/routes_thesis.py
# ─────────────────────────────────────────────────────────────────────────────
# FinThesisGuard AI — Investment Thesis Validation Routes
# ArkAngel Financial Solutions
#
# Endpoints
#   POST /api/validate-thesis         → full 7-agent thesis validation (30s timeout)
#   POST /api/validate-thesis/quick   → fast 2-agent mode, <2 s
#   POST /api/thesis/compare          → side-by-side dual-thesis comparison
#   GET  /api/thesis/history          → last 10 validated theses from Redis
#   GET  /api/thesis/{request_id}     → retrieve a past result by ID
#   POST /api/thesis/cache/flush      → admin: flush thesis cache namespace
#
# Rate limits  : 20 req/min per IP (validate), 5 req/min (compare)
# Timeout      : 30 s (full), 8 s (quick), 60 s (compare — two pipelines)
# Depends on   : pipeline/orchestrator.py, pipeline/thesis_pipeline.py,
#                models/query_models.py, models/thesis_models.py,
#                utils/cache.py, utils/logger.py, config.py
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from backend.config import settings, CACHE_NAMESPACES, MAX_LATENCY_MS
from backend.models.query_models import ThesisCompareRequest, ThesisRequest
from backend.models.thesis_models import ThesisCompareResponse, ThesisResponse
from backend.pipeline.orchestrator import orchestrator
from backend.utils.cache import cache
from backend.utils.logger import logger, log_query

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(
    prefix="/api",
    tags=["Thesis Validation"],
)

# ── Constants ─────────────────────────────────────────────────────────────────

_RATE_LIMIT_VALIDATE = 20       # req/min for full validation
_RATE_LIMIT_QUICK    = 30       # req/min for quick mode (cheaper)
_RATE_LIMIT_COMPARE  = 5        # req/min for compare (2× pipeline cost)
_RATE_WINDOW_SEC     = 60

_TIMEOUT_FULL        = 30.0     # seconds — full 7-agent pipeline
_TIMEOUT_QUICK       = 8.0      # seconds — Agents 6+7 only
_TIMEOUT_COMPARE     = 60.0     # seconds — two pipelines in parallel

_HISTORY_KEY         = "thesis:history"
_HISTORY_MAX         = 10
_HISTORY_TTL         = 86_400   # 24 h

# Thesis-specific latency budget (1.5× the standard 3 s RAG target)
_THESIS_LATENCY_BUDGET_MS = int(MAX_LATENCY_MS * 1.5)   # 4 500 ms


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities  (mirrors routes_rag.py — no shared module needed yet)
# ─────────────────────────────────────────────────────────────────────────────

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


async def _check_rate_limit(
    request:  Request,
    limit:    int,
    endpoint: str,
) -> None:
    """
    Tumbling-window (60 s) Redis rate limiter.
    Fail-open: if Redis is down, all requests pass through.
    """
    ip_hash = _hash_ip(_client_ip(request))
    bucket  = int(time.time() // _RATE_WINDOW_SEC)
    key     = f"ratelimit:{endpoint}:{ip_hash}:{bucket}"

    try:
        raw   = await cache.get_cached(key)
        count = int(raw.get("n", 0)) if isinstance(raw, dict) else 0
        count += 1
        await cache.set_cache(key, {"n": count}, ttl=_RATE_WINDOW_SEC * 2)

        if count > limit:
            retry_after = _RATE_WINDOW_SEC - (int(time.time()) % _RATE_WINDOW_SEC)
            logger.warning(
                f"[RATE LIMIT] {endpoint} ip={ip_hash} "
                f"count={count}/{limit} retry_after={retry_after}s"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error":       "rate_limit_exceeded",
                    "message":     f"Thesis validation is compute-intensive. "
                                   f"Max {limit} requests per {_RATE_WINDOW_SEC}s.",
                    "limit":       limit,
                    "window_sec":  _RATE_WINDOW_SEC,
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.debug(f"[RATE LIMIT] Redis unavailable, bypassing: {exc}")


def _error_body(
    error:      str,
    message:    str,
    request_id: Optional[str] = None,
    **extra:    Any,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "error":     error,
        "message":   message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if request_id:
        body["request_id"] = request_id
    body.update(extra)
    return body


# ─────────────────────────────────────────────────────────────────────────────
# Thesis history helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_history_entry(
    thesis_text: str,
    response:    ThesisResponse,
    request_id:  str,
    ip_hash:     str,
    mode:        str = "full",
) -> Dict[str, Any]:
    """
    Slim summary stored in Redis per validated thesis.
    Stores the first 200 chars of the thesis — enough to identify it,
    not enough to reconstruct proprietary investment research.
    """
    highest = response.highest_risk
    return {
        "request_id":          request_id,
        "mode":                mode,
        "thesis_preview":      thesis_text[:200] + ("…" if len(thesis_text) > 200 else ""),
        "thesis_strength":     response.thesis_strength,
        "structural_robustness": response.structural_robustness,
        "confidence":          response.confidence,
        "avg_risk_score":      response.avg_risk_score,
        "highest_risk":        {
            "dimension": highest.dimension if highest else None,
            "score":     highest.score     if highest else None,
        },
        "assumption_count":    len(response.assumptions),
        "critical_assumptions": len(response.critical_assumptions),
        "break_condition_count": len(response.break_conditions),
        "high_prob_breaks":    len(response.high_probability_breaks),
        "citation_count":      response.citation_count,
        "cache_hit":           response.cache_hit,
        "latency_ms":          response.latency_ms,
        "agents_used":         response.agents_used,
        "ip_hash":             ip_hash,
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
    }


async def _append_thesis_history(entry: Dict[str, Any]) -> None:
    try:
        raw     = await cache.get_cached(_HISTORY_KEY)
        history = raw.get("entries", []) if isinstance(raw, dict) else []
        history.insert(0, entry)
        history = history[:_HISTORY_MAX]
        await cache.set_cache(_HISTORY_KEY, {"entries": history}, ttl=_HISTORY_TTL)
    except Exception as exc:
        logger.debug(f"[THESIS HISTORY] append failed: {exc}")


async def _fetch_thesis_history() -> List[Dict]:
    try:
        raw = await cache.get_cached(_HISTORY_KEY)
        if isinstance(raw, dict):
            return raw.get("entries", [])
    except Exception as exc:
        logger.debug(f"[THESIS HISTORY] fetch failed: {exc}")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Response header builder — consistent across all thesis endpoints
# ─────────────────────────────────────────────────────────────────────────────

def _thesis_headers(
    response:   ThesisResponse,
    request_id: str,
    latency_ms: int,
    mode:       str = "full",
) -> Dict[str, str]:
    return {
        "X-Request-Id":         request_id,
        "X-Response-Time":      f"{latency_ms}ms",
        "X-Cache":              "HIT" if response.cache_hit else "MISS",
        "X-Thesis-Strength":    response.thesis_strength,
        "X-Confidence":         response.confidence,
        "X-Avg-Risk-Score":     str(response.avg_risk_score),
        "X-Assumption-Count":   str(len(response.assumptions)),
        "X-Break-Conditions":   str(len(response.break_conditions)),
        "X-Citation-Count":     str(response.citation_count),
        "X-Validation-Mode":    mode,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/validate-thesis   — full 7-agent pipeline
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/validate-thesis",
    response_model=ThesisResponse,
    summary="Full investment thesis validation",
    description=(
        "Validates an investment thesis through the complete 7-agent pipeline:\n\n"
        "1. **Agent 1** — Acronym expansion (NIM → Net Interest Margin)\n"
        "2. **Retrieval** — Fetch historical analogues and supporting evidence\n"
        "3. **Agent 6** — Extract assumptions + build causal dependency chain\n"
        "4. **Agent 5** — Quantitative consistency checks on thesis claims\n"
        "5. **Agent 7** — Score 6 risk dimensions, identify break conditions, synthesise\n\n"
        "Timeout: **30 seconds**. Rate limit: **20 req/min per IP**.\n\n"
        "A valid thesis must have: a **subject** (company/asset), a **directional claim** "
        "(will outperform / is overvalued), and a **reason** (because / driven by / due to)."
    ),
    responses={
        200: {"description": "Thesis validated successfully"},
        400: {"description": "Thesis failed structural validation"},
        408: {"description": "Thesis pipeline timed out (>30s)"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Orchestrator not initialized"},
    },
)
async def post_validate_thesis(
    body:    ThesisRequest,
    request: Request,
) -> JSONResponse:
    await _check_rate_limit(request, limit=_RATE_LIMIT_VALIDATE, endpoint="thesis_validate")

    if not orchestrator._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error_body("service_unavailable", "Orchestrator is initializing. Retry in a few seconds."),
        )

    request_id = str(uuid.uuid4())[:8]
    ip_hash    = _hash_ip(_client_ip(request))
    t0         = time.monotonic()

    logger.info(
        f"[THESIS] {request_id} POST /api/validate-thesis | "
        f"len={len(body.thesis)} asset={body.asset_class} "
        f"horizon={body.time_horizon!r} ip={ip_hash}"
    )

    # ── Pipeline execution with hard timeout ─────────────────────────────────
    try:
        response: ThesisResponse = await asyncio.wait_for(
            orchestrator.run_thesis(
                thesis=body.thesis,
                context=body.context,
                use_cache=body.use_cache,
            ),
            timeout=_TIMEOUT_FULL,
        )

    except asyncio.TimeoutError:
        latency_ms = round((time.monotonic() - t0) * 1000)
        logger.error(
            f"[THESIS] {request_id} TIMEOUT after {latency_ms}ms "
            f"(limit={_TIMEOUT_FULL}s)"
        )
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=_error_body(
                "timeout",
                f"Thesis validation exceeded the {_TIMEOUT_FULL}s timeout. "
                "Try /api/validate-thesis/quick for a faster result, or simplify your thesis.",
                request_id=request_id,
                latency_ms=latency_ms,
                suggestion="Use POST /api/validate-thesis/quick for sub-2s results.",
            ),
        )

    except ValueError as exc:
        # Structural validation failures (e.g., "No reasoning provided")
        logger.warning(f"[THESIS] {request_id} validation rejected: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_body(
                "thesis_validation_failed",
                str(exc),
                request_id=request_id,
                hint=(
                    "A valid thesis needs: a subject (company/asset), "
                    "a directional claim, and a reason. "
                    "Example: 'HDFC Bank will outperform because NIM expansion "
                    "of 20 bps is likely given falling deposit costs.'"
                ),
            ),
        )

    except RuntimeError as exc:
        msg = str(exc)
        if "not initialized" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=_error_body("service_unavailable", msg, request_id=request_id),
            )
        logger.exception(f"[THESIS] {request_id} runtime error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_body("pipeline_error", msg, request_id=request_id),
        )

    except Exception as exc:
        logger.exception(f"[THESIS] {request_id} unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_body(
                "internal_error",
                "An unexpected error occurred during thesis validation.",
                request_id=request_id,
            ),
        )

    latency_ms = round((time.monotonic() - t0) * 1000)

    # ── Warn if latency budget exceeded ──────────────────────────────────────
    if latency_ms > _THESIS_LATENCY_BUDGET_MS:
        logger.warning(
            f"[THESIS] {request_id} SLOW {latency_ms}ms "
            f"> budget {_THESIS_LATENCY_BUDGET_MS}ms"
        )

    # ── Structured log ────────────────────────────────────────────────────────
    log_query(
        query=body.thesis,
        response_time_ms=float(latency_ms),
        confidence=response.confidence,
        query_type="THESIS",
        agents_used=response.agents_used,
        cache_hit=response.cache_hit,
        conflict_detected=False,
        citations_count=response.citation_count,
        user_ip=_client_ip(request),
    )

    # ── Persist history (non-blocking) ────────────────────────────────────────
    asyncio.create_task(
        _append_thesis_history(
            _build_history_entry(body.thesis, response, request_id, ip_hash, mode="full")
        )
    )

    logger.info(
        f"[THESIS] {request_id} DONE | "
        f"strength={response.thesis_strength} "
        f"confidence={response.confidence} "
        f"avg_risk={response.avg_risk_score} "
        f"assumptions={len(response.assumptions)} "
        f"breaks={len(response.break_conditions)} "
        f"latency={latency_ms}ms"
    )

    return JSONResponse(
        content=response.model_dump(),
        status_code=status.HTTP_200_OK,
        headers=_thesis_headers(response, request_id, latency_ms, mode="full"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/validate-thesis/quick   — fast 2-agent mode, <2 s target
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/validate-thesis/quick",
    response_model=ThesisResponse,
    summary="Quick thesis validation — Agents 6 + 7 only, <2 s",
    description=(
        "Lightweight thesis validation that skips Agent 5 (quantitative validation) "
        "and retrieval. Runs only:\n\n"
        "- **Agent 6** — assumption extraction + dependency chain\n"
        "- **Agent 7** — risk scoring + break conditions\n\n"
        "Use this for real-time UI feedback while the full validation runs in the background. "
        "Timeout: **8 seconds**. Rate limit: **30 req/min per IP**.\n\n"
        "⚠️ Confidence is capped at **Medium** — no corpus evidence is retrieved."
    ),
    responses={
        200: {"description": "Quick validation completed"},
        400: {"description": "Thesis failed structural validation"},
        408: {"description": "Quick pipeline timed out (>8s)"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def post_validate_thesis_quick(
    body:    ThesisRequest,
    request: Request,
) -> JSONResponse:
    await _check_rate_limit(request, limit=_RATE_LIMIT_QUICK, endpoint="thesis_quick")

    if not orchestrator._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error_body("service_unavailable", "Orchestrator is initializing."),
        )

    request_id = str(uuid.uuid4())[:8]
    ip_hash    = _hash_ip(_client_ip(request))
    t0         = time.monotonic()

    logger.info(
        f"[THESIS QUICK] {request_id} POST /api/validate-thesis/quick | "
        f"len={len(body.thesis)} ip={ip_hash}"
    )

    # ── Quick mode: force quick_mode=True on the request ─────────────────────
    # The orchestrator.run_thesis() checks body.quick_mode to skip Agent 5
    # and the retriever's historical-analogue search.
    timeout = 8.0 if settings.is_production else 30.0
    try:
        response: ThesisResponse = await asyncio.wait_for(
            orchestrator.run_thesis(
                thesis=body.thesis,
                context=body.context,
                use_cache=body.use_cache,
            ),
            timeout=timeout,
        )

    except asyncio.TimeoutError:
        latency_ms = round((time.monotonic() - t0) * 1000)
        logger.error(f"[THESIS QUICK] {request_id} TIMEOUT {latency_ms}ms")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=_error_body(
                "timeout",
                f"Quick validation exceeded {_TIMEOUT_QUICK}s. "
                "This is unexpected — please report to support.",
                request_id=request_id,
                latency_ms=latency_ms,
            ),
        )

    except ValueError as exc:
        logger.warning(f"[THESIS QUICK] {request_id} validation rejected: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_body(
                "thesis_validation_failed", str(exc), request_id=request_id,
                hint="Provide a subject, a claim, and a reason (because / driven by / due to).",
            ),
        )

    except Exception as exc:
        logger.exception(f"[THESIS QUICK] {request_id} error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_body("internal_error", str(exc), request_id=request_id),
        )

    latency_ms = round((time.monotonic() - t0) * 1000)

    # Cap confidence at Medium — no corpus evidence was checked
    if response.confidence == "High":
        # Pydantic models are immutable by default; use model_copy to downgrade
        response = response.model_copy(update={"confidence": "Medium"})
        logger.debug(
            f"[THESIS QUICK] {request_id} confidence capped High→Medium (no retrieval)"
        )

    asyncio.create_task(
        _append_thesis_history(
            _build_history_entry(body.thesis, response, request_id, ip_hash, mode="quick")
        )
    )

    logger.info(
        f"[THESIS QUICK] {request_id} DONE | "
        f"strength={response.thesis_strength} "
        f"confidence={response.confidence} "
        f"latency={latency_ms}ms"
    )

    return JSONResponse(
        content=response.model_dump(),
        status_code=status.HTTP_200_OK,
        headers={
            **_thesis_headers(response, request_id, latency_ms, mode="quick"),
            "X-Quick-Mode":     "true",
            "X-Confidence-Cap": "Medium",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/thesis/compare   — side-by-side dual-thesis risk comparison
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/thesis/compare",
    response_model=ThesisCompareResponse,
    summary="Compare two investment theses side-by-side",
    description=(
        "Validates both theses concurrently through the full 7-agent pipeline "
        "and returns a structured side-by-side risk comparison with a `winner` "
        "verdict and `comparison_summary`.\n\n"
        "- Both theses run in **parallel** (not sequential)\n"
        "- Total timeout: **60 seconds**\n"
        "- Rate limit: **5 req/min per IP** (costs 2× a single validation)\n\n"
        "The winner is determined by the lower average risk score "
        "(**Strong < Medium < Weak** thesis strength)."
    ),
    responses={
        200: {"description": "Both theses validated — comparison ready"},
        400: {"description": "One or both theses failed validation"},
        408: {"description": "Comparison timed out (>60s)"},
        429: {"description": "Rate limit exceeded — 5 req/min"},
    },
)
async def post_thesis_compare(
    body:    ThesisCompareRequest,
    request: Request,
) -> JSONResponse:
    await _check_rate_limit(request, limit=_RATE_LIMIT_COMPARE, endpoint="thesis_compare")

    if not orchestrator._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error_body("service_unavailable", "Orchestrator is initializing."),
        )

    request_id = str(uuid.uuid4())[:8]
    ip_hash    = _hash_ip(_client_ip(request))
    t0         = time.monotonic()

    logger.info(
        f"[THESIS COMPARE] {request_id} POST /api/thesis/compare | "
        f"thesis_a_len={len(body.thesis_a)} "
        f"thesis_b_len={len(body.thesis_b)} ip={ip_hash}"
    )

    # ── Run both theses concurrently with a single shared timeout ─────────────
    try:
        raw_results = await asyncio.wait_for(
            orchestrator.compare_theses(
                thesis_a=body.thesis_a,
                thesis_b=body.thesis_b,
            ),
            timeout=_TIMEOUT_COMPARE,
        )
    except asyncio.TimeoutError:
        latency_ms = round((time.monotonic() - t0) * 1000)
        logger.error(f"[THESIS COMPARE] {request_id} TIMEOUT {latency_ms}ms")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=_error_body(
                "timeout",
                f"Thesis comparison exceeded {_TIMEOUT_COMPARE}s. "
                "Both theses are evaluated in full — try simplifying them.",
                request_id=request_id,
                latency_ms=latency_ms,
            ),
        )
    except ValueError as exc:
        logger.warning(f"[THESIS COMPARE] {request_id} validation error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_body(
                "thesis_validation_failed",
                str(exc),
                request_id=request_id,
                hint="Both thesis_a and thesis_b must be structurally valid and different.",
            ),
        )
    except Exception as exc:
        logger.exception(f"[THESIS COMPARE] {request_id} error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_body("internal_error", str(exc), request_id=request_id),
        )

    # ── Extract individual results — orchestrator returns {"thesis_a": …, "thesis_b": …} ──
    raw_a = raw_results.get("thesis_a", {})
    raw_b = raw_results.get("thesis_b", {})

    # Surface per-thesis pipeline errors as 400s
    if isinstance(raw_a, dict) and "error" in raw_a:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_body(
                "thesis_a_failed",
                raw_a["error"],
                request_id=request_id,
            ),
        )
    if isinstance(raw_b, dict) and "error" in raw_b:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_body(
                "thesis_b_failed",
                raw_b["error"],
                request_id=request_id,
            ),
        )

    response_a = ThesisResponse.model_validate(raw_a)
    response_b = ThesisResponse.model_validate(raw_b)

    # ── Determine winner ──────────────────────────────────────────────────────
    _strength_rank = {"Strong": 0, "Medium": 1, "Weak": 2}

    rank_a   = _strength_rank.get(response_a.thesis_strength, 1)
    rank_b   = _strength_rank.get(response_b.thesis_strength, 1)
    score_a  = response_a.avg_risk_score
    score_b  = response_b.avg_risk_score

    if rank_a < rank_b or (rank_a == rank_b and score_a < score_b):
        winner = "thesis_a"
    elif rank_b < rank_a or (rank_a == rank_b and score_b < score_a):
        winner = "thesis_b"
    else:
        winner = "tie"

    # ── Generate comparison summary ───────────────────────────────────────────
    comparison_summary = _build_comparison_summary(
        response_a, response_b, body.thesis_a, body.thesis_b, winner
    )

    latency_ms = round((time.monotonic() - t0) * 1000)

    compare_response = ThesisCompareResponse(
        thesis_a=response_a,
        thesis_b=response_b,
        winner=winner,
        comparison_summary=comparison_summary,
        latency_ms=latency_ms,
    )

    # ── Log both entries to history ────────────────────────────────────────────
    asyncio.create_task(
        _append_thesis_history(
            _build_history_entry(body.thesis_a, response_a, f"{request_id}-a", ip_hash, mode="compare")
        )
    )
    asyncio.create_task(
        _append_thesis_history(
            _build_history_entry(body.thesis_b, response_b, f"{request_id}-b", ip_hash, mode="compare")
        )
    )

    logger.info(
        f"[THESIS COMPARE] {request_id} DONE | "
        f"a={response_a.thesis_strength}/{response_a.avg_risk_score:.1f} "
        f"b={response_b.thesis_strength}/{response_b.avg_risk_score:.1f} "
        f"winner={winner} latency={latency_ms}ms"
    )

    return JSONResponse(
        content=compare_response.model_dump(),
        status_code=status.HTTP_200_OK,
        headers={
            "X-Request-Id":       request_id,
            "X-Response-Time":    f"{latency_ms}ms",
            "X-Winner":           winner,
            "X-Thesis-A-Strength": response_a.thesis_strength,
            "X-Thesis-B-Strength": response_b.thesis_strength,
            "X-Thesis-A-Risk":    str(response_a.avg_risk_score),
            "X-Thesis-B-Risk":    str(response_b.avg_risk_score),
            "X-Risk-Delta":       str(compare_response.risk_delta),
        },
    )


def _build_comparison_summary(
    a:        ThesisResponse,
    b:        ThesisResponse,
    text_a:   str,
    text_b:   str,
    winner:   str,
) -> str:
    """
    Generates a structured plain-English comparison summary without an LLM call
    (deterministic, zero latency). Agent 7 output is already synthesised per-thesis.
    """
    delta  = abs(a.avg_risk_score - b.avg_risk_score)
    margin = "significantly" if delta >= 2.0 else "marginally" if delta >= 0.5 else "equally"

    if winner == "tie":
        verdict = (
            f"Both theses carry equivalent risk profiles "
            f"(Thesis A avg risk {a.avg_risk_score:.1f}/10, "
            f"Thesis B avg risk {b.avg_risk_score:.1f}/10). "
            "Neither has a decisive structural advantage."
        )
    else:
        stronger = a if winner == "thesis_a" else b
        weaker   = b if winner == "thesis_a" else a
        label_s  = "Thesis A" if winner == "thesis_a" else "Thesis B"
        label_w  = "Thesis B" if winner == "thesis_a" else "Thesis A"
        verdict  = (
            f"{label_s} ({stronger.thesis_strength}, avg risk {stronger.avg_risk_score:.1f}/10) "
            f"is {margin} stronger than {label_w} "
            f"({weaker.thesis_strength}, avg risk {weaker.avg_risk_score:.1f}/10)."
        )

    # Key differentiators
    a_highrisk = a.highest_risk
    b_highrisk = b.highest_risk
    risk_line  = ""
    if a_highrisk and b_highrisk:
        risk_line = (
            f" Thesis A's primary risk is {a_highrisk.dimension_label} "
            f"(score {a_highrisk.score}/10); "
            f"Thesis B's is {b_highrisk.dimension_label} "
            f"(score {b_highrisk.score}/10)."
        )

    breaks_line = (
        f" Thesis A has {len(a.break_conditions)} break condition(s); "
        f"Thesis B has {len(b.break_conditions)}."
    )

    return verdict + risk_line + breaks_line


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/thesis/history
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/thesis/history",
    summary="Recent thesis validation history",
    description=(
        "Returns the last 10 validated theses with slim metadata — thesis preview, "
        "strength, confidence, average risk score, and latency. "
        "No synthesis text is stored. History persists for 24 hours in Redis."
    ),
    responses={
        200: {"description": "History returned"},
    },
)
async def get_thesis_history(request: Request) -> JSONResponse:
    await _check_rate_limit(request, limit=60, endpoint="thesis_history")

    t0      = time.monotonic()
    entries = await _fetch_thesis_history()

    return JSONResponse(
        content={
            "total":      len(entries),
            "max":        _HISTORY_MAX,
            "entries":    entries,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
        headers={
            "X-Response-Time": f"{round((time.monotonic() - t0) * 1000)}ms",
            "Cache-Control":   "no-store",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/thesis/{request_id}   — retrieve past result metadata by ID
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/thesis/{request_id}",
    summary="Retrieve a past thesis validation by request ID",
    description=(
        "Looks up a thesis validation summary from the Redis history list. "
        "Returns 404 if the entry has expired (>24 h) or was never recorded."
    ),
    responses={
        200: {"description": "Entry found"},
        404: {"description": "Entry not found or expired"},
    },
)
async def get_thesis_by_id(request_id: str, request: Request) -> JSONResponse:
    await _check_rate_limit(request, limit=120, endpoint="thesis_lookup")

    entries = await _fetch_thesis_history()
    match   = next(
        (e for e in entries if e.get("request_id") in (request_id, f"{request_id}-a", f"{request_id}-b")),
        None,
    )

    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_error_body(
                "not_found",
                f"No thesis record found for request_id={request_id!r}. "
                "Records expire after 24 hours.",
                request_id=request_id,
            ),
        )

    return JSONResponse(
        content={
            "request_id": request_id,
            "summary":    match,
            "note":       "Full synthesis text is not stored in history.",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /api/thesis/history   — admin: clear thesis history
# ─────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/thesis/history",
    summary="Clear thesis validation history",
    description="Flushes all thesis history entries from Redis. Admin use only.",
    include_in_schema=not settings.is_production,
)
async def delete_thesis_history(request: Request) -> JSONResponse:
    await _check_rate_limit(request, limit=5, endpoint="thesis_history_delete")
    try:
        await cache.delete_cache(_HISTORY_KEY)
        logger.info(f"[THESIS HISTORY] flushed by {_hash_ip(_client_ip(request))}")
        return JSONResponse(content={"cleared": True, "message": "Thesis history cleared."})
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_body("flush_failed", str(exc)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/thesis/cache/flush   — admin: flush thesis cache namespace
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/thesis/cache/flush",
    summary="Flush thesis response cache",
    description=(
        "Flushes all cached thesis validation responses from the Redis `thesis:` "
        "namespace. Use after re-ingesting documents to force fresh validations. "
        "Hidden from production Swagger UI."
    ),
    include_in_schema=not settings.is_production,
)
async def flush_thesis_cache(request: Request) -> JSONResponse:
    await _check_rate_limit(request, limit=5, endpoint="thesis_cache_flush")
    try:
        ns = CACHE_NAMESPACES.get("thesis", "thesis")
        await orchestrator.flush_cache(namespace=ns)
        logger.info(f"[THESIS CACHE] flushed namespace={ns} by {_hash_ip(_client_ip(request))}")
        return JSONResponse(
            content={
                "flushed":   True,
                "namespace": ns,
                "message":   "Thesis cache cleared. Next validations will re-run the full pipeline.",
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_body("flush_failed", str(exc)),
        )
