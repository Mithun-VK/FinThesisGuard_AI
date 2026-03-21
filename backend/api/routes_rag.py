# api/routes_rag.py
# ─────────────────────────────────────────────────────────────────────────────
# FinThesisGuard AI — Financial Q&A (RAG) Routes
# ArkAngel Financial Solutions
#
# Endpoints
#   POST GET /api/query          → single financial Q&A (streaming-capable)
#   POST     /api/query/batch    → up to 10 parallel queries
#   GET      /api/query/history  → last 20 queries from Redis
#
# Rate limits  : 60 req/min per IP (query), 10 req/min (batch)
# Auth         : None (add Bearer token middleware in main.py for production)
# Depends on   : pipeline/orchestrator.py, models/query_models.py,
#                models/response_models.py, utils/cache.py, utils/logger.py
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.config import settings, MAX_LATENCY_MS, CACHE_NAMESPACES
from backend.models.query_models import BatchQueryRequest, QueryRequest
from backend.models.response_models import BatchQueryResponse, QueryResponse
from backend.pipeline.orchestrator import orchestrator
from backend.utils.cache import cache
from backend.utils.logger import logger, log_query

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(
    prefix="/api",
    tags=["Financial Q&A"],
)

# ── Constants ─────────────────────────────────────────────────────────────────

_RATE_LIMIT_QUERY   = 60     # requests per window
_RATE_LIMIT_BATCH   = 10     # batch requests per window
_RATE_WINDOW_SEC    = 60     # sliding window in seconds
_HISTORY_KEY        = "query:history"
_HISTORY_MAX        = 20     # entries kept in Redis list
_HISTORY_TTL        = 86_400 # 24 h — history entries survive Redis restart window
_BATCH_MAX          = 10
_STREAM_CHUNK_DELAY = 0.0    # seconds between SSE heartbeats (0 = as-fast-as-possible)


# ─────────────────────────────────────────────────────────────────────────────
# Rate-limiter — Redis sliding-window counter, graceful bypass when Redis is down
# ─────────────────────────────────────────────────────────────────────────────

def _client_ip(request: Request) -> str:
    """Extract real client IP, honouring X-Forwarded-For from reverse proxies."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _hash_ip(ip: str) -> str:
    """One-way hash for privacy-safe Redis keys."""
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


async def _check_rate_limit(
    request:   Request,
    limit:     int = _RATE_LIMIT_QUERY,
    window:    int = _RATE_WINDOW_SEC,
    endpoint:  str = "query",
) -> None:
    """
    Sliding-window rate limiter backed by Redis INCR + EXPIRE.
    Raises HTTP 429 if limit exceeded.
    Silently bypasses if Redis is unavailable (fail-open policy).
    """
    ip      = _client_ip(request)
    ip_hash = _hash_ip(ip)
    bucket  = int(time.time() // window)          # 60-second tumbling bucket
    key     = f"ratelimit:{endpoint}:{ip_hash}:{bucket}"

    try:
        raw = await cache.get_cached(key)
        count = int(raw.get("n", 0)) if isinstance(raw, dict) else 0
        count += 1
        await cache.set_cache(key, {"n": count}, ttl=window * 2)

        if count > limit:
            retry_after = window - (int(time.time()) % window)
            logger.warning(
                f"[RATE LIMIT] {endpoint} | ip_hash={ip_hash} "
                f"count={count}/{limit} retry_after={retry_after}s"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error":       "rate_limit_exceeded",
                    "message":     f"Too many requests. Maximum {limit} per {window}s.",
                    "limit":       limit,
                    "window_sec":  window,
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
    except HTTPException:
        raise
    except Exception as exc:
        # Redis down → fail open, log and continue
        logger.debug(f"[RATE LIMIT] Redis unavailable, bypassing: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Query-history helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_history_entry(
    query:      str,
    response:   QueryResponse,
    request_id: str,
    ip_hash:    str,
) -> Dict[str, Any]:
    """Slim summary stored in Redis — never stores full answer text (PII / size)."""
    return {
        "request_id":     request_id,
        "query_preview":  query[:120] + ("…" if len(query) > 120 else ""),
        "query_type":     response.query_type,
        "confidence":     response.confidence,
        "cache_hit":      response.cache_hit,
        "citation_count": response.citation_count,
        "conflict_count": response.conflict_count,
        "latency_ms":     response.latency_ms,
        "agents_used":    response.agents_used,
        "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
        "ip_hash":        ip_hash,
    }


async def _append_history(entry: Dict[str, Any]) -> None:
    """
    Prepend entry to the Redis history list, trim to _HISTORY_MAX.
    Uses a simple list stored as a JSON array under a single key.
    Silently skips if Redis is unavailable.
    """
    try:
        raw: Optional[Dict] = await cache.get_cached(_HISTORY_KEY)
        history: List[Dict] = raw.get("entries", []) if isinstance(raw, dict) else []
        history.insert(0, entry)
        history = history[:_HISTORY_MAX]
        await cache.set_cache(_HISTORY_KEY, {"entries": history}, ttl=_HISTORY_TTL)
    except Exception as exc:
        logger.debug(f"[HISTORY] append failed (Redis unavailable): {exc}")


async def _fetch_history() -> List[Dict]:
    """Retrieve query history from Redis. Returns empty list on failure."""
    try:
        raw = await cache.get_cached(_HISTORY_KEY)
        if isinstance(raw, dict):
            return raw.get("entries", [])
    except Exception as exc:
        logger.debug(f"[HISTORY] fetch failed: {exc}")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Streaming helper — SSE (Server-Sent Events) generator
# ─────────────────────────────────────────────────────────────────────────────

async def _stream_query_response(
    query:      str,
    req:        QueryRequest,
    request_id: str,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted events for streaming responses.

    Event types
    ───────────
    event: token       → incremental answer token(s) from the LLM
    event: metadata    → final QueryResponse fields (citations, confidence, …)
    event: error       → structured error if pipeline fails
    event: done        → sentinel, signals stream end

    Client-side usage (JavaScript)
    ─────────────────────────────
        const es = new EventSource('/api/query?stream=true');
        es.addEventListener('token',    e => appendToAnswer(e.data));
        es.addEventListener('metadata', e => renderMetadata(JSON.parse(e.data)));
        es.addEventListener('done',     () => es.close());
    """
    t0 = time.monotonic()

    # ── Heartbeat so the browser doesn't time out before the first token ──
    yield "event: heartbeat\ndata: {}\n\n"

    try:
        # 1. Try streaming from the LLM client directly for low-latency UX
        if settings.enable_streaming:
            # Buffer metadata from a parallel non-streaming pipeline run
            # so we can send citations/confidence after streaming is done.
            pipeline_task = asyncio.create_task(
                orchestrator.run_query(
                    query=query,
                    top_k=req.top_k,
                    use_cache=req.use_cache,
                    filters=req.filters,
                )
            )

            # Stream answer tokens from the LLM client concurrently
            accumulated_tokens: List[str] = []
            try:
                async for token in orchestrator.rag_pipeline.llm_client.call_llm_stream(
                    prompt=query,
                    system_prompt=(
                        "You are FinThesisGuard, a precision financial analysis assistant. "
                        "Answer using only the retrieved sources. Cite inline as [1], [2]."
                    ),
                    temperature=settings.llm_temperature_factual,
                    max_tokens=settings.llm_max_tokens_synthesis,
                ):
                    accumulated_tokens.append(token)
                    payload = json.dumps({"token": token, "request_id": request_id})
                    yield f"event: token\ndata: {payload}\n\n"
                    if _STREAM_CHUNK_DELAY:
                        await asyncio.sleep(_STREAM_CHUNK_DELAY)
            except Exception as stream_exc:
                # Streaming failed mid-way — fall back to awaiting the pipeline task
                logger.warning(
                    f"[STREAM] {request_id} LLM stream broke: {stream_exc} — "
                    "switching to buffered response"
                )

            # Wait for full pipeline (citations, conflicts, confidence)
            response: QueryResponse = await pipeline_task

        else:
            # Streaming disabled in config — run normal pipeline, yield all at once
            response = await orchestrator.run_query(
                query=query,
                top_k=req.top_k,
                use_cache=req.use_cache,
                filters=req.filters,
            )
            payload = json.dumps({"token": response.answer, "request_id": request_id})
            yield f"event: token\ndata: {payload}\n\n"

        # ── Metadata event: citations, confidence, latency, etc. ──────────
        metadata = {
            "request_id":     response.request_id or request_id,
            "confidence":     response.confidence,
            "query_type":     response.query_type,
            "citations":      [c.model_dump() for c in response.citations],
            "conflicts":      [c.model_dump() for c in response.conflicts],
            "data_gaps":      response.data_gaps,
            "agents_used":    response.agents_used,
            "cache_hit":      response.cache_hit,
            "latency_ms":     response.latency_ms,
            "citation_count": response.citation_count,
            "conflict_count": response.conflict_count,
            "sub_queries":    response.sub_queries,
            "within_latency_target": response.within_latency_target,
        }
        yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"

        total_ms = round((time.monotonic() - t0) * 1000)
        yield f"event: done\ndata: {json.dumps({'latency_ms': total_ms})}\n\n"

    except Exception as exc:
        logger.exception(f"[STREAM] {request_id} fatal error: {exc}")
        err_payload = json.dumps({
            "error":      "stream_error",
            "message":    str(exc),
            "request_id": request_id,
        })
        yield f"event: error\ndata: {err_payload}\n\n"
        yield f"event: done\ndata: {json.dumps({'error': True})}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/query
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Financial Q&A — single query",
    description=(
        "Submit a financial question. The pipeline expands acronyms, decomposes "
        "complex queries, retrieves and re-ranks corpus chunks, detects conflicts "
        "across sources, validates quantitative claims, and synthesises a cited "
        "answer. Set `stream=true` for a Server-Sent Events streaming response."
    ),
    responses={
        200: {"description": "Successful query response"},
        400: {"description": "Invalid query — validation failed"},
        429: {"description": "Rate limit exceeded — 60 req/min per IP"},
        500: {"description": "Internal pipeline error"},
        503: {"description": "Orchestrator not initialized"},
    },
)
async def post_query(
    body:    QueryRequest,
    request: Request,
) -> QueryResponse | StreamingResponse:
    # ── Pre-flight checks ────────────────────────────────────────────────────
    await _check_rate_limit(request, limit=_RATE_LIMIT_QUERY, endpoint="query")

    if not orchestrator._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error":   "service_unavailable",
                "message": "Orchestrator is initializing. Retry in a few seconds.",
            },
        )

    request_id = str(uuid.uuid4())[:8]
    ip_hash    = _hash_ip(_client_ip(request))
    t0         = time.monotonic()

    logger.info(
        f"[RAG] {request_id} POST /api/query | "
        f"len={len(body.query)} stream={body.stream} ip={ip_hash}"
    )

    # ── Streaming path ───────────────────────────────────────────────────────
    if body.stream:
        if not settings.enable_streaming:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error":   "streaming_disabled",
                    "message": "Streaming is disabled in this deployment. Use stream=false.",
                },
            )
        return StreamingResponse(
            _stream_query_response(body.query, body, request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control":               "no-cache",
                "X-Accel-Buffering":           "no",   # Disable nginx buffering
                "X-Request-Id":                request_id,
                "Access-Control-Allow-Origin": "*",
            },
        )

    # ── Buffered path ────────────────────────────────────────────────────────
    try:
        response: QueryResponse = await orchestrator.run_query(
            query=body.query,
            top_k=body.top_k,
            use_cache=body.use_cache,
            filters=body.filters,
        )
    except ValueError as exc:
        # Validation errors from the pipeline (e.g., query too short after trim)
        logger.warning(f"[RAG] {request_id} validation error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "validation_error", "message": str(exc)},
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "not initialized" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": "service_unavailable", "message": msg},
            )
        logger.exception(f"[RAG] {request_id} runtime error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "pipeline_error", "message": msg},
        )
    except Exception as exc:
        logger.exception(f"[RAG] {request_id} unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error":      "internal_error",
                "message":    "An unexpected error occurred. Please try again.",
                "request_id": request_id,
            },
        )

    latency_ms = round((time.monotonic() - t0) * 1000)

    # ── Log structured query event ───────────────────────────────────────────
    log_query(
        query=body.query,
        response_time_ms=float(latency_ms),
        confidence=response.confidence,
        query_type=response.query_type,
        agents_used=response.agents_used,
        cache_hit=response.cache_hit,
        conflict_detected=response.has_conflicts,
        citations_count=response.citation_count,
        user_ip=_client_ip(request),
    )

    # ── Persist to history (non-blocking, best-effort) ───────────────────────
    asyncio.create_task(
        _append_history(
            _build_history_entry(body.query, response, request_id, ip_hash)
        )
    )

    # ── Warn if latency budget exceeded ─────────────────────────────────────
    if latency_ms > MAX_LATENCY_MS:
        logger.warning(
            f"[RAG] {request_id} LATENCY BUDGET EXCEEDED "
            f"{latency_ms}ms > {MAX_LATENCY_MS}ms"
        )

    return JSONResponse(
        content=response.model_dump(),
        status_code=status.HTTP_200_OK,
        headers={
            "X-Request-Id":      request_id,
            "X-Response-Time":   f"{latency_ms}ms",
            "X-Cache":           "HIT" if response.cache_hit else "MISS",
            "X-Confidence":      response.confidence,
            "X-Query-Type":      response.query_type,
            "X-Citation-Count":  str(response.citation_count),
            "X-Conflict-Count":  str(response.conflict_count),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/query/batch
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/query/batch",
    response_model=BatchQueryResponse,
    summary="Batch financial Q&A — up to 10 queries",
    description=(
        "Submit up to 10 financial queries in a single request. All queries run "
        "concurrently. Individual query failures are captured and returned as "
        "low-confidence error responses — they do NOT abort the batch. "
        "Rate limit: 10 batch requests per minute per IP."
    ),
    responses={
        200: {"description": "Batch completed — check individual result confidence"},
        400: {"description": "Invalid batch request"},
        429: {"description": "Batch rate limit exceeded — 10 req/min per IP"},
        503: {"description": "Orchestrator not initialized"},
    },
)
async def post_query_batch(
    body:    BatchQueryRequest,
    request: Request,
) -> BatchQueryResponse:
    await _check_rate_limit(request, limit=_RATE_LIMIT_BATCH, endpoint="query_batch")

    if not orchestrator._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "service_unavailable", "message": "Orchestrator initializing."},
        )

    if len(body.queries) > _BATCH_MAX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error":   "batch_too_large",
                "message": f"Maximum {_BATCH_MAX} queries per batch. Got {len(body.queries)}.",
                "max":     _BATCH_MAX,
                "got":     len(body.queries),
            },
        )

    request_id = str(uuid.uuid4())[:8]
    ip_hash    = _hash_ip(_client_ip(request))
    t0         = time.monotonic()
    n          = len(body.queries)

    logger.info(
        f"[BATCH] {request_id} POST /api/query/batch | "
        f"n={n} ip={ip_hash}"
    )

    # ── Run all queries concurrently, capture per-query exceptions ───────────
    async def _run_one(idx: int, query: str) -> QueryResponse:
        try:
            return await orchestrator.run_query(
                query=query,
                top_k=body.top_k,
                use_cache=body.use_cache,
            )
        except Exception as exc:
            logger.warning(f"[BATCH] {request_id} query[{idx}] failed: {exc}")
            return QueryResponse.error_response(
                error_message=str(exc),
                request_id=f"{request_id}-{idx}",
            )

    results: List[QueryResponse] = list(
        await asyncio.gather(*[_run_one(i, q) for i, q in enumerate(body.queries)])
    )

    total_latency_ms = round((time.monotonic() - t0) * 1000)
    successful = sum(1 for r in results if r.confidence != "Low" or r.cache_hit)
    failed     = n - successful

    batch_response = BatchQueryResponse(
        results=results,
        total_queries=n,
        successful=successful,
        failed=failed,
        total_latency_ms=total_latency_ms,
    )

    logger.info(
        f"[BATCH] {request_id} DONE | "
        f"n={n} success={successful} failed={failed} "
        f"avg_latency={batch_response.avg_latency_ms}ms "
        f"total={total_latency_ms}ms"
    )

    return JSONResponse(
        content=batch_response.model_dump(),
        status_code=status.HTTP_200_OK,
        headers={
            "X-Request-Id":    request_id,
            "X-Response-Time": f"{total_latency_ms}ms",
            "X-Batch-Size":    str(n),
            "X-Successful":    str(successful),
            "X-Failed":        str(failed),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/query/history
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/query/history",
    summary="Recent query history",
    description=(
        "Returns the last 20 queries processed by this instance, with slim "
        "metadata (no answer text — only query preview, confidence, latency, "
        "agents used). History resets if Redis is unavailable or the server restarts."
    ),
    responses={
        200: {"description": "History returned successfully"},
        503: {"description": "Orchestrator not initialized"},
    },
)
async def get_query_history(request: Request) -> JSONResponse:
    await _check_rate_limit(request, limit=60, endpoint="history")

    t0 = time.monotonic()
    entries = await _fetch_history()

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
# DELETE /api/query/history  (admin — clears history)
# ─────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/query/history",
    summary="Clear query history",
    description="Flushes the in-Redis query history list. Admin use only.",
    include_in_schema=not settings.is_production,   # hide in production Swagger UI
)
async def delete_query_history(request: Request) -> JSONResponse:
    await _check_rate_limit(request, limit=5, endpoint="history_delete")
    try:
        await cache.delete_cache(_HISTORY_KEY)
        logger.info(f"[HISTORY] flushed by {_hash_ip(_client_ip(request))}")
        return JSONResponse(content={"cleared": True, "message": "Query history cleared."})
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "flush_failed", "message": str(exc)},
        )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/query/{request_id}  — fetch a single past result by ID
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/query/{request_id}",
    summary="Retrieve a specific query result by request ID",
    description=(
        "Looks up a cached query result from Redis by its `request_id`. "
        "Returns 404 if the result has expired or was never cached."
    ),
    responses={
        200: {"description": "Result found"},
        404: {"description": "Result not found or expired"},
    },
)
async def get_query_by_id(request_id: str, request: Request) -> JSONResponse:
    await _check_rate_limit(request, limit=120, endpoint="query_lookup")

    # The orchestrator stores results under the rag: namespace with an MD5 key.
    # We can only look up by request_id via the history entries.
    entries = await _fetch_history()
    match = next((e for e in entries if e.get("request_id") == request_id), None)

    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error":      "not_found",
                "message":    f"No query found for request_id={request_id!r}. "
                              "Results expire after 24 hours.",
                "request_id": request_id,
            },
        )

    return JSONResponse(
        content={
            "request_id": request_id,
            "summary":    match,
            "note":       "Full answer text is not stored in history for privacy.",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/query/cache/flush  — flush RAG cache namespace (dev/admin)
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/query/cache/flush",
    summary="Flush RAG query cache",
    description=(
        "Flushes all cached RAG query responses from Redis. "
        "Useful after re-ingesting documents to force fresh retrievals. "
        "Hidden from production Swagger UI."
    ),
    include_in_schema=not settings.is_production,
)
async def flush_rag_cache(request: Request) -> JSONResponse:
    await _check_rate_limit(request, limit=5, endpoint="cache_flush")

    try:
        await orchestrator.flush_cache(namespace=CACHE_NAMESPACES.get("rag", "rag"))
        logger.info(f"[CACHE FLUSH] RAG namespace flushed by {_hash_ip(_client_ip(request))}")
        return JSONResponse(
            content={
                "flushed":   True,
                "namespace": CACHE_NAMESPACES.get("rag", "rag"),
                "message":   "RAG query cache cleared. Next queries will re-run the full pipeline.",
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "flush_failed", "message": str(exc)},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Exception handlers — registered on router for cleaner error JSON shapes
# ─────────────────────────────────────────────────────────────────────────────

def _error_body(
    error:      str,
    message:    str,
    request_id: Optional[str] = None,
    **extra:    Any,
) -> Dict[str, Any]:
    """Standardised error JSON shape used across all endpoints."""
    body: Dict[str, Any] = {
        "error":     error,
        "message":   message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if request_id:
        body["request_id"] = request_id
    body.update(extra)
    return body
