# backend/main.py
# ─────────────────────────────────────────────────────────────────────────────
# FinThesisGuard AI — FastAPI Application Entry Point
# ArkAngel Financial Solutions
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response

from backend.config import get_config_summary, settings
from backend.pipeline.orchestrator import orchestrator
from backend.utils.logger import logger, log_startup

from backend.api.routes_health import router as health_router
from backend.api.routes_rag    import router as rag_router
from backend.api.routes_thesis import router as thesis_router

# ─────────────────────────────────────────────────────────────────────────────
# Boot timestamp
# ─────────────────────────────────────────────────────────────────────────────

_BOOT_TIME: float = time.monotonic()
_BOOT_UTC:  str   = datetime.now(timezone.utc).isoformat()

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — Patch Redis socket timeouts before any connection is made.
# Called at the very start of lifespan so the cache singleton picks
# up the new values on its first connect (lazy init).
# ─────────────────────────────────────────────────────────────────────────────

def _patch_redis_timeouts() -> None:
    """
    Increases Redis socket timeouts from their defaults (1–2 s) to 5 s.

    Why: The embedder cold-start blocks the asyncio event loop for ~15 s on
    first boot. This causes Redis connection-pool keepalive pings to time out,
    which produces spurious 'TimeoutError: Timeout connecting to server' on the
    next embed-cache lookup. A 5 s socket timeout survives the warm-up gap.
    """
    try:
        from backend.utils.cache import cache
        # Reach into the underlying redis.asyncio client if it exposes the pool
        client = getattr(cache, '_client', None) or getattr(cache, 'client', None)
        if client is None:
            logger.debug("[STARTUP] Redis client not yet initialised — timeout patch deferred")
            return
        pool = getattr(client, 'connection_pool', None)
        if pool is None:
            return
        conn_kwargs = getattr(pool, 'connection_kwargs', {})
        conn_kwargs['socket_timeout']         = 5.0
        conn_kwargs['socket_connect_timeout'] = 5.0
        logger.info("[STARTUP] Redis socket timeouts patched → 5.0s")
    except Exception as exc:
        logger.debug(f"[STARTUP] Redis timeout patch skipped (non-fatal): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Startup warm-up probes
# ─────────────────────────────────────────────────────────────────────────────

# ── FIX 1 helper — runs synchronously on a thread so it doesn't starve the
#    event loop while the CPU loads ~90 MB of model weights from disk. ─────────
async def _probe_embedding_model() -> str:
    """
    Loads all-MiniLM-L6-v2 from disk and runs one warm-up inference.

    On first boot this takes 10–15 s (model load + ONNX JIT).
    On every subsequent call the model is already in process memory → ~20 ms.

    Runs in a ThreadPoolExecutor so it does NOT block the asyncio event loop,
    which prevents Redis / Pinecone keepalive timeouts during the cold-start.
    """
    t0 = time.monotonic()
    try:
        from backend.retrieval.embedder import embedder

        # Run blocking CPU work off the event loop
        loop = asyncio.get_running_loop()
        vec  = await loop.run_in_executor(
            None,                               # default ThreadPoolExecutor
            lambda: asyncio.run(embedder.encode_query(
                "FinThesisGuard embedding warm-up probe"
            )) if asyncio.iscoroutinefunction(embedder.encode_query)
            else embedder._load_model(),        # fallback: just load weights
        )

        # encode_query may itself be a coroutine — call it properly
        if asyncio.iscoroutinefunction(embedder.encode_query):
            vec = await embedder.encode_query("FinThesisGuard embedding warm-up probe")

        dim = len(vec) if vec is not None else 0
        if dim == 0:
            return "WARN: encode_query returned zero-length vector"
        elapsed = (time.monotonic() - t0) * 1000
        return f"ok dim={dim} ({elapsed:.0f}ms)"
    except Exception as exc:
        return f"WARN: {exc}"


async def _warmup_orchestrator() -> str:
    t0 = time.monotonic()
    try:
        await orchestrator.initialize()
        return f"ok ({(time.monotonic() - t0)*1000:.0f}ms)"
    except Exception as exc:
        logger.warning(f"[STARTUP] Orchestrator warm-up partial (non-fatal): {exc}")
        return f"degraded — {exc}"


async def _probe_pinecone() -> str:
    t0 = time.monotonic()
    try:
        from backend.retrieval.pinecone_client import pinecone_client
        await pinecone_client.initialize_index()
        stats   = await pinecone_client.get_index_stats()
        vectors = stats.get("total_vector_count", 0)
        return (
            f"ok index={settings.pinecone_index} "
            f"vectors={vectors:,} "
            f"({(time.monotonic() - t0)*1000:.0f}ms)"
        )
    except Exception as exc:
        return f"WARN: {exc}"


async def _probe_redis() -> str:
    from backend.utils.cache import cache
    t0 = time.monotonic()
    try:
        await cache.set_cache("startup:probe", {"ping": "pong"}, ttl=10)
        val = await cache.get_cached("startup:probe")
        if val is None:
            return "WARN: write-then-read returned None"
        return f"ok ({(time.monotonic() - t0)*1000:.0f}ms)"
    except Exception as exc:
        return f"WARN: {exc} (cache will bypass gracefully)"


async def _probe_groq() -> str:
    t0 = time.monotonic()
    try:
        from backend.utils.llm_client import llm_client
        reply = await llm_client.call_llm(
            prompt="Reply with exactly one word: ready",
            system_prompt="You are a health check assistant. Reply with one word only.",
            max_tokens=5,
        )
        return (
            f"ok model={settings.groq_model} "
            f"reply={reply!r} "
            f"({(time.monotonic() - t0)*1000:.0f}ms)"
        )
    except Exception as exc:
        return f"WARN: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — Lifespan: Embedder runs FIRST and SEQUENTIALLY before any other step.
#
# Previous order:  Orchestrator → Embedder (concurrent with Pinecone/Redis/Groq)
# New order:       Embedder (sequential, blocks until model is loaded)
#                  → Orchestrator (sequential, now embedder is already warm)
#                  → Pinecone + Redis + Groq (concurrent, fast)
#
# Why this matters:
#   Uvicorn does not begin accepting HTTP requests until lifespan() yields.
#   By loading the model here we guarantee zero cold-start latency on request 1.
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    t_boot = time.monotonic()

    logger.info("=" * 70)
    logger.info(f"  {settings.app_name}  v{settings.app_version}")
    logger.info(f"  Environment : {settings.environment.upper()}")
    logger.info(f"  Host        : {settings.app_host}:{settings.app_port}")
    logger.info(f"  Boot UTC    : {_BOOT_UTC}")
    logger.info("=" * 70)

    try:
        config_summary = get_config_summary()
        log_startup(config_summary)
    except Exception as exc:
        logger.warning(f"[STARTUP] Config summary failed: {exc}")

    # ── FIX 2: patch Redis timeouts before the cache is first used ────────────
    _patch_redis_timeouts()

    # ── Step 1: Embedder — MUST complete before uvicorn accepts any request ───
    # This is the cold-start bottleneck: ~3.6 s model load + ~10 s first encode.
    # Runs in a thread so the event loop remains responsive during disk I/O.
    logger.info(
        "[STARTUP] Step 1/5 — Embedding model warm-up "
        "(may take 15 s on first boot; model stays in memory after) …"
    )
    embed_result = await _probe_embedding_model()
    logger.info(f"[STARTUP] Embedder : {embed_result}")

    # ── Step 2: Orchestrator (Redis verify + pipeline graph compile) ──────────
    # Now that the embedder is warm, orchestrator.initialize() won't trigger
    # another cold-start if it internally calls encode_query.
    logger.info("[STARTUP] Step 2/5 — Orchestrator warm-up …")
    orch_result = await _warmup_orchestrator()
    logger.info(f"[STARTUP] Orchestrator : {orch_result}")

    # ── Steps 3–5: Remaining probes are I/O-bound — run concurrently ──────────
    logger.info("[STARTUP] Step 3-5/5 — Pinecone / Redis / Groq probes …")
    pinecone_result, redis_result, groq_result = await asyncio.gather(
        _probe_pinecone(),
        _probe_redis(),
        _probe_groq(),
    )
    logger.info(f"[STARTUP] Pinecone : {pinecone_result}")
    logger.info(f"[STARTUP] Redis    : {redis_result}")
    logger.info(f"[STARTUP] Groq     : {groq_result}")

    # ── Startup summary ───────────────────────────────────────────────────────
    total_boot_ms = (time.monotonic() - t_boot) * 1000
    all_ok = all(
        r.startswith("ok")
        for r in [embed_result, pinecone_result, redis_result, groq_result]
    )

    if all_ok:
        logger.info("=" * 70)
        logger.info(
            f"  ✓ {settings.app_name} ready in {total_boot_ms:.0f}ms"
            f"  [{settings.environment.upper()}]"
        )
        logger.info("=" * 70)
    else:
        logger.warning("=" * 70)
        logger.warning(
            f"  ⚠  {settings.app_name} started with WARNINGS in "
            f"{total_boot_ms:.0f}ms — some services degraded"
        )
        logger.warning("=" * 70)

    # ── Uvicorn starts accepting requests here ────────────────────────────────
    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    t_shutdown = time.monotonic()
    logger.info("[SHUTDOWN] Draining connections …")
    try:
        await orchestrator.shutdown()
        logger.info(
            f"[SHUTDOWN] Orchestrator closed in "
            f"{(time.monotonic() - t_shutdown)*1000:.0f}ms"
        )
    except Exception as exc:
        logger.warning(f"[SHUTDOWN] Orchestrator shutdown error (ignored): {exc}")

    logger.info(
        f"[SHUTDOWN] {settings.app_name} stopped cleanly. "
        f"Total uptime: {(time.monotonic() - _BOOT_TIME):.1f}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application instance
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description=(
        "**FinThesisGuard AI** — Production-grade Indian financial intelligence engine.\n\n"
        "### Key capabilities\n"
        "- **Financial Q&A** — RAG over annual reports, RBI/SEBI circulars, earnings transcripts\n"
        "- **Conflict Detection** — Agent 4 flags contradictions across sources\n"
        "- **Quant Validation** — Agent 5 cross-checks NIM, ROE, NPA formulas\n"
        "- **Thesis Validation** — Agents 6+7 stress-test investment theses with risk scoring\n\n"
        "### Rate limits\n"
        "| Endpoint | Limit |\n"
        "|---|---|\n"
        "| `/api/query` | 60 req/min |\n"
        "| `/api/query/batch` | 10 req/min |\n"
        "| `/api/validate-thesis` | 20 req/min |\n"
        "| `/api/thesis/compare` | 5 req/min |\n\n"
        "### Response headers\n"
        "Every response includes `X-Request-Id`, `X-Response-Time`, `X-Cache`, `X-Confidence`."
    ),
    version=settings.app_version,
    docs_url="/docs"          if not settings.is_production else None,
    redoc_url="/redoc"        if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
    lifespan=lifespan,
    contact={
        "name":  "ArkAngel Financial Solutions",
        "email": "support@arkangelfinancial.com",
    },
    license_info={"name": "Proprietary — ArkAngel Financial Solutions"},
)


# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────

if settings.is_production:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type", "Authorization",
        "X-Request-Id", "X-Response-Time",
        "X-Cache", "X-Confidence", "X-Thesis-Strength", "X-Query-Type",
    ],
    expose_headers=[
        "X-Request-Id", "X-Response-Time", "X-Cache",
        "X-Confidence", "X-Thesis-Strength", "X-Query-Type",
        "X-Citation-Count", "X-Conflict-Count", "Retry-After",
    ],
    max_age=600,
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]         = "DENY"
        response.headers["X-XSS-Protection"]        = "1; mode=block"
        response.headers["Referrer-Policy"]          = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"]       = "geolocation=(), microphone=()"
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        return response

app.add_middleware(SecurityHeadersMiddleware)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = (
            request.headers.get("X-Request-Id")
            or request.headers.get("X-Correlation-Id")
            or str(uuid.uuid4())[:12]
        )
        request.state.request_id = request_id
        request.state.start_time = time.monotonic()

        forwarded = request.headers.get("X-Forwarded-For")
        client_ip = forwarded.split(",")[0].strip() if forwarded else (
            request.client.host if request.client else "unknown"
        )

        logger.info(
            f"[REQUEST] {request_id} "
            f"{request.method} {request.url.path} "
            f"ip={client_ip} "
            f"ua={request.headers.get('User-Agent', '')[:60]!r}"
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = round((time.monotonic() - request.state.start_time) * 1000)
            logger.error(
                f"[REQUEST] {request_id} UNHANDLED {type(exc).__name__}: {exc} "
                f"after {latency_ms}ms"
            )
            raise

        latency_ms = round((time.monotonic() - request.state.start_time) * 1000)
        response.headers["X-Request-Id"]    = request_id
        response.headers["X-Response-Time"] = f"{latency_ms}ms"

        path         = request.url.path
        is_streaming = path.endswith("/query") and "stream" in str(request.url.query)
        if latency_ms > 5_000 and not is_streaming:
            logger.warning(f"[SLOW RESPONSE] {request_id} {path} took {latency_ms}ms")

        logger.info(
            f"[RESPONSE] {request_id} "
            f"status={response.status_code} "
            f"latency={latency_ms}ms "
            f"cache={response.headers.get('X-Cache', '-')}"
        )
        return response

app.add_middleware(RequestTracingMiddleware)


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int = 512 * 1024):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error":   "payload_too_large",
                    "message": f"Request body exceeds {self.max_bytes // 1024} KB.",
                    "max_kb":  self.max_bytes // 1024,
                },
            )
        return await call_next(request)

app.add_middleware(MaxBodySizeMiddleware, max_bytes=512 * 1024)


# ─────────────────────────────────────────────────────────────────────────────
# Router registration
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(health_router, prefix="")
app.include_router(rag_router,    prefix="")
app.include_router(thesis_router, prefix="")


# ─────────────────────────────────────────────────────────────────────────────
# Global exception handlers
# ─────────────────────────────────────────────────────────────────────────────

def _error_envelope(
    error: str, message: str, status_code: int,
    request_id: str | None = None, **extra: Any,
) -> JSONResponse:
    body: Dict[str, Any] = {
        "error":       error,
        "message":     message,
        "status_code": status_code,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }
    if request_id:
        body["request_id"] = request_id
    body.update(extra)
    return JSONResponse(content=body, status_code=status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError,
) -> JSONResponse:
    request_id  = getattr(request.state, "request_id", None)
    field_errors = [
        {
            "field":   " → ".join(str(x) for x in err.get("loc", [])),
            "message": err.get("msg", "validation error"),
            "type":    err.get("type", ""),
        }
        for err in exc.errors()
    ]
    logger.warning(
        f"[422] {request_id} {request.method} {request.url.path} "
        f"— {len(field_errors)} error(s): "
        f"{field_errors[0]['message'] if field_errors else 'unknown'}"
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error":        "validation_error",
            "message":      "Request body failed validation. Check `field_errors`.",
            "status_code":  422,
            "field_errors": field_errors,
            "request_id":   request_id,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    if isinstance(exc.detail, dict):
        body = exc.detail
        body.setdefault("status_code", exc.status_code)
        body.setdefault("timestamp",   datetime.now(timezone.utc).isoformat())
        if request_id:
            body.setdefault("request_id", request_id)
        response = JSONResponse(content=body, status_code=exc.status_code)
    else:
        response = _error_envelope(
            error="http_error", message=str(exc.detail),
            status_code=exc.status_code, request_id=request_id,
        )
    if exc.status_code == 429 and exc.headers and "Retry-After" in exc.headers:
        response.headers["Retry-After"] = exc.headers["Retry-After"]
    if exc.status_code >= 500:
        logger.error(f"[{exc.status_code}] {request_id} {request.method} {request.url.path} — {exc.detail}")
    elif exc.status_code != 429:
        logger.warning(f"[{exc.status_code}] {request_id} {request.method} {request.url.path} — {exc.detail}")
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.error(
        f"[500 UNHANDLED] {request_id} "
        f"{request.method} {request.url.path}\n{traceback.format_exc()}"
    )
    return _error_envelope(
        error="internal_server_error",
        message=(
            "An unexpected error occurred. Our team has been notified. "
            "Please retry or contact support."
        ),
        status_code=500,
        request_id=request_id,
        support_hint="Reference your request_id when contacting support.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Root + ping
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root(request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(content={
        "app":            settings.app_name,
        "version":        settings.app_version,
        "environment":    settings.environment,
        "status":         "running",
        "uptime_seconds": round(time.monotonic() - _BOOT_TIME, 1),
        "docs_url":       "/docs" if not settings.is_production else "disabled",
        "health_url":     "/health",
        "endpoints": {
            "query":           "POST /api/query",
            "batch_query":     "POST /api/query/batch",
            "validate_thesis": "POST /api/validate-thesis",
            "quick_thesis":    "POST /api/validate-thesis/quick",
            "compare_theses":  "POST /api/thesis/compare",
            "query_history":   "GET  /api/query/history",
            "thesis_history":  "GET  /api/thesis/history",
        },
        "request_id": request_id,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    })


@app.get("/ping", include_in_schema=False)
async def ping() -> JSONResponse:
    return JSONResponse(content={"pong": True})


# ─────────────────────────────────────────────────────────────────────────────
# Uvicorn entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
        access_log=False,
        workers=1,
        loop="asyncio",
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10,
        server_header=False,
        date_header=False,
    )
