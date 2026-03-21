# backend/ingestion/ingest_pipeline.py
"""
FinThesisGuard AI — End-to-End Ingestion Pipeline Runner
ArkAngel Financial Solutions

The single entry point that wires the full ingestion pipeline:

    PDF / URL / Directory
        ↓
    [1] pdf_parser       → raw text + tables
        ↓
    [2]# `ingestion/ingest_pipeline.py` — Production Grade Code

# backend/ingestion/ingest_pipeline.py
"""
from backend import ingestion

"""
FinThesisGuard AI — End-to-End Ingestion Pipeline Runner
ArkAngel Financial Solutions

The single entry point that wires the entire ingestion chain:

    File / URL / Directory
        │
        ▼
    pdf_parser      → raw text + table blocks + page metadata
        │
        ▼
    chunker         → semantic chunks (adaptive size per source_type)
        │
        ▼
    metadata_tagger → rich metadata dict per chunk
        │
        ▼
    embedder        → float32 embedding vectors (batched)
        │
        ▼
    pinecone_client → upsert vectors with metadata

Key design decisions:
    - Idempotent: re-ingesting same file updates existing vectors
      (Pinecone upsert overwrites by vector ID)
    - Parallel file processing: asyncio.Semaphore(4) across a directory
    - Progress bar: tqdm with per-stage sub-steps
    - Per-document stats persisted in Redis (TTL 7 days)
    - Structured failure isolation: one bad file never kills a batch
    - All durations logged per stage for performance profiling
"""

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, AsyncIterator
from urllib.parse import urlparse

import httpx
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from backend.ingestion.pdf_parser      import pdf_parser
from backend.ingestion.chunker         import Chunker
from backend.ingestion.metadata_tagger import metadata_tagger
from backend.retrieval.embedder        import embedder
from backend.retrieval.pinecone_client import pinecone_client
from backend.utils.cache               import cache
from backend.utils.logger              import logger, log_metric
from backend.config                    import settings


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DIR_CONCURRENCY:    int   = 4          # Max parallel files in ingest_directory
EMBED_BATCH_SIZE:   int   = 64         # Chunks per embedder batch call
MAX_URL_SIZE_MB:    float = 15.0       # Max download size for URL ingestion
URL_FETCH_TIMEOUT:  int   = 30         # seconds

# Supported file extensions for directory ingestion
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".txt", ".md", ".csv", ".json",
})

# Redis key prefixes (TTL = 7 days)
REDIS_STATS_KEY:   str = "pipeline:stats:global"
REDIS_DOC_PREFIX:  str = "pipeline:doc:"
REDIS_TTL:         int = 60 * 60 * 24 * 7

# Chunk sizes per source type (adaptive)
CHUNK_SIZE_MAP: dict[str, int] = {
    "annual_report":       1200,
    "earnings_transcript": 1000,
    "rbi":                  800,
    "sebi":                 800,
    "mca":                  800,
    "tax_law":              900,
    "broker_research":      700,
    "news":                 600,
    "blog":                 500,
    "unknown":             1000,
}
CHUNK_OVERLAP_MAP: dict[str, int] = {
    "annual_report":        150,
    "earnings_transcript":  120,
    "rbi":                  100,
    "sebi":                 100,
    "mca":                  100,
    "tax_law":              110,
    "broker_research":       80,
    "news":                  60,
    "blog":                  50,
    "unknown":              100,
}

# User-agent for URL fetching
_URL_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


# ─────────────────────────────────────────────
# RESULT DATACLASSES
# ─────────────────────────────────────────────

@dataclass
class StageResult:
    """Timing and status for a single pipeline stage."""
    stage:      str
    success:    bool
    duration_ms: float = 0.0
    items:      int    = 0
    error:      Optional[str] = None


@dataclass
class IngestResult:
    """
    Complete result for one ingested document.
    Returned by ingest_file(), ingest_url().
    Collected into list by ingest_directory() and ingest_batch().
    """
    success:          bool
    document_id:      str
    source:           str            = ""
    source_type:      str            = "unknown"
    file_path:        Optional[str]  = None
    url:              Optional[str]  = None
    chunks_created:   int            = 0
    vectors_upserted: int            = 0
    total_tokens:     int            = 0
    duration_ms:      float          = 0.0
    error:            Optional[str]  = None
    stages:           list[StageResult] = field(default_factory=list)
    was_duplicate:    bool           = False
    timestamp:        str            = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def summary(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"[{status}] {self.source} | "
            f"chunks={self.chunks_created} | "
            f"vectors={self.vectors_upserted} | "
            f"{self.duration_ms:.0f}ms"
            + (f" | ERROR: {self.error}" if self.error else "")
        )


@dataclass
class PipelineStats:
    """Aggregated stats across all ingested documents."""
    total_docs:       int   = 0
    successful_docs:  int   = 0
    failed_docs:      int   = 0
    duplicate_docs:   int   = 0
    total_chunks:     int   = 0
    total_vectors:    int   = 0
    total_tokens:     int   = 0
    total_duration_ms: float = 0.0
    last_updated:     str   = ""
    failed_sources:   list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def avg_chunks_per_doc(self) -> float:
        return (
            round(self.total_chunks / self.successful_docs, 1)
            if self.successful_docs > 0 else 0.0
        )

    @property
    def avg_duration_ms(self) -> float:
        return (
            round(self.total_duration_ms / self.total_docs, 0)
            if self.total_docs > 0 else 0.0
        )


# ─────────────────────────────────────────────
# PIPELINE STAGE TIMER
# ─────────────────────────────────────────────

class _Stage:
    """
    Context manager for timing and logging a single pipeline stage.

    Usage:
        async with _Stage("parse", pbar) as s:
            result = await pdf_parser.parse(path)
            s.items = len(result.pages)
        # Logs: [PIPELINE] parse | 245ms | 12 pages
    """

    def __init__(
        self,
        name:        str,
        pbar:        Optional[tqdm] = None,
        doc_id:      str            = "",
        raise_error: bool           = True,
    ):
        self.name        = name
        self.pbar        = pbar
        self.doc_id      = doc_id
        self.raise_error = raise_error
        self.items       = 0
        self._start      = 0.0
        self._result:    Optional[StageResult] = None

    async def __aenter__(self) -> "_Stage":
        self._start = time.perf_counter()
        if self.pbar:
            self.pbar.set_postfix_str(f"stage={self.name}", refresh=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration_ms = (time.perf_counter() - self._start) * 1000

        if exc_type is not None:
            error_msg = f"{exc_type.__name__}: {exc_val}"
            self._result = StageResult(
                stage       = self.name,
                success     = False,
                duration_ms = duration_ms,
                error       = error_msg,
            )
            logger.error(
                f"[PIPELINE] ✗ {self.name} | "
                f"doc={self.doc_id[:16]} | "
                f"{duration_ms:.0f}ms | "
                f"error={error_msg}"
            )
            # Suppress exception so caller can collect StageResult
            # but re-raise if configured to do so
            return not self.raise_error

        self._result = StageResult(
            stage       = self.name,
            success     = True,
            duration_ms = duration_ms,
            items       = self.items,
        )
        logger.info(
            f"[PIPELINE] ✓ {self.name} | "
            f"doc={self.doc_id[:16]} | "
            f"{duration_ms:.0f}ms"
            + (f" | items={self.items}" if self.items else "")
        )
        return False   # don't suppress exceptions by default

    @property
    def result(self) -> Optional[StageResult]:
        return self._result


# ─────────────────────────────────────────────
# INGEST PIPELINE CLASS
# ─────────────────────────────────────────────

class IngestPipeline:
    """
    End-to-end ingestion pipeline orchestrator.

    Usage (single file):
        from backend.ingestion.ingest_pipeline import pipeline

        result = await pipeline.ingest_file(
            file_path   = "data/raw/sebi/sebi_circular_20260115.pdf",
            source_type = "sebi",
            metadata    = {"source": "SEBI Circular Jan 2026"},
        )
        print(result.summary())

    Usage (whole directory):
        results = await pipeline.ingest_directory("data/raw/rbi/")

    Usage (URL):
        result = await pipeline.ingest_url(
            url         = "https://sebi.gov.in/circular123.pdf",
            source_type = "sebi",
        )

    Usage (stats):
        stats = pipeline.get_ingestion_stats()
    """

    def __init__(self):
        self._chunker = Chunker()
        self._stats   = PipelineStats()
        self._lock    = asyncio.Lock()     # Protects _stats mutations

    # ─────────────────────────────────────────
    # PUBLIC: ingest_file
    # ─────────────────────────────────────────

    async def ingest_file(
        self,
        file_path:   str,
        source_type: Optional[str]  = None,
        metadata:    Optional[dict] = None,
        show_pbar:   bool           = True,
    ) -> IngestResult:
        """
        Ingests a single PDF or text file through the full pipeline.
        Idempotent: re-ingesting the same file upserts (overwrites) vectors.

        Args:
            file_path:   Absolute or relative path to file
            source_type: Override auto-detected source type
                         ("rbi" / "sebi" / "annual_report" etc.)
            metadata:    Extra metadata merged into every chunk's metadata
            show_pbar:   Show tqdm progress bar (default True)

        Returns:
            IngestResult with success, document_id, chunks_created,
            vectors_upserted, duration_ms, per-stage timings

        Examples:
            result = await pipeline.ingest_file(
                "data/raw/sebi/sebi_circular_jan26.pdf",
                source_type="sebi",
            )
            # result.chunks_created   → 18
            # result.vectors_upserted → 18
            # result.duration_ms      → 3420.0
        """
        path         = Path(file_path)
        metadata     = metadata or {}
        pipeline_start = time.perf_counter()

        # Build document ID from path (stable, reproducible)
        document_id = _build_document_id(file_path)

        # Infer source name for logging
        source_name = metadata.get("source") or path.stem.replace("_", " ").title()

        logger.info(
            f"[PIPELINE] Starting ingest_file | "
            f"file={path.name} | "
            f"doc_id={document_id[:16]}"
        )

        # ── Validate file ──────────────────────────────────────────────────
        if not path.exists():
            return self._fail(
                document_id = document_id,
                source      = source_name,
                source_type = source_type or "unknown",
                file_path   = file_path,
                error       = f"File not found: {file_path}",
            )

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return self._fail(
                document_id = document_id,
                source      = source_name,
                source_type = source_type or "unknown",
                file_path   = file_path,
                error       = f"Unsupported extension: {path.suffix}",
            )

        # ── Dedup check ────────────────────────────────────────────────────
        file_hash = await asyncio.to_thread(_md5_file, file_path)
        dedup_key = f"pipeline:dedup:{file_hash}"
        existing  = await cache.get_cached(dedup_key)

        if existing:
            logger.info(
                f"[PIPELINE] Duplicate detected | "
                f"doc_id={document_id[:16]} | "
                f"hash={file_hash[:12]} — re-ingesting to update vectors"
            )
            # We do NOT skip — we re-upsert (idempotent update)
            # But we flag it so caller knows

        # Progress bar: 5 stages
        pbar_desc   = f"{path.name[:35]:<35}"
        stage_names = ["parse", "chunk", "tag", "embed", "upsert"]
        pbar = tqdm(
            total       = 5,
            desc        = pbar_desc,
            unit        = "stage",
            leave       = False,
            disable     = not show_pbar,
            bar_format  = (
                "{desc} |{bar:20}| {n_fmt}/{total_fmt} stages "
                "[{elapsed}<{remaining}] {postfix}"
            ),
        )

        stages:       list[StageResult] = []
        raw_text:     str               = ""
        chunks:       list[dict]        = []
        tagged_chunks: list[dict]       = []
        embeddings:   list[list[float]] = []
        vectors_upserted: int           = 0
        total_tokens: int               = 0

        # ── Detect source type ─────────────────────────────────────────────
        detected_source_type = (
            source_type
            or metadata.get("source_type")
            or metadata_tagger._detect_source_type(path.name, "")
            or "unknown"
        )

        # ── Stage 1: PARSE ─────────────────────────────────────────────────
        try:
            async with _Stage("parse", pbar, document_id) as s:
                if path.suffix.lower() == ".json":
                    # News articles saved by crawler as JSON — extract text directly
                    import json as _json
                    with open(str(path), "r", encoding="utf-8") as _f:
                        _article = _json.load(_f)
                    raw_text = "\n\n".join(filter(None, [
                        _article.get("title",   ""),
                        _article.get("content", ""),
                        _article.get("url",     ""),
                        _article.get("date",    ""),
                        _article.get("source",  ""),
                    ]))
                    s.items = 1
                else:
                    parse_result = await asyncio.to_thread(
                        pdf_parser.parse_pdf_with_metadata,
                        str(path),
                    )
                    raw_text = parse_result.text
                    s.items  = parse_result.page_count
                stages.append(s.result)
            pbar.update(1)

            if not raw_text or len(raw_text.strip()) < 50:
                raise ValueError(
                    f"Parser returned empty/minimal text: {len(raw_text)} chars"
                )

        except Exception as e:
            pbar.close()
            return self._fail(
                document_id = document_id,
                source      = source_name,
                source_type = detected_source_type,
                file_path   = file_path,
                stages      = stages,
                error       = f"Parse failed: {e}",
            )

        # ── Stage 2: CHUNK ─────────────────────────────────────────────────
        try:
            chunk_size    = CHUNK_SIZE_MAP.get(detected_source_type, 1000)
            chunk_overlap = CHUNK_OVERLAP_MAP.get(detected_source_type, 100)

            async with _Stage("chunk", pbar, document_id) as s:
                chunks = await asyncio.to_thread(
                    self._chunker.chunk_document,
                    raw_text,
                    {
                        "doc_id":      document_id,
                        "source_type": detected_source_type,
                        "source":      source_name,
                        **metadata,
                    },
                    chunk_size,
                    chunk_overlap,
                )
                # chunk_document() returns list[Chunk] dataclass objects.
                # All downstream helpers (_tag_all_chunks, _build_pinecone_vectors)
                # treat chunks as plain dicts, so convert here once.
                chunks = [c.to_dict() for c in chunks]
                s.items = len(chunks)
                stages.append(s.result)
            pbar.update(1)

            if not chunks:
                raise ValueError("Chunker returned zero chunks")

        except Exception as e:
            pbar.close()
            return self._fail(
                document_id = document_id,
                source      = source_name,
                source_type = detected_source_type,
                file_path   = file_path,
                stages      = stages,
                error       = f"Chunk failed: {e}",
            )

        # ── Stage 3: TAG ───────────────────────────────────────────────────
        try:
            file_info = {
                "filename":    path.name,
                "source":      source_name,
                "source_type": detected_source_type,
                **{k: v for k, v in metadata.items() if k in (
                    "company", "sector", "date", "document_id",
                )},
            }

            async with _Stage("tag", pbar, document_id) as s:
                tagged_chunks = await asyncio.to_thread(
                    _tag_all_chunks,
                    chunks,
                    file_info,
                    document_id,
                )
                s.items = len(tagged_chunks)
                stages.append(s.result)
            pbar.update(1)

        except Exception as e:
            pbar.close()
            return self._fail(
                document_id = document_id,
                source      = source_name,
                source_type = detected_source_type,
                file_path   = file_path,
                stages      = stages,
                error       = f"Tag failed: {e}",
            )

        # ── Stage 4: EMBED ─────────────────────────────────────────────────
        try:
            texts_to_embed = [c["text"] for c in tagged_chunks]

            async with _Stage("embed", pbar, document_id) as s:
                embeddings, total_tokens = await _embed_in_batches(
                    texts       = texts_to_embed,
                    batch_size  = EMBED_BATCH_SIZE,
                    source_type = detected_source_type,
                )
                s.items = len(embeddings)
                stages.append(s.result)
            pbar.update(1)

            if len(embeddings) != len(tagged_chunks):
                raise ValueError(
                    f"Embedding count mismatch: "
                    f"got {len(embeddings)}, expected {len(tagged_chunks)}"
                )

        except Exception as e:
            pbar.close()
            return self._fail(
                document_id = document_id,
                source      = source_name,
                source_type = detected_source_type,
                file_path   = file_path,
                stages      = stages,
                error       = f"Embed failed: {e}",
            )

        # ── Stage 5: UPSERT ────────────────────────────────────────────────
        try:
            vectors = _build_pinecone_vectors(
                tagged_chunks = tagged_chunks,
                embeddings    = embeddings,
                document_id   = document_id,
            )

            async with _Stage("upsert", pbar, document_id) as s:
                # upsert_namespace(vectors, namespace) upserts into the
                # namespace matching source_type and returns the upserted count.
                vectors_upserted = await pinecone_client.upsert_namespace(
                    vectors,
                    detected_source_type,
                )
                s.items = vectors_upserted
                stages.append(s.result)
            pbar.update(1)

        except Exception as e:
            pbar.close()
            return self._fail(
                document_id = document_id,
                source      = source_name,
                source_type = detected_source_type,
                file_path   = file_path,
                stages      = stages,
                error       = f"Upsert failed: {e}",
            )

        pbar.close()

        # ── Post-pipeline ──────────────────────────────────────────────────
        duration_ms = (time.perf_counter() - pipeline_start) * 1000

        # Cache MD5 hash to flag for future dedup logging
        await cache.set_cache(dedup_key, document_id, ttl=REDIS_TTL)

        result = IngestResult(
            success          = True,
            document_id      = document_id,
            source           = source_name,
            source_type      = detected_source_type,
            file_path        = str(path.resolve()),
            chunks_created   = len(chunks),
            vectors_upserted = vectors_upserted,
            total_tokens     = total_tokens,
            duration_ms      = duration_ms,
            stages           = stages,
            was_duplicate    = bool(existing),
        )

        # Persist to Redis
        await _persist_doc_result(result)

        # Update global stats
        async with self._lock:
            self._stats.total_docs       += 1
            self._stats.successful_docs  += 1
            self._stats.total_chunks     += len(chunks)
            self._stats.total_vectors    += vectors_upserted
            self._stats.total_tokens     += total_tokens
            self._stats.total_duration_ms += duration_ms
            self._stats.last_updated     = datetime.now(timezone.utc).isoformat()
            if existing:
                self._stats.duplicate_docs += 1

        logger.info(
            f"[PIPELINE] ✓ Complete | "
            f"doc_id={document_id[:16]} | "
            f"chunks={len(chunks)} | "
            f"vectors={vectors_upserted} | "
            f"tokens={total_tokens} | "
            f"{duration_ms:.0f}ms"
        )
        log_metric("pipeline_ingest_duration_ms", duration_ms, unit="ms")
        log_metric("pipeline_vectors_upserted",   vectors_upserted, unit="count")

        return result

    # ─────────────────────────────────────────
    # PUBLIC: ingest_url
    # ─────────────────────────────────────────

    async def ingest_url(
        self,
        url:         str,
        source_type: Optional[str]  = None,
        metadata:    Optional[dict] = None,
        show_pbar:   bool           = True,
    ) -> IngestResult:
        """
        Downloads a URL (PDF or HTML) and ingests it in one call.

        For PDFs: streams to a temp file → ingest_file()
        For HTML: strips tags → ingest as plain text

        Args:
            url:         Public URL of document
            source_type: Override auto-detected source type
            metadata:    Extra metadata merged into every chunk

        Returns:
            IngestResult (same as ingest_file)

        Examples:
            result = await pipeline.ingest_url(
                url="https://www.sebi.gov.in/legal/circulars/mar-2026/abc.pdf",
                source_type="sebi",
            )
        """
        metadata    = metadata or {}
        pipeline_start = time.perf_counter()
        document_id = _build_document_id(url)
        source_name = metadata.get("source") or _url_to_source_name(url)

        logger.info(
            f"[PIPELINE] Starting ingest_url | "
            f"url={url[:80]} | "
            f"doc_id={document_id[:16]}"
        )

        # ── Download ───────────────────────────────────────────────────────
        try:
            file_path, content_type = await _download_url(url)
        except Exception as e:
            return self._fail(
                document_id = document_id,
                source      = source_name,
                source_type = source_type or "unknown",
                url         = url,
                error       = f"URL download failed: {e}",
            )

        # ── Route to file ingestor ─────────────────────────────────────────
        try:
            result = await self.ingest_file(
                file_path   = str(file_path),
                source_type = source_type,
                metadata    = {
                    "source": source_name,
                    "url":    url,
                    **metadata,
                },
                show_pbar   = show_pbar,
            )
            result.url = url
            return result

        finally:
            # Always clean up temp file
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass

    # ─────────────────────────────────────────
    # PUBLIC: ingest_directory
    # ─────────────────────────────────────────

    async def ingest_directory(
        self,
        dir_path:     str,
        source_type:  Optional[str]  = None,
        metadata:     Optional[dict] = None,
        recursive:    bool           = False,
        show_pbar:    bool           = True,
        concurrency:  int            = DIR_CONCURRENCY,
    ) -> list[IngestResult]:
        """
        Ingests all supported files in a directory.
        Processes up to `concurrency` files in parallel.

        Args:
            dir_path:    Path to directory
            source_type: Apply same source_type to all files
                         (None = auto-detect per file)
            metadata:    Base metadata merged into all chunks
            recursive:   If True, also processes subdirectories
            show_pbar:   Master progress bar across all files
            concurrency: Max parallel file ingestions (default 4)

        Returns:
            List[IngestResult], one per file (including failures)

        Examples:
            results = await pipeline.ingest_directory(
                "data/raw/rbi/",
                source_type="rbi",
                concurrency=4,
            )
            successful = [r for r in results if r.success]
            print(f"Ingested {len(successful)}/{len(results)} files")
        """
        dir_path_obj = Path(dir_path)
        if not dir_path_obj.is_dir():
            logger.error(f"[PIPELINE] ingest_directory: not a directory: {dir_path}")
            return []

        # Collect all supported files
        if recursive:
            all_files = [
                p for p in dir_path_obj.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        else:
            all_files = [
                p for p in dir_path_obj.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]

        all_files.sort()   # Deterministic ordering

        if not all_files:
            logger.warning(
                f"[PIPELINE] ingest_directory: no supported files found in {dir_path}"
            )
            return []

        logger.info(
            f"[PIPELINE] ingest_directory | "
            f"dir={dir_path} | "
            f"files={len(all_files)} | "
            f"concurrency={concurrency}"
        )

        # Master progress bar across all files
        master_pbar = tqdm(
            total      = len(all_files),
            desc       = f"{'Ingesting directory':<35}",
            unit       = "file",
            disable    = not show_pbar,
            bar_format = (
                "{desc} |{bar:30}| {n_fmt}/{total_fmt} files "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )

        semaphore = asyncio.Semaphore(concurrency)
        results:  list[IngestResult] = []
        metadata  = metadata or {}

        async def _ingest_one(file_path: Path) -> IngestResult:
            async with semaphore:
                result = await self.ingest_file(
                    file_path   = str(file_path),
                    source_type = source_type,
                    metadata    = metadata,
                    show_pbar   = False,   # suppress per-file pbars in batch
                )
                master_pbar.set_postfix_str(
                    f"last={file_path.name[:20]} "
                    f"{'✓' if result.success else '✗'}",
                    refresh=True,
                )
                master_pbar.update(1)
                return result

        # Run all concurrently (throttled by semaphore)
        tasks   = [_ingest_one(f) for f in all_files]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        master_pbar.close()

        # Summary log
        ok      = sum(1 for r in results if r.success)
        failed  = sum(1 for r in results if not r.success)
        total_v = sum(r.vectors_upserted for r in results)

        logger.info(
            f"[PIPELINE] ingest_directory complete | "
            f"dir={dir_path} | "
            f"ok={ok} | failed={failed} | "
            f"total_vectors={total_v}"
        )

        # Log failures
        for r in results:
            if not r.success:
                logger.error(
                    f"[PIPELINE] Failed: {r.file_path or r.url} | "
                    f"error={r.error}"
                )

        return list(results)

    # ─────────────────────────────────────────
    # PUBLIC: ingest_batch
    # ─────────────────────────────────────────

    async def ingest_batch(
        self,
        sources:     list[dict],
        concurrency: int  = DIR_CONCURRENCY,
        show_pbar:   bool = True,
    ) -> list[IngestResult]:
        """
        Ingests a mixed batch of files and URLs concurrently.

        Args:
            sources: List of source dicts, each with:
                {
                    "type":        "file" | "url",          # required
                    "path":        "/path/to/file.pdf",     # for type=file
                    "url":         "https://...",           # for type=url
                    "source_type": "sebi",                  # optional
                    "metadata":    {"company": "HDFC"},     # optional
                }
            concurrency: Max parallel ingestions
            show_pbar:   Master progress bar

        Returns:
            List[IngestResult] in the same order as sources

        Examples:
            results = await pipeline.ingest_batch([
                {"type": "file", "path": "data/raw/rbi/rbi_circ.pdf",
                 "source_type": "rbi"},
                {"type": "url",
                 "url": "https://sebi.gov.in/circ/2026/abc.pdf",
                 "source_type": "sebi"},
                {"type": "file", "path": "data/raw/bse/hdfc_ar.pdf",
                 "source_type": "annual_report",
                 "metadata": {"company": "HDFC Bank"}},
            ])
        """
        if not sources:
            return []

        logger.info(
            f"[PIPELINE] ingest_batch | "
            f"total={len(sources)} | "
            f"concurrency={concurrency}"
        )

        master_pbar = tqdm(
            total      = len(sources),
            desc       = f"{'Ingesting batch':<35}",
            unit       = "doc",
            disable    = not show_pbar,
            bar_format = (
                "{desc} |{bar:30}| {n_fmt}/{total_fmt} docs "
                "[{elapsed}<{remaining}] {postfix}"
            ),
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def _process_source(src: dict) -> IngestResult:
            async with semaphore:
                src_type   = src.get("type", "file")
                source_type = src.get("source_type")
                metadata   = src.get("metadata") or {}

                if src_type == "url":
                    result = await self.ingest_url(
                        url         = src["url"],
                        source_type = source_type,
                        metadata    = metadata,
                        show_pbar   = False,
                    )
                else:
                    result = await self.ingest_file(
                        file_path   = src["path"],
                        source_type = source_type,
                        metadata    = metadata,
                        show_pbar   = False,
                    )

                master_pbar.set_postfix_str(
                    f"{'✓' if result.success else '✗'} {result.source[:20]}",
                    refresh=True,
                )
                master_pbar.update(1)
                return result

        tasks   = [_process_source(s) for s in sources]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        master_pbar.close()

        ok     = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        logger.info(
            f"[PIPELINE] ingest_batch complete | "
            f"ok={ok} | failed={failed}"
        )

        return list(results)

    # ─────────────────────────────────────────
    # PUBLIC: get_ingestion_stats
    # ─────────────────────────────────────────

    def get_ingestion_stats(self) -> dict:
        """
        Returns aggregated ingestion stats for this session.

        Returns:
            {
                "total_docs":          12,
                "successful_docs":     11,
                "failed_docs":          1,
                "duplicate_docs":       3,
                "total_chunks":       284,
                "total_vectors":      284,
                "total_tokens":    142000,
                "avg_chunks_per_doc":  25.8,
                "avg_duration_ms":   3210.0,
                "total_duration_ms": 35310.0,
                "last_updated":     "2026-03-19T00:30:00+00:00",
                "failed_sources":   ["corrupt_file.pdf"],
            }
        """
        d = self._stats.to_dict()
        d["avg_chunks_per_doc"] = self._stats.avg_chunks_per_doc
        d["avg_duration_ms"]    = self._stats.avg_duration_ms
        return d

    async def get_full_stats(self) -> dict:
        """
        Returns stats merged from both in-memory session stats
        and Redis-persisted historical stats.
        """
        session_stats = self.get_ingestion_stats()

        try:
            redis_stats_raw = await cache.get_cached(REDIS_STATS_KEY)
            if redis_stats_raw:
                redis_stats = (
                    json.loads(redis_stats_raw)
                    if isinstance(redis_stats_raw, str)
                    else redis_stats_raw
                )
                # Merge: session + historical
                return {
                    "session":    session_stats,
                    "historical": redis_stats,
                    "combined_total_docs":    (
                        session_stats["total_docs"]
                        + redis_stats.get("total_docs", 0)
                    ),
                    "combined_total_vectors": (
                        session_stats["total_vectors"]
                        + redis_stats.get("total_vectors", 0)
                    ),
                }
        except Exception:
            pass

        return {"session": session_stats, "historical": {}}

    # ─────────────────────────────────────────
    # PRIVATE: failure helper
    # ─────────────────────────────────────────

    def _fail(
        self,
        document_id: str,
        source:      str,
        source_type: str,
        error:       str,
        file_path:   Optional[str]           = None,
        url:         Optional[str]           = None,
        stages:      list[StageResult]       = None,
    ) -> IngestResult:
        """Builds a failed IngestResult and updates global stats."""
        logger.error(
            f"[PIPELINE] ✗ FAILED | "
            f"doc_id={document_id[:16]} | "
            f"source={source} | "
            f"error={error}"
        )

        result = IngestResult(
            success     = False,
            document_id = document_id,
            source      = source,
            source_type = source_type,
            file_path   = file_path,
            url         = url,
            error       = error,
            stages      = stages or [],
        )

        # Update stats synchronously (called from sync context sometimes)
        self._stats.total_docs    += 1
        self._stats.failed_docs   += 1
        self._stats.last_updated  = datetime.now(timezone.utc).isoformat()
        if source not in self._stats.failed_sources:
            self._stats.failed_sources.append(source)

        return result


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _build_document_id(path_or_url: str) -> str:
    """
    Builds a stable, reproducible document ID from path or URL.
    Format: doc_{md5_first_16}
    """
    return "doc_" + hashlib.md5(
        path_or_url.strip().encode()
    ).hexdigest()[:16]


def _md5_file(file_path: str, chunk_size: int = 65536) -> str:
    """Computes MD5 hash of file contents (runs in thread pool)."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _tag_all_chunks(
    chunks:      list[dict],
    file_info:   dict,
    document_id: str,
) -> list[dict]:
    """
    Tags all chunks with metadata (runs in thread pool via asyncio.to_thread).
    Injects chunk_index and doc_id into each chunk's metadata.
    """
    tagged: list[dict] = []
    for i, chunk in enumerate(chunks):
        chunk_text     = chunk.get("text", "")
        # Exclude "text" and the nested "metadata" sub-dict from Chunk.to_dict()
        # — Pinecone rejects nested objects. The tagger builds its own flat dict.
        # Flatten chunk.metadata into base_metadata directly instead.
        chunk_meta = chunk.get("metadata", {}) or {}
        base_metadata  = {
            **{k: v for k, v in chunk.items() if k not in ("text", "metadata")},
            **{k: v for k, v in chunk_meta.items()},
            "chunk_index": i,
            "doc_id":      document_id,
        }
        meta = metadata_tagger.tag_chunk(
            chunk_text    = chunk_text,
            file_info     = file_info,
            base_metadata = base_metadata,
        )
        tagged.append({
            "text": chunk_text,
            **meta,
        })
    return tagged


async def _embed_in_batches(
    texts:       list[str],
    batch_size:  int,
    source_type: str,
) -> tuple[list[list[float]], int]:
    """
    Embeds texts in batches using embedder.encode_batch().
    Returns (embeddings, approx_total_tokens).

    encode_batch() is synchronous, so each batch runs in a thread
    pool via asyncio.to_thread to keep the event loop unblocked.
    total_tokens is approximated as total whitespace-split word count
    (the MiniLM embedder does not expose a token counter).
    """
    all_embeddings: list[list[float]] = []
    total_tokens:   int               = 0

    for i in range(0, len(texts), batch_size):
        batch        = texts[i: i + batch_size]
        batch_vectors = await asyncio.to_thread(
            embedder.encode_batch,
            batch,
        )
        all_embeddings.extend(batch_vectors)
        # Approximate token count (words ≈ tokens for monitoring purposes)
        total_tokens += sum(len(t.split()) for t in batch)

        logger.debug(
            f"[PIPELINE] Embedded batch "
            f"{i // batch_size + 1}/"
            f"{(len(texts) + batch_size - 1) // batch_size} | "
            f"size={len(batch)} | "
            f"approx_tokens={total_tokens}"
        )

    return all_embeddings, total_tokens


def _build_pinecone_vectors(
    tagged_chunks: list[dict],
    embeddings:    list[list[float]],
    document_id:   str,
) -> list[dict]:
    """
    Builds Pinecone-ready vector dicts:
    [{"id": "...", "values": [...], "metadata": {...}}, ...]

    Vector ID format: {document_id}_chunk_{index:04d}
    This makes IDs stable and idempotent across re-ingestions.
    """
    vectors: list[dict] = []

    for i, (chunk, embedding) in enumerate(zip(tagged_chunks, embeddings)):
        vector_id = f"{document_id}_chunk_{i:04d}"
        text      = chunk.get("text", "")

        # Pinecone only accepts: str, int, float, bool, list[str].
        # Strip any nested dicts and convert invalid list values to strings.
        metadata: dict = {}
        for k, v in chunk.items():
            if k == "text":
                continue
            if isinstance(v, dict):
                # Flatten one level: {"metadata": {"x": 1}} → skip (already merged upstream)
                continue
            if isinstance(v, list):
                # Pinecone lists must contain only strings
                metadata[k] = [str(i) for i in v]
            elif v is None:
                continue  # Pinecone rejects None
            else:
                metadata[k] = v

        # Pinecone metadata value size limit: 1000 chars for text
        if len(text) > 1000:
            metadata["text"] = text[:1000]
        else:
            metadata["text"] = text

        vectors.append({
            "id":       vector_id,
            "values":   embedding,
            "metadata": metadata,
        })

    return vectors


async def _download_url(url: str) -> tuple[Path, str]:
    """
    Downloads a URL to a temp file.
    Returns (temp_file_path, content_type).
    Raises DownloadError on failure.
    """
    import tempfile

    async with httpx.AsyncClient(
        timeout          = httpx.Timeout(URL_FETCH_TIMEOUT),
        follow_redirects = True,
        verify           = False,
        headers          = {"User-Agent": _URL_UA},
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        content      = response.content
        size_mb      = len(content) / (1024 * 1024)

        if size_mb > MAX_URL_SIZE_MB:
            raise ValueError(
                f"URL content too large: {size_mb:.1f}MB > {MAX_URL_SIZE_MB}MB"
            )

        # Choose extension based on content type
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            suffix = ".pdf"
        elif "html" in content_type:
            suffix = ".html"
        else:
            suffix = ".txt"

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            prefix="finthesisguard_",
        )
        tmp.write(content)
        tmp.close()

        logger.debug(
            f"[PIPELINE] URL downloaded | "
            f"url={url[:60]} | "
            f"size={size_mb:.2f}MB | "
            f"type={suffix}"
        )
        return Path(tmp.name), content_type


def _url_to_source_name(url: str) -> str:
    """Converts URL to a human-readable source name."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        slug   = Path(parsed.path).stem.replace("_", " ").replace("-", " ")
        return f"{domain} — {slug.title()}"[:80]
    except Exception:
        return url[:60]


async def _persist_doc_result(result: IngestResult) -> None:
    """Persists IngestResult to Redis for get_full_stats() historical view."""
    try:
        key  = f"pipeline:doc:{result.document_id}"
        data = {
            "document_id":      result.document_id,
            "source":           result.source,
            "source_type":      result.source_type,
            "success":          result.success,
            "chunks_created":   result.chunks_created,
            "vectors_upserted": result.vectors_upserted,
            "duration_ms":      result.duration_ms,
            "timestamp":        result.timestamp,
            "error":            result.error,
        }
        await cache.set_cache(key, data, ttl=REDIS_TTL)
    except Exception as e:
        logger.debug(f"[PIPELINE] Could not persist doc result to Redis: {e}")

import requests, io
from backend.ingestion.pdf_parser import PDFParser

def fetch_sebi_pdf_text(pdf_url: str) -> str:
    """Download and extract text from SEBI PDF circular."""
    resp = requests.get(pdf_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    parser = PDFParser()
    return parser.parse_bytes(io.BytesIO(resp.content))  # adjust to your PDFParser API

# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

pipeline = IngestPipeline()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "pipeline",
    "IngestPipeline",
    "IngestResult",
    "PipelineStats",
    "StageResult",
    "CHUNK_SIZE_MAP",
    "CHUNK_OVERLAP_MAP",
    "SUPPORTED_EXTENSIONS",
]