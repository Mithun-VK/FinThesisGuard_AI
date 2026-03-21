#!/usr/bin/env python3
"""
FinThesisGuard AI — Production Web Ingestion Trigger
ArkAngel Financial Solutions

Orchestrates the full scrape → parse → chunk → embed → upsert pipeline
by pulling LIVE data from the web via the existing Crawler before feeding
it into IngestPipeline.

Scrape sources (all driven by crawler.py):
    1. SEBI  — circulars + advisories       → data/raw/sebi/
    2. RBI   — notifications + circulars
               + press releases             → data/raw/rbi/
    3. BSE   — annual reports for a
               configurable company list    → data/raw/bse/
    4. News  — RSS-based financial news
               (ET, Mint, Moneycontrol …)   → data/raw/news/  (JSON)

Run modes:
    python run_ingestion.py                   # full run (all sources)
    python run_ingestion.py --sources sebi rbi
    python run_ingestion.py --days 7          # news window
    python run_ingestion.py --dry-run         # crawl only, no embed/upsert

Environment variables (override defaults):
    INGEST_START_DATE   YYYY-MM-DD   default: 90 days ago
    INGEST_END_DATE     YYYY-MM-DD   default: today
    INGEST_NEWS_DAYS    int          default: 7
    INGEST_BSE_CODES    comma-sep    default: 500180,532540,500325
                                     (HDFC Bank, TCS, Reliance)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ── Your existing modules ────────────────────────────────────────────────────
from backend.retrieval.pinecone_client import pinecone_client
from backend.ingestion.crawler         import crawler, NEWS_RSS_FEEDS
from backend.ingestion.ingest_pipeline import pipeline, IngestResult, PipelineStats
from backend.utils.logger              import logger


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (all overridable via env vars or CLI flags)
# ─────────────────────────────────────────────────────────────────────────────

def _default_start() -> str:
    """90 days ago as ISO date string."""
    return (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")


def _default_end() -> str:
    """Today as ISO date string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# BSE scrip codes to pull annual reports for
DEFAULT_BSE_CODES: list[str] = os.environ.get(
    "INGEST_BSE_CODES",
    "500180,532540,500325",          # HDFC Bank, TCS, Reliance
).split(",")

# News keywords — tweak to match your RAG domain
DEFAULT_NEWS_KEYWORDS: list[str] = [
    "RBI monetary policy", "SEBI circular", "NRI dividend",
    "NIM", "NPA", "HDFC Bank", "TCS earnings", "Reliance quarterly",
    "repo rate", "NIFTY", "BSE Sensex", "Indian banking",
]

# Data directories (mirror crawler.py constants)
BASE_DATA_DIR = Path("data/raw")
NEWS_DIR      = BASE_DATA_DIR / "news"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1  — WEB CRAWL
# ─────────────────────────────────────────────────────────────────────────────

async def crawl_sebi(start_date: str, end_date: str) -> list[str]:
    """
    Downloads SEBI circulars + advisories for the given date range.
    Returns list of local PDF paths.
    """
    print(f"\n🔍 [SEBI] Crawling circulars & advisories  {start_date} → {end_date}")
    try:
        paths = await crawler.crawl_sebi(
            start_date = start_date,
            end_date   = end_date,
            max_pages  = 15,
            doc_types  = ["circular", "advisory"],
        )
        print(f"   ✅ SEBI: {len(paths)} PDFs downloaded → data/raw/sebi/")
        return paths
    except Exception as exc:
        print(f"   ⚠️  SEBI crawl error (continuing): {exc}")
        logger.warning(f"[RUN] SEBI crawl failed: {exc}")
        return []


async def crawl_rbi(start_date: str, end_date: str) -> list[str]:
    """
    Downloads RBI notifications, circulars, and press releases.
    Returns list of local PDF paths.
    """
    print(f"\n🔍 [RBI]  Crawling notifications, circulars, press releases  "
          f"{start_date} → {end_date}")
    try:
        paths = await crawler.crawl_rbi(
            start_date = start_date,
            end_date   = end_date,
            doc_types  = ["notification", "circular", "press_release"],
        )
        print(f"   ✅ RBI: {len(paths)} PDFs downloaded → data/raw/rbi/")
        return paths
    except Exception as exc:
        print(f"   ⚠️  RBI crawl error (continuing): {exc}")
        logger.warning(f"[RUN] RBI crawl failed: {exc}")
        return []


async def crawl_bse(company_codes: list[str]) -> list[str]:
    """
    Downloads BSE annual reports for each company code.
    Returns list of local PDF paths (aggregated across all companies).
    """
    all_paths: list[str] = []
    for code in company_codes:
        print(f"\n🔍 [BSE]  Crawling annual reports for scrip code {code}")
        try:
            paths = await crawler.crawl_bse_filings(
                company_code = code.strip(),
                filing_types = ["annual_report"],
            )
            print(f"   ✅ BSE {code}: {len(paths)} PDFs downloaded → data/raw/bse/")
            all_paths.extend(paths)
        except Exception as exc:
            print(f"   ⚠️  BSE {code} crawl error (continuing): {exc}")
            logger.warning(f"[RUN] BSE {code} crawl failed: {exc}")
    return all_paths


async def crawl_news(keywords: list[str], days: int) -> list[dict]:
    """
    Fetches matching news articles from all configured RSS feeds.
    Articles are saved as JSON files to data/raw/news/ by crawler.py
    and also returned as dicts for direct ingestion.
    Returns list of article dicts.
    """
    print(f"\n🔍 [NEWS] Scraping RSS feeds  (last {days} days | "
          f"{len(NEWS_RSS_FEEDS)} sources)")
    print(f"   Keywords: {', '.join(keywords[:6])}{'…' if len(keywords) > 6 else ''}")
    try:
        articles = await crawler.crawl_news(
            keywords                = keywords,
            days                    = days,
            max_articles_per_source = 50,
        )
        print(f"   ✅ News: {len(articles)} matching articles saved → data/raw/news/")
        return articles
    except Exception as exc:
        print(f"   ⚠️  News crawl error (continuing): {exc}")
        logger.warning(f"[RUN] News crawl failed: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2  — INGEST CRAWLED FILES INTO PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

async def ingest_pdf_paths(
    paths:       list[str],
    source_type: str,
    dry_run:     bool = False,
) -> list[IngestResult]:
    """
    Runs each downloaded PDF through the full pipeline:
    pdf_parser → chunker → metadata_tagger → embedder → pinecone upsert.

    Args:
        paths:       List of local PDF file paths (from crawler).
        source_type: "sebi" / "rbi" / "annual_report" etc.
        dry_run:     If True, skip embed + upsert (crawl-only mode).

    Returns:
        List of IngestResult for each file.
    """
    if not paths:
        return []

    print(f"\n📥 Ingesting {len(paths)} {source_type.upper()} PDFs …")
    results: list[IngestResult] = []

    for file_path in paths:
        if dry_run:
            print(f"   [DRY-RUN] Would ingest: {Path(file_path).name}")
            continue

        result = await pipeline.ingest_file(
            file_path   = file_path,
            source_type = source_type,
            show_pbar   = True,
        )
        results.append(result)

        status = "✓" if result.success else "✗"
        print(
            f"   [{status}] {Path(file_path).name} | "
            f"chunks={result.chunks_created} | "
            f"vectors={result.vectors_upserted} | "
            f"{result.duration_ms:.0f}ms"
            + (f" | ERR: {result.error}" if result.error else "")
        )

    return results


async def ingest_news_articles(
    articles: list[dict],
    dry_run:  bool = False,
) -> list[IngestResult]:
    """
    Ingests news articles saved as JSON files in data/raw/news/.

    crawler.py persists each article as:
        data/raw/news/news_{source}_{hash}.json

    This function collects those files and feeds them into the pipeline
    as source_type="news".  Articles not yet on disk are skipped
    (they were filtered out by the crawler's keyword/date logic).

    Args:
        articles:  Article dicts returned by crawl_news()  (used to
                   resolve filenames via url_hash).
        dry_run:   If True, skip embed + upsert.
    """
    if not articles:
        return []

    # Resolve the JSON files the crawler already saved
    json_paths: list[str] = []
    for article in articles:
        url_hash = article.get("url_hash", "")
        source   = article.get("source", "unknown")
        if url_hash:
            candidate = NEWS_DIR / f"news_{source}_{url_hash[:12]}.json"
            if candidate.exists():
                json_paths.append(str(candidate))

    # Fallback: scan the news directory for any .json not yet in our list
    if not json_paths:
        json_paths = [str(p) for p in NEWS_DIR.glob("news_*.json")]

    if not json_paths:
        print("   ℹ️  No news JSON files found to ingest.")
        return []

    print(f"\n📥 Ingesting {len(json_paths)} news articles …")
    results: list[IngestResult] = []

    for file_path in json_paths:
        if dry_run:
            print(f"   [DRY-RUN] Would ingest: {Path(file_path).name}")
            continue

        result = await pipeline.ingest_file(
            file_path   = file_path,
            source_type = "news",
            show_pbar   = True,
        )
        results.append(result)

        status = "✓" if result.success else "✗"
        print(
            f"   [{status}] {Path(file_path).name} | "
            f"chunks={result.chunks_created} | "
            f"vectors={result.vectors_upserted} | "
            f"{result.duration_ms:.0f}ms"
            + (f" | ERR: {result.error}" if result.error else "")
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(
    all_results:    list[IngestResult],
    crawl_counts:   dict[str, int],
    total_elapsed:  float,
    dry_run:        bool,
) -> None:
    """Prints a formatted final summary table to stdout."""
    success = [r for r in all_results if r.success]
    failed  = [r for r in all_results if not r.success]
    total_vectors = sum(r.vectors_upserted for r in success)
    total_chunks  = sum(r.chunks_created  for r in success)

    print("\n" + "═" * 60)
    print("  INGESTION SUMMARY")
    print("═" * 60)

    # Crawl counts
    for source, count in crawl_counts.items():
        label = f"  {source.upper()} docs crawled"
        print(f"{label:<35} {count:>6}")

    print("─" * 60)

    if dry_run:
        print("  [DRY-RUN MODE] Embed + Upsert skipped")
    else:
        print(f"  {'Documents ingested':<33} {len(all_results):>6}")
        print(f"  {'  ✓ Successful':<33} {len(success):>6}")
        print(f"  {'  ✗ Failed':<33} {len(failed):>6}")
        print(f"  {'Total chunks created':<33} {total_chunks:>6}")
        print(f"  {'Total vectors upserted':<33} {total_vectors:>6}")

    print(f"  {'Total elapsed':<33} {total_elapsed:>5.1f}s")
    print("═" * 60)

    if failed:
        print("\n  FAILED DOCUMENTS:")
        for r in failed:
            src = r.file_path or r.url or r.document_id
            print(f"    ✗ {Path(src).name if src else r.document_id}: {r.error}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

async def run_web_ingestion(
    sources:     Optional[list[str]] = None,
    start_date:  Optional[str]       = None,
    end_date:    Optional[str]       = None,
    news_days:   int                 = 7,
    bse_codes:   Optional[list[str]] = None,
    dry_run:     bool                = False,
) -> dict:
    """
    Full web → RAG ingestion run.

    Flow:
        1. Initialise Pinecone index
        2. Crawl SEBI / RBI / BSE / News (concurrently where possible)
        3. Feed every downloaded file through IngestPipeline
        4. Print summary + return stats dict

    Args:
        sources:    Subset of ["sebi", "rbi", "bse", "news"].
                    None = all sources.
        start_date: YYYY-MM-DD inclusive start (PDF sources).
        end_date:   YYYY-MM-DD inclusive end.
        news_days:  How many days back to fetch news.
        bse_codes:  BSE scrip codes to pull annual reports for.
        dry_run:    Crawl only — skip embed + upsert.

    Returns:
        Stats dict with keys: total_vectors, total_docs, ...
    """
    t0 = time.perf_counter()

    start_date = start_date or os.environ.get("INGEST_START_DATE") or _default_start()
    end_date   = end_date   or os.environ.get("INGEST_END_DATE")   or _default_end()
    news_days  = int(os.environ.get("INGEST_NEWS_DAYS", news_days))
    bse_codes  = bse_codes  or DEFAULT_BSE_CODES
    sources    = sources    or ["sebi", "rbi", "bse", "news"]

    print("\n" + "═" * 60)
    print("  FinThesisGuard AI — Web Ingestion Pipeline")
    print("═" * 60)
    print(f"  Sources    : {', '.join(sources)}")
    print(f"  Date range : {start_date} → {end_date}  (PDF sources)")
    print(f"  News window: last {news_days} days")
    print(f"  BSE codes  : {', '.join(bse_codes)}")
    print(f"  Mode       : {'DRY-RUN (no upsert)' if dry_run else 'PRODUCTION'}")
    print("═" * 60)

    # ── 0. Initialise Pinecone ──────────────────────────────────────────────
    if not dry_run:
        print("\n⚡ Initialising Pinecone index …")
        await pinecone_client.initialize_index()
        print("   ✅ Pinecone ready")

    # ── 1. CRAWL phase (concurrent across independent sources) ──────────────
    sebi_paths:  list[str]  = []
    rbi_paths:   list[str]  = []
    bse_paths:   list[str]  = []
    news_articles: list[dict] = []

    crawl_tasks = {}

    if "sebi" in sources:
        crawl_tasks["sebi"] = crawl_sebi(start_date, end_date)
    if "rbi" in sources:
        crawl_tasks["rbi"]  = crawl_rbi(start_date, end_date)
    if "bse" in sources:
        crawl_tasks["bse"]  = crawl_bse(bse_codes)
    if "news" in sources:
        crawl_tasks["news"] = crawl_news(DEFAULT_NEWS_KEYWORDS, news_days)

    # Run all crawlers concurrently
    if crawl_tasks:
        print("\n🌐 Starting concurrent web crawl …")
        results_map = dict(
            zip(
                crawl_tasks.keys(),
                await asyncio.gather(*crawl_tasks.values(), return_exceptions=False),
            )
        )
        sebi_paths    = results_map.get("sebi",  [])
        rbi_paths     = results_map.get("rbi",   [])
        bse_paths     = results_map.get("bse",   [])
        news_articles = results_map.get("news",  [])

    crawl_counts = {
        "sebi":  len(sebi_paths),
        "rbi":   len(rbi_paths),
        "bse":   len(bse_paths),
        "news":  len(news_articles),
    }
    total_crawled = sum(crawl_counts.values())
    print(
        f"\n✅ Crawl complete — "
        f"{total_crawled} total documents "
        f"({len(sebi_paths)} SEBI | {len(rbi_paths)} RBI | "
        f"{len(bse_paths)} BSE | {len(news_articles)} news)"
    )

    # ── 2. INGEST phase ─────────────────────────────────────────────────────
    all_ingest_results: list[IngestResult] = []

    # SEBI PDFs
    if sebi_paths:
        results = await ingest_pdf_paths(sebi_paths, source_type="sebi", dry_run=dry_run)
        all_ingest_results.extend(results)

    # RBI PDFs
    if rbi_paths:
        results = await ingest_pdf_paths(rbi_paths, source_type="rbi", dry_run=dry_run)
        all_ingest_results.extend(results)

    # BSE annual reports
    if bse_paths:
        results = await ingest_pdf_paths(bse_paths, source_type="annual_report", dry_run=dry_run)
        all_ingest_results.extend(results)

    # News JSON files
    if news_articles:
        results = await ingest_news_articles(news_articles, dry_run=dry_run)
        all_ingest_results.extend(results)

    # ── 3. Verify + summarise ───────────────────────────────────────────────
    final_stats: dict = {}

    if not dry_run:
        try:
            index_stats = await pinecone_client.get_index_stats()
            final_stats["pinecone_total_vectors"] = index_stats.get("total_vector_count", 0)
            print(
                f"\n🎉 Pinecone index now contains "
                f"{final_stats['pinecone_total_vectors']:,} vectors"
            )
        except Exception as exc:
            logger.warning(f"[RUN] Could not fetch Pinecone stats: {exc}")

    elapsed = time.perf_counter() - t0
    _print_summary(all_ingest_results, crawl_counts, elapsed, dry_run)

    # Build return dict
    pipeline_stats = pipeline.get_ingestion_stats()
    final_stats.update({
        "total_docs_crawled":   total_crawled,
        "crawl_breakdown":      crawl_counts,
        "docs_ingested":        len(all_ingest_results),
        "docs_successful":      sum(1 for r in all_ingest_results if r.success),
        "docs_failed":          sum(1 for r in all_ingest_results if not r.success),
        "total_chunks":         sum(r.chunks_created  for r in all_ingest_results if r.success),
        "total_vectors":        sum(r.vectors_upserted for r in all_ingest_results if r.success),
        "elapsed_seconds":      round(elapsed, 2),
        "pipeline_stats":       pipeline_stats,
        "dry_run":              dry_run,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
    })

    return final_stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FinThesisGuard AI — Web Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (all sources, last 90 days of PDFs, last 7 days of news)
  python run_ingestion.py

  # Only SEBI + news, custom date range
  python run_ingestion.py --sources sebi news --start 2026-01-01 --end 2026-03-21

  # BSE annual reports for specific companies
  python run_ingestion.py --sources bse --bse-codes 500180 532540

  # Crawl only — no embedding, no Pinecone upsert
  python run_ingestion.py --dry-run
        """
    )
    parser.add_argument(
        "--sources", nargs="+",
        choices=["sebi", "rbi", "bse", "news"],
        default=None,
        help="Sources to ingest (default: all)",
    )
    parser.add_argument(
        "--start", dest="start_date", default=None,
        help="Start date YYYY-MM-DD (PDF sources). Default: 90 days ago",
    )
    parser.add_argument(
        "--end", dest="end_date", default=None,
        help="End date YYYY-MM-DD (PDF sources). Default: today",
    )
    parser.add_argument(
        "--days", dest="news_days", type=int, default=7,
        help="News lookback window in days (default: 7)",
    )
    parser.add_argument(
        "--bse-codes", nargs="+", default=None,
        help="BSE scrip codes e.g. 500180 532540 (default: HDFC/TCS/Reliance)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Crawl only — skip embedding and Pinecone upsert",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    stats = asyncio.run(
        run_web_ingestion(
            sources    = args.sources,
            start_date = args.start_date,
            end_date   = args.end_date,
            news_days  = args.news_days,
            bse_codes  = args.bse_codes,
            dry_run    = args.dry_run,
        )
    )

    # Print machine-readable stats to stderr for log aggregators
    print(json.dumps(stats, indent=2, default=str), file=sys.stderr)
    sys.exit(0 if stats.get("docs_failed", 0) == 0 else 1)