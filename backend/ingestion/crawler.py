# backend/ingestion/crawler.py
"""
FinThesisGuard AI — Document Crawler
ArkAngel Financial Solutions

Downloads financial documents from public Indian regulatory and
financial sources for ingestion into the RAG corpus.

Sources:
    - SEBI: sebi.gov.in/circulars, advisories, orders
    - RBI:  rbi.org.in/notifications, circulars, press releases
    - BSE:  bseindia.com annual reports and filings
    - News: ET, Mint, Moneycontrol, Business Standard RSS feeds

Design principles:
    - Rate limiting: 1 req/sec per domain (configurable)
    - User-agent rotation: 6 browser UAs to avoid blocks
    - Retry: 3 attempts with exponential backoff on timeout
    - Duplicate detection: URL MD5 hash checked against local registry
    - Structured filenames: {source}_{company}_{doctype}_{date}.pdf
    - All downloads saved to data/raw/{source_type}/
"""

import asyncio
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urlencode

import httpx

from backend.config import settings
from backend.utils.logger import logger, log_metric


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Download directories
BASE_DATA_DIR:       Path = Path("data/raw")
SEBI_DIR:            Path = BASE_DATA_DIR / "sebi"
RBI_DIR:             Path = BASE_DATA_DIR / "rbi"
BSE_DIR:             Path = BASE_DATA_DIR / "bse"
NEWS_DIR:            Path = BASE_DATA_DIR / "news"
DOWNLOAD_REGISTRY:   Path = BASE_DATA_DIR / ".download_registry.json"

# Rate limiting (seconds between requests per domain)
RATE_LIMIT_SECONDS:  float = 1.2
REQUEST_TIMEOUT:     int   = 30      # seconds
MAX_RETRIES:         int   = 3
RETRY_BACKOFF_BASE:  int   = 2       # exponential: 2^attempt

# File size limits
MAX_PDF_SIZE_MB:     int   = 50
MAX_HTML_SIZE_MB:    int   = 5

# User-agent pool — rotated per request
USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",

    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",

    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) "
    "Gecko/20100101 Firefox/123.0",

    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",

    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",

    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
]

# SEBI endpoints
# NOTE: OtherAction.do?doListing=yes URLs now redirect to the SEBI homepage (dead).
# The working listing page is HomeAction.do?doListingAll=yes&search=Circular.
# Individual circulars live at /legal/circulars/{mon-year}/{slug}_{id}.html
# PDFs are embedded at sebi.gov.in/sebi_data/commondocs/{mon-year}/{slug}_p.pdf
SEBI_BASE_URL:              str  = "https://www.sebi.gov.in"
SEBI_CIRCULARS_LIST_URL:    str  = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListingAll=yes&search=Circular"
SEBI_MASTER_CIRC_LIST_URL:  str  = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListingAll=yes&search=Master+Circular"
# PDF CDN base — PDFs stored as /sebi_data/commondocs/{mon-year}/{slug}_p.pdf
SEBI_CDN_BASE:              str  = "https://www.sebi.gov.in/sebi_data/commondocs"

# RBI endpoints
# Legacy script URLs (still live, but tables load via JS — partial static HTML)
RBI_BASE_URL:              str = "https://www.rbi.org.in"
RBI_NOTIFICATIONS_URL:     str = "https://www.rbi.org.in/scripts/NotificationUser.aspx"
RBI_PRESS_RELEASES_URL:    str = "https://www.rbi.org.in/scripts/PressReleaseDisplay.aspx"
RBI_CIRCULARS_URL:         str = "https://www.rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx"
# RBI RSS feeds — reliable alternative to the JS-rendered portal pages.
# These do NOT require a session cookie and are not subject to bot-detection (418).
# Format: Atom/RSS XML with <link> to the document page and PDF attachments.
RBI_NOTIFICATIONS_RSS_URL:  str = "https://www.rbi.org.in/scripts/rss.aspx"
# rss_pressrelease.aspx returns malformed XML (invalid token at col 0, likely BOM issue)
# Replaced with the RBI Atom feed which covers press releases and is well-formed
RBI_PRESS_RELEASES_RSS_URL: str = "https://www.rbi.org.in/scripts/atomfeed.aspx"
RBI_CIRCULARS_RSS_URL:      str = "https://www.rbi.org.in/scripts/rss_circular.aspx"

# BSE endpoints
# NOTE: /corporates/annreport.aspx?scripcode=X&type=Annual+Report&period=YEAR returned 404
# as of early 2026 — BSE restructured their filings portal.
# The BSE JSON API endpoint below is the stable replacement used by the BSE community.
# It returns a JSON array of {SCRIP_CD, FILENAME, REPORT_YEAR, ATTACHMENTNAME} objects.
BSE_BASE_URL:          str  = "https://www.bseindia.com"
BSE_ANNREPORT_API_URL: str  = "https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w"
BSE_ANNREPORT_URL:     str  = "https://www.bseindia.com/stock-share-price/annualreport"

# News RSS feeds
NEWS_RSS_FEEDS: dict[str, str] = {
    # Economic Times — markets feed (stable, confirmed working)
    "economic_times":     "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    # Hindu BusinessLine — primary replacement for Mint (XML well-formed)
    "hindu_businessline": "https://www.thehindubusinessline.com/markets/?service=rss",
    # Financial Express — corrected feed URL (was using /market/ sub-path which broke)
    "financial_express":  "https://www.financialexpress.com/feed/",
    # NDTV Profit via Feedburner — replacement for Bloomberg Quint (bqprime → ndtvprofit redirect broke)
    "ndtv_profit":        "https://feeds.feedburner.com/NDTV-Business",
    # Zee Business — replacement for Business Standard (403 on scraper UA)
    "zee_business":       "https://www.zeebiz.com/rss",
    # Zee Business markets feed — replaces Moneycontrol (all MC RSS feeds return malformed XML)
    # /latest.xml returns 403; /market.xml is the stable public feed
    "zee_business_markets": "https://www.zeebiz.com/market.xml",
}


# ─────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────

@dataclass
class DownloadResult:
    """Result of a single document download."""
    success:      bool
    url:          str
    file_path:    Optional[str]  = None
    filename:     Optional[str]  = None
    source:       str            = ""
    doc_type:     str            = "unknown"
    title:        Optional[str]  = None
    date:         Optional[str]  = None
    file_size_kb: float          = 0.0
    duration_ms:  float          = 0.0
    attempt:      int            = 1
    error:        Optional[str]  = None
    url_hash:     str            = field(default="")

    def __post_init__(self):
        if not self.url_hash:
            self.url_hash = _url_hash(self.url)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NewsArticle:
    """A single news article from RSS feed."""
    title:       str
    content:     str
    url:         str
    date:        str
    source:      str
    keywords_matched: list[str] = field(default_factory=list)
    url_hash:    str            = field(default="")

    def __post_init__(self):
        if not self.url_hash:
            self.url_hash = _url_hash(self.url)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────

class CrawlerError(Exception):
    """Base exception for all crawler failures."""
    def __init__(
        self,
        message: str,
        url: str = "",
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.url            = url
        self.original_error = original_error


class RateLimitError(CrawlerError):
    """Raised when rate limit is hit (HTTP 429)."""
    pass


class DownloadError(CrawlerError):
    """Raised when download fails after all retries."""
    pass


class ParseError(CrawlerError):
    """Raised when HTML parsing fails."""
    pass


# ─────────────────────────────────────────────
# CRAWLER CLASS
# ─────────────────────────────────────────────

class Crawler:
    """
    Async document crawler for Indian financial regulatory sources.

    Usage:
        from backend.ingestion.crawler import crawler

        # Download SEBI circulars
        paths = await crawler.crawl_sebi("2026-01-01", "2026-03-19")

        # Download RBI notifications
        paths = await crawler.crawl_rbi("2026-01-01", "2026-03-19")

        # Download BSE annual report
        paths = await crawler.crawl_bse_filings("500180")  # HDFC Bank

        # Fetch financial news
        articles = await crawler.crawl_news(
            keywords=["NIM", "HDFC Bank", "NPA"],
            days=7,
        )
    """

    def __init__(self):
        self._ua_index:       int   = 0
        self._domain_last_req: dict = {}     # domain → last request timestamp
        self._registry:       dict  = {}     # url_hash → file_path
        self._total_downloaded: int = 0
        self._total_skipped:    int = 0
        self._total_failed:     int = 0
        self._total_bytes:      int = 0

        # Ensure directories exist
        _ensure_directories()

        # Load download registry
        self._registry = _load_registry()

    # ─────────────────────────────────────────
    # PUBLIC: crawl_sebi
    # ─────────────────────────────────────────

    async def crawl_sebi(
        self,
        start_date: str,
        end_date: str,
        max_pages: int = 15,
        doc_types: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Scrapes SEBI circulars and master circulars within a date range.
        Downloads PDFs to data/raw/sebi/

        Uses the working listing URL:
            HomeAction.do?doListingAll=yes&search=Circular
        which returns a paginated HTML table with dates and /legal/circulars/ links.
        Each circular page embeds a PDF at:
            sebi_data/commondocs/{mon-year}/{slug}_p.pdf

        Args:
            start_date: ISO date string "YYYY-MM-DD"
            end_date:   ISO date string "YYYY-MM-DD"
            max_pages:  Maximum listing pages to scrape (default 15)
            doc_types:  Unused — kept for API compatibility

        Returns:
            List of absolute file paths for downloaded PDFs
        """
        start_dt = _parse_date(start_date)
        end_dt   = _parse_date(end_date)

        logger.info(
            f"[CRAWLER] SEBI crawl start | "
            f"range={start_date} to {end_date} | "
            f"max_pages={max_pages}"
        )

        # Both circulars and master circulars are scraped from the same listing
        list_urls = [
            ("circular",        SEBI_CIRCULARS_LIST_URL),
            ("master_circular", SEBI_MASTER_CIRC_LIST_URL),
        ]

        all_paths: list[str] = []
        seen_urls: set[str]  = set()

        async with self._make_client() as client:
            for doc_type, list_url in list_urls:
                # Listing pages paginate via JS but each page's HTML is accessible
                # by appending &pageno=N to the base URL (confirmed via page source)
                for page in range(1, max_pages + 1):
                    page_url = f"{list_url}&pageno={page}" if page > 1 else list_url
                    logger.debug(
                        f"[CRAWLER] SEBI {doc_type} listing page {page} | {page_url}"
                    )

                    try:
                        html  = await self._fetch_html(client, page_url)
                        links = _parse_sebi_links(html, start_dt, end_dt)

                        in_range = [l for l in links if not l.get("out_of_range")]

                        if not in_range:
                            # If we got rows but none in range, check if we've gone
                            # past start_date (all rows older) — stop pagination
                            all_rows = _parse_sebi_links(html, _parse_date("2000-01-01"), end_dt)
                            if not all_rows:
                                logger.info(
                                    f"[CRAWLER] SEBI {doc_type} page {page}: "
                                    f"empty page — stopping"
                                )
                                break
                            oldest = all_rows[-1].get("date", "")
                            oldest_dt = _parse_rss_date(oldest) if oldest else None
                            if oldest_dt and oldest_dt < start_dt:
                                logger.info(
                                    f"[CRAWLER] SEBI {doc_type} page {page}: "
                                    f"all entries older than {start_date} — stopping"
                                )
                                break
                            # Still within range but no matches on this page, continue
                            await asyncio.sleep(RATE_LIMIT_SECONDS)
                            continue

                        logger.info(
                            f"[CRAWLER] SEBI {doc_type} page {page} | "
                            f"{len(in_range)} links in range"
                        )

                        for link_info in in_range:
                            page_href = link_info["url"]
                            if page_href in seen_urls:
                                continue
                            seen_urls.add(page_href)

                            # Each entry links to /legal/circulars/{mon}/{slug}_{id}.html
                            # That page embeds the PDF — follow it to get the direct URL
                            try:
                                circ_html = await self._fetch_html(client, page_href)
                                pdf_url   = _extract_pdf_from_sebi_page(circ_html)
                                if not pdf_url:
                                    logger.warning(
                                        f"[CRAWLER] SEBI: no PDF on {page_href[-60:]}"
                                    )
                                    continue
                                await asyncio.sleep(RATE_LIMIT_SECONDS)
                            except Exception as e:
                                logger.warning(
                                    f"[CRAWLER] SEBI follow failed {page_href[-60:]}: {e}"
                                )
                                continue

                            result = await self._download_pdf(
                                client   = client,
                                url      = pdf_url,
                                dest_dir = SEBI_DIR,
                                metadata = {
                                    "source":   "sebi",
                                    "doc_type": doc_type,
                                    "title":    link_info.get("title", ""),
                                    "date":     link_info.get("date", ""),
                                },
                            )
                            if result.success and result.file_path:
                                all_paths.append(result.file_path)

                        await asyncio.sleep(RATE_LIMIT_SECONDS)

                    except CrawlerError as e:
                        logger.warning(
                            f"[CRAWLER] SEBI {doc_type} page {page} failed: {e}"
                        )
                        continue

        logger.info(
            f"[CRAWLER] SEBI crawl complete | "
            f"downloaded={len(all_paths)} PDFs"
        )
        log_metric("crawler_sebi_downloads", len(all_paths), unit="count")
        return all_paths

    # ─────────────────────────────────────────
    # PUBLIC: crawl_rbi
    # ─────────────────────────────────────────

    async def crawl_rbi(
        self,
        start_date: str,
        end_date: str,
        doc_types: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Scrapes RBI notifications, circulars, and press releases.
        Downloads PDFs to data/raw/rbi/

        Args:
            start_date: ISO date string "YYYY-MM-DD"
            end_date:   ISO date string "YYYY-MM-DD"
            doc_types:  Filter: ["notification", "circular", "press_release"]
                        None = all types

        Returns:
            List of absolute file paths for downloaded PDFs

        Examples:
            paths = await crawler.crawl_rbi("2026-01-01", "2026-03-19")
        """
        start_dt = _parse_date(start_date)
        end_dt   = _parse_date(end_date)

        logger.info(
            f"[CRAWLER] RBI crawl start | "
            f"range={start_date} to {end_date}"
        )

        endpoints = [
            # Each tuple: (doc_type, legacy_html_url, rss_feed_url)
            # Legacy HTML pages load content via JS (0 static links found).
            # RSS feeds are the reliable fallback — no session cookie needed,
            # no bot-detection (418), returns XML with direct doc links.
            ("notification",  RBI_NOTIFICATIONS_URL,   RBI_NOTIFICATIONS_RSS_URL),
            ("circular",      RBI_CIRCULARS_URL,        RBI_CIRCULARS_RSS_URL),
            ("press_release", RBI_PRESS_RELEASES_URL,   RBI_PRESS_RELEASES_RSS_URL),
        ]
        if doc_types:
            endpoints = [(t, u1, u2) for t, u1, u2 in endpoints if t in doc_types]

        all_paths: list[str] = []
        seen_urls: set[str]  = set()   # de-duplicate across legacy + new portal

        async with self._make_rbi_client() as client:
            for doc_type, legacy_url, rss_url in endpoints:
                links: list[dict] = []

                # ── Try 1: legacy HTML page (static links, usually 0 results) ──
                try:
                    html  = await self._fetch_html(client, legacy_url)
                    links = _parse_rbi_links(html, start_dt, end_dt)
                    logger.info(
                        f"[CRAWLER] RBI {doc_type} HTML | "
                        f"found {len(links)} links in date range"
                    )
                    await asyncio.sleep(RATE_LIMIT_SECONDS)
                except CrawlerError as e:
                    logger.warning(f"[CRAWLER] RBI {doc_type} HTML failed: {e}")

                # ── Try 2: RSS feed (reliable, no bot-detection, XML-parseable) ──
                # Re-use the existing RSS fetcher; it returns article dicts with
                # a "url" key pointing to the document or its landing page.
                # We then attempt to download any PDF linked from that page.
                if not links:
                    try:
                        rss_articles = await self._fetch_rss(
                            client    = client,
                            url       = rss_url,
                            source    = f"rbi_{doc_type}",
                            keywords  = [],          # no keyword filter for RBI
                            cutoff_dt = start_dt,
                            max_items = 50,
                        )
                        for art in rss_articles:
                            art_date = art.get("date", "")
                            art_url  = art.get("url", "")
                            # Parse pub date to check range
                            art_dt = _parse_rss_date(art_date) if art_date else None
                            if art_dt and art_dt > end_dt:
                                continue
                            if art_dt and art_dt < start_dt:
                                continue
                            if art_url:
                                links.append({
                                    "url":   art_url,
                                    "title": art.get("title", ""),
                                    "date":  art_date,
                                    "needs_resolve": not art_url.lower().endswith(".pdf"),
                                })
                        logger.info(
                            f"[CRAWLER] RBI {doc_type} RSS | "
                            f"found {len(links)} items in date range"
                        )
                        await asyncio.sleep(RATE_LIMIT_SECONDS)
                    except Exception as e:
                        logger.warning(f"[CRAWLER] RBI {doc_type} RSS failed: {e}")

                # ── Download each resolved PDF ──
                for link_info in links:
                    pdf_url = link_info.get("url", "")
                    if not pdf_url or pdf_url in seen_urls:
                        continue

                    # Resolve relative URLs
                    if not pdf_url.startswith("http"):
                        pdf_url = urljoin(RBI_BASE_URL, pdf_url)

                    # If the link points to a landing page (not a .pdf), follow it
                    if link_info.get("needs_resolve") and not pdf_url.lower().endswith(".pdf"):
                        try:
                            page_html = await self._fetch_html(client, pdf_url)
                            resolved  = _extract_pdf_from_rbi_page(page_html)
                            if resolved:
                                pdf_url = resolved
                            else:
                                logger.debug(
                                    f"[CRAWLER] RBI: no PDF on page {pdf_url[:80]}"
                                )
                                continue
                            await asyncio.sleep(RATE_LIMIT_SECONDS)
                        except Exception as e:
                            logger.warning(f"[CRAWLER] RBI follow failed: {e}")
                            continue

                    seen_urls.add(pdf_url)
                    result = await self._download_pdf(
                        client   = client,
                        url      = pdf_url,
                        dest_dir = RBI_DIR,
                        metadata = {
                            "source":   "rbi",
                            "doc_type": doc_type,
                            "title":    link_info.get("title", ""),
                            "date":     link_info.get("date", ""),
                        },
                    )
                    if result.success and result.file_path:
                        all_paths.append(result.file_path)

        logger.info(
            f"[CRAWLER] RBI crawl complete | "
            f"downloaded={len(all_paths)} PDFs"
        )
        log_metric("crawler_rbi_downloads", len(all_paths), unit="count")
        return all_paths

    # ─────────────────────────────────────────
    # PUBLIC: crawl_bse_filings
    # ─────────────────────────────────────────

    async def crawl_bse_filings(
        self,
        company_code: str,
        filing_types: Optional[list[str]] = None,
        years: Optional[list[int]] = None,
    ) -> list[str]:
        """
        Downloads annual reports and key filings for a BSE-listed company.

        Args:
            company_code: BSE scrip code e.g. "500180" (HDFC Bank)
            filing_types: ["annual_report", "results", "investor_pres"]
                          None = annual reports only
            years:        List of years e.g. [2024, 2025, 2026]
                          None = last 3 years

        Returns:
            List of downloaded file paths

        Examples:
            # HDFC Bank
            paths = await crawler.crawl_bse_filings("500180")

            # TCS
            paths = await crawler.crawl_bse_filings(
                "532540",
                filing_types=["annual_report", "results"],
                years=[2025, 2026],
            )
        """
        if years is None:
            current_year = datetime.now().year
            years = [current_year - 2, current_year - 1, current_year]

        logger.info(
            f"[CRAWLER] BSE filings | "
            f"company={company_code} | "
            f"years={years}"
        )

        all_paths: list[str] = []

        async with self._make_client() as client:
            # ── Step 1: Fetch annual report listing from BSE JSON API ──────────
            # BSE JSON API returns array of report objects for a given scrip code.
            # Each object has: SCRIP_CD, REPORT_YEAR, FILENAME, ATTACHMENTNAME
            api_url = f"{BSE_ANNREPORT_API_URL}?scripcode={company_code}&type=AR"

            try:
                await self._rate_limit(api_url)
                response = await client.get(api_url)
                response.raise_for_status()
                report_list = response.json()
            except Exception as e:
                logger.warning(
                    f"[CRAWLER] BSE JSON API failed for {company_code}: {e} — "
                    f"falling back to corporate filings page"
                )
                report_list = []

            # ── Step 2: Filter by requested years and download each PDF ────────
            # The JSON objects use REPORT_YEAR like "2024-25", "2025-26" etc.
            # We match on the first year component matching our years list.
            for report in report_list:
                try:
                    report_year_str = str(report.get("REPORT_YEAR", ""))
                    # "2024-25" → first component is 2024
                    report_year_int = int(report_year_str.split("-")[0])
                    if years and report_year_int not in years:
                        continue

                    filename = report.get("FILENAME", "") or report.get("ATTACHMENTNAME", "")
                    if not filename:
                        continue

                    # Build direct PDF URL — BSE hosts attachments at xml-data/corpfiling
                    if filename.startswith("http"):
                        pdf_url = filename
                    else:
                        pdf_url = f"{BSE_BASE_URL}/xml-data/corpfiling/AttachHis/{filename}"

                    title = (
                        f"Annual Report {report_year_str} - "
                        f"{report.get('SCRIP_CD', company_code)}"
                    )

                    result = await self._download_pdf(
                        client   = client,
                        url      = pdf_url,
                        dest_dir = BSE_DIR / company_code,
                        metadata = {
                            "source":       "bse",
                            "doc_type":     "annual_report",
                            "company_code": company_code,
                            "year":         report_year_str,
                            "title":        title,
                        },
                    )
                    if result.success and result.file_path:
                        all_paths.append(result.file_path)

                    await asyncio.sleep(RATE_LIMIT_SECONDS)

                except Exception as e:
                    logger.warning(
                        f"[CRAWLER] BSE filing {company_code} parse error: {e}"
                    )
                    continue

        logger.info(
            f"[CRAWLER] BSE complete | "
            f"company={company_code} | "
            f"downloaded={len(all_paths)}"
        )
        return all_paths

    # ─────────────────────────────────────────
    # PUBLIC: crawl_news
    # ─────────────────────────────────────────

    async def crawl_news(
        self,
        keywords: list[str],
        days: int = 7,
        sources: Optional[list[str]] = None,
        max_articles_per_source: int = 20,
    ) -> list[dict]:
        """
        Fetches financial news articles from RSS feeds.
        Filters by keywords and date range.

        Args:
            keywords:                 List of keywords/company names to filter
            days:                     How many days back to look (default 7)
            sources:                  RSS source names to include
                                      None = all sources
            max_articles_per_source:  Cap per RSS feed (default 20)

        Returns:
            List of NewsArticle dicts:
            [
                {
                    "title":    "HDFC Bank NIM expands in Q3 FY26",
                    "content":  "...",
                    "date":     "2026-03-15",
                    "url":      "https://...",
                    "source":   "economic_times",
                    "keywords_matched": ["HDFC Bank", "NIM"],
                    "url_hash": "abc123...",
                }
            ]

        Examples:
            articles = await crawler.crawl_news(
                keywords=["HDFC Bank", "NIM", "NPA"],
                days=7,
            )
        """
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
        keywords_lower = [k.lower() for k in keywords]

        feeds_to_crawl = {
            name: url for name, url in NEWS_RSS_FEEDS.items()
            if sources is None or name in sources
        }

        logger.info(
            f"[CRAWLER] News crawl | "
            f"keywords={keywords} | "
            f"days={days} | "
            f"sources={list(feeds_to_crawl.keys())}"
        )

        all_articles: list[dict] = []

        async with self._make_client() as client:
            for source_name, rss_url in feeds_to_crawl.items():
                try:
                    articles = await self._fetch_rss(
                        client     = client,
                        url        = rss_url,
                        source     = source_name,
                        keywords   = keywords_lower,
                        cutoff_dt  = cutoff_dt,
                        max_items  = max_articles_per_source,
                    )
                    all_articles.extend(articles)
                    logger.info(
                        f"[CRAWLER] RSS {source_name} | "
                        f"matched={len(articles)} articles"
                    )
                    await asyncio.sleep(RATE_LIMIT_SECONDS)

                except Exception as e:
                    logger.warning(
                        f"[CRAWLER] RSS {source_name} failed: {e}"
                    )
                    continue

        # Save to disk
        saved_paths: list[str] = []
        for article in all_articles:
            path = _save_news_article(article)
            if path:
                saved_paths.append(path)

        logger.info(
            f"[CRAWLER] News complete | "
            f"total={len(all_articles)} articles | "
            f"saved={len(saved_paths)}"
        )
        log_metric("crawler_news_articles", len(all_articles), unit="count")

        return [a if isinstance(a, dict) else a.to_dict() for a in all_articles]

    # ─────────────────────────────────────────
    # PRIVATE: HTTP client factory
    # ─────────────────────────────────────────

    def _make_client(self) -> httpx.AsyncClient:
        """Creates an httpx async client with shared settings."""
        return httpx.AsyncClient(
            timeout         = httpx.Timeout(REQUEST_TIMEOUT),
            follow_redirects= True,
            verify          = False,          # Some govt sites have cert issues
            headers         = self._next_headers(),
            limits          = httpx.Limits(
                max_connections=5,
                max_keepalive_connections=3,
            ),
        )

    def _make_rbi_client(self) -> httpx.AsyncClient:
        """
        RBI-specific httpx client.
        RBI's Liferay portal returns HTTP 418 (bot detection) unless the
        request includes a Referer header and a fully-qualified Accept string
        that matches a real browser. This client passes those checks.
        """
        ua = USER_AGENTS[self._ua_index % len(USER_AGENTS)]
        self._ua_index += 1
        return httpx.AsyncClient(
            timeout          = httpx.Timeout(REQUEST_TIMEOUT),
            follow_redirects = True,
            verify           = False,
            limits           = httpx.Limits(
                max_connections=5,
                max_keepalive_connections=3,
            ),
            headers = {
                "User-Agent":      ua,
                "Accept":          (
                    "text/html,application/xhtml+xml,application/xml;"
                    "q=0.9,image/avif,image/webp,image/apng,*/*;"
                    "q=0.8,application/signed-exchange;v=b3;q=0.7"
                ),
                "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer":         "https://www.rbi.org.in/",
                "Origin":          "https://www.rbi.org.in",
                "Connection":      "keep-alive",
                "Cache-Control":   "max-age=0",
                "Sec-Fetch-Dest":  "document",
                "Sec-Fetch-Mode":  "navigate",
                "Sec-Fetch-Site":  "same-origin",
                "Sec-Fetch-User":  "?1",
                "Upgrade-Insecure-Requests": "1",
            },
        )

    def _next_headers(self) -> dict[str, str]:
        """Returns headers with the next user-agent in rotation."""
        ua = USER_AGENTS[self._ua_index % len(USER_AGENTS)]
        self._ua_index += 1
        return {
            "User-Agent":      ua,
            "Accept":          "text/html,application/pdf,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection":      "keep-alive",
            "Cache-Control":   "no-cache",
        }

    # ─────────────────────────────────────────
    # PRIVATE: fetch HTML
    # ─────────────────────────────────────────

    async def _fetch_html(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> str:
        """
        Fetches HTML from URL with rate limiting and retry.

        Returns:
            HTML string

        Raises:
            CrawlerError after max retries
        """
        await self._rate_limit(url)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.get(
                    url,
                    headers=self._next_headers(),
                )

                if response.status_code == 429:
                    wait = RETRY_BACKOFF_BASE ** attempt * 5
                    logger.warning(
                        f"[CRAWLER] Rate limited by {_domain(url)} | "
                        f"waiting {wait}s"
                    )
                    await asyncio.sleep(wait)
                    continue

                if response.status_code == 404:
                    raise CrawlerError(
                        f"404 Not Found: {url}", url=url
                    )

                response.raise_for_status()

                html = response.text
                if len(html) > MAX_HTML_SIZE_MB * 1024 * 1024:
                    html = html[: MAX_HTML_SIZE_MB * 1024 * 1024]

                logger.debug(
                    f"[CRAWLER] Fetched HTML | "
                    f"url={url[:80]} | "
                    f"size={len(html)//1024}KB | "
                    f"status={response.status_code}"
                )
                return html

            except httpx.TimeoutException:
                wait = RETRY_BACKOFF_BASE ** (attempt - 1)
                logger.warning(
                    f"[CRAWLER] Timeout | "
                    f"url={url[:60]} | "
                    f"attempt={attempt}/{MAX_RETRIES} | "
                    f"retry in {wait}s"
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(wait)

            except CrawlerError:
                raise

            except Exception as e:
                wait = RETRY_BACKOFF_BASE ** (attempt - 1)
                logger.warning(
                    f"[CRAWLER] Fetch error | "
                    f"url={url[:60]} | "
                    f"attempt={attempt}/{MAX_RETRIES} | "
                    f"error={e}"
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(wait)

        raise DownloadError(
            f"Failed to fetch HTML after {MAX_RETRIES} attempts: {url}",
            url=url,
        )

    # ─────────────────────────────────────────
    # PRIVATE: download PDF
    # ─────────────────────────────────────────

    async def _download_pdf(
        self,
        client:   httpx.AsyncClient,
        url:      str,
        dest_dir: Path,
        metadata: dict,
    ) -> DownloadResult:
        """
        Downloads a PDF to dest_dir with duplicate detection.
        Skips if URL already downloaded (checked via registry).

        Returns:
            DownloadResult with success status and file path
        """
        url_hash = _url_hash(url)

        # Duplicate check
        if url_hash in self._registry:
            existing_path = self._registry[url_hash]
            if Path(existing_path).exists():
                self._total_skipped += 1
                logger.debug(
                    f"[CRAWLER] Skipping duplicate | "
                    f"url={url[:60]} | "
                    f"existing={existing_path}"
                )
                return DownloadResult(
                    success   = True,
                    url       = url,
                    file_path = existing_path,
                    url_hash  = url_hash,
                    error     = "skipped_duplicate",
                )

        await self._rate_limit(url)

        for attempt in range(1, MAX_RETRIES + 1):
            download_start = time.perf_counter()
            try:
                response = await client.get(
                    url,
                    headers=self._next_headers(),
                )

                if response.status_code == 429:
                    wait = RETRY_BACKOFF_BASE ** attempt * 5
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
                    logger.debug(
                        f"[CRAWLER] Skipping non-PDF: {url[:60]} | "
                        f"content-type={content_type}"
                    )
                    return DownloadResult(
                        success  = False,
                        url      = url,
                        url_hash = url_hash,
                        error    = f"not_pdf: {content_type}",
                    )

                content = response.content
                size_mb = len(content) / (1024 * 1024)

                if size_mb > MAX_PDF_SIZE_MB:
                    logger.warning(
                        f"[CRAWLER] PDF too large ({size_mb:.1f}MB > "
                        f"{MAX_PDF_SIZE_MB}MB): {url[:60]}"
                    )
                    return DownloadResult(
                        success  = False,
                        url      = url,
                        url_hash = url_hash,
                        error    = f"file_too_large: {size_mb:.1f}MB",
                    )

                # Build structured filename
                dest_dir.mkdir(parents=True, exist_ok=True)
                filename = _build_filename(url, metadata)
                file_path = dest_dir / filename

                with open(file_path, "wb") as f:
                    f.write(content)

                duration_ms = (time.perf_counter() - download_start) * 1000
                self._total_downloaded += 1
                self._total_bytes      += len(content)

                # Register download
                self._registry[url_hash] = str(file_path)
                _save_registry(self._registry)

                logger.info(
                    f"[CRAWLER] Downloaded | "
                    f"file={filename} | "
                    f"size={size_mb:.2f}MB | "
                    f"attempt={attempt} | "
                    f"{duration_ms:.0f}ms"
                )
                log_metric("crawler_download_size_kb", size_mb * 1024, unit="kb")

                return DownloadResult(
                    success      = True,
                    url          = url,
                    file_path    = str(file_path),
                    filename     = filename,
                    source       = metadata.get("source", ""),
                    doc_type     = metadata.get("doc_type", ""),
                    title        = metadata.get("title"),
                    date         = metadata.get("date"),
                    file_size_kb = size_mb * 1024,
                    duration_ms  = duration_ms,
                    attempt      = attempt,
                    url_hash     = url_hash,
                )

            except httpx.TimeoutException:
                wait = RETRY_BACKOFF_BASE ** (attempt - 1)
                logger.warning(
                    f"[CRAWLER] PDF download timeout | "
                    f"url={url[:60]} | "
                    f"attempt={attempt}/{MAX_RETRIES}"
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(wait)

            except Exception as e:
                wait = RETRY_BACKOFF_BASE ** (attempt - 1)
                logger.warning(
                    f"[CRAWLER] PDF download error | "
                    f"url={url[:60]} | "
                    f"attempt={attempt} | error={e}"
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(wait)

        self._total_failed += 1
        return DownloadResult(
            success  = False,
            url      = url,
            url_hash = url_hash,
            error    = f"failed_after_{MAX_RETRIES}_attempts",
        )

    # ─────────────────────────────────────────
    # PRIVATE: fetch RSS
    # ─────────────────────────────────────────

    async def _fetch_rss(
        self,
        client:    httpx.AsyncClient,
        url:       str,
        source:    str,
        keywords:  list[str],
        cutoff_dt: datetime,
        max_items: int,
    ) -> list[dict]:
        """Fetches and parses RSS feed, filtering by keywords and date."""
        await self._rate_limit(url)

        try:
            response = await client.get(url, headers=self._next_headers())
            response.raise_for_status()
            xml_content = response.text
        except Exception as e:
            logger.warning(f"[CRAWLER] RSS fetch failed {url}: {e}")
            return []

        articles: list[dict] = []

        try:
            # Strip BOM before parsing — RBI RSS feeds are served with a BOM
            # (﻿ in str form after decoding) that causes "invalid token at col 0".
            xml_clean = xml_content.lstrip("\r\n ").lstrip(chr(0xfeff)).lstrip("\r\n ")
            # If it still doesn't start with < (binary garbage), try raw bytes
            if not xml_clean.lstrip().startswith("<"):
                xml_bytes = response.content.lstrip(b"\xff\xfe\xfe\xff\xef\xbb\xbf")
                xml_clean = xml_bytes.decode("utf-8", errors="replace").lstrip(" \r\n")
            root = ET.fromstring(xml_clean)
            # Handle both RSS 2.0 and Atom
            items = (
                root.findall(".//item")           # RSS 2.0
                or root.findall(".//{http://www.w3.org/2005/Atom}entry")  # Atom
            )

            for item in items[:max_items * 3]:   # Over-fetch then filter
                title   = _xml_text(item, ["title"])
                link    = _xml_text(item, ["link", "guid",
                          "{http://www.w3.org/2005/Atom}link"])
                pub_date = _xml_text(item, ["pubDate", "published",
                           "{http://www.w3.org/2005/Atom}published",
                           "dc:date"])
                description = _xml_text(item, ["description", "summary",
                              "{http://www.w3.org/2005/Atom}summary",
                              "content:encoded"])

                if not title or not link:
                    continue

                # Date filter
                article_dt = _parse_rss_date(pub_date)
                if article_dt and article_dt < cutoff_dt:
                    continue

                # Keyword filter
                combined = f"{title} {description}".lower()
                matched  = [kw for kw in keywords if kw in combined]
                if not matched:
                    continue

                url_hash = _url_hash(link)
                if url_hash in self._registry:
                    continue

                article = NewsArticle(
                    title            = title.strip(),
                    content          = _clean_html(description or ""),
                    url              = link.strip(),
                    date             = pub_date or "",
                    source           = source,
                    keywords_matched = matched,
                    url_hash         = url_hash,
                )
                articles.append(article.to_dict())

                if len(articles) >= max_items:
                    break

        except ET.ParseError as e:
            logger.warning(f"[CRAWLER] RSS parse error {url}: {e}")

        return articles

    # ─────────────────────────────────────────
    # PRIVATE: rate limiting
    # ─────────────────────────────────────────

    async def _rate_limit(self, url: str) -> None:
        """
        Enforces per-domain rate limiting.
        Waits if last request to domain was < RATE_LIMIT_SECONDS ago.
        """
        domain = _domain(url)
        now    = time.monotonic()
        last   = self._domain_last_req.get(domain, 0)
        elapsed = now - last

        if elapsed < RATE_LIMIT_SECONDS:
            wait = RATE_LIMIT_SECONDS - elapsed
            logger.debug(
                f"[CRAWLER] Rate limit | "
                f"domain={domain} | "
                f"waiting {wait:.2f}s"
            )
            await asyncio.sleep(wait)

        self._domain_last_req[domain] = time.monotonic()

    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns crawler operational statistics."""
        return {
            "total_downloaded":   self._total_downloaded,
            "total_skipped":      self._total_skipped,
            "total_failed":       self._total_failed,
            "total_bytes":        self._total_bytes,
            "total_size_mb":      round(self._total_bytes / (1024 * 1024), 2),
            "registry_size":      len(self._registry),
        }

    def clear_registry(self) -> None:
        """Clears the download registry (use with caution)."""
        self._registry = {}
        _save_registry(self._registry)
        logger.warning("[CRAWLER] Download registry cleared")

    def is_already_downloaded(self, url: str) -> bool:
        """Checks if a URL has already been downloaded."""
        return _url_hash(url) in self._registry


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _ensure_directories() -> None:
    """Creates data/raw subdirectories if they don't exist."""
    for d in [SEBI_DIR, RBI_DIR, BSE_DIR, NEWS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_registry() -> dict:
    """Loads URL → file_path registry from disk."""
    if DOWNLOAD_REGISTRY.exists():
        try:
            with open(DOWNLOAD_REGISTRY, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_registry(registry: dict) -> None:
    """Persists registry to disk atomically."""
    tmp = DOWNLOAD_REGISTRY.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(registry, f, indent=2)
    tmp.replace(DOWNLOAD_REGISTRY)


def _url_hash(url: str) -> str:
    """MD5 hash of URL for deduplication."""
    return hashlib.md5(url.strip().encode()).hexdigest()


def _domain(url: str) -> str:
    """Extracts domain from URL."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return url[:30]


def _parse_date(date_str: str) -> datetime:
    """Parses ISO date string to datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _parse_rss_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parses RSS pubDate strings in multiple formats.
    Returns timezone-aware datetime or None.
    """
    if not date_str:
        return None
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _build_filename(url: str, metadata: dict) -> str:
    """
    Builds a structured filename from URL and metadata.
    Format: {source}_{doctype}_{company}_{date}_{hash8}.pdf
    """
    source   = re.sub(r'[^a-z0-9]', '', metadata.get("source", "doc").lower())
    doc_type = re.sub(r'[^a-z0-9]', '_', metadata.get("doc_type", "").lower())
    company  = re.sub(r'[^a-z0-9]', '_', metadata.get("company_code", "").lower())
    date_str = re.sub(r'[^0-9]', '', metadata.get("date", ""))[:8]
    url_hash_short = _url_hash(url)[:8]

    parts = [source]
    if doc_type:
        parts.append(doc_type)
    if company:
        parts.append(company)
    if date_str:
        parts.append(date_str)
    parts.append(url_hash_short)

    return "_".join(filter(None, parts)) + ".pdf"


def _parse_sebi_links(
    html: str,
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict]:
    """
    Parses SEBI listing page HTML (HomeAction.do?doListingAll=yes&search=Circular).

    The listing page renders an HTML table:
        <tr>
          <td>Mar 20, 2026</td>
          <td>Master Circulars</td>
          <td><a href="/legal/master-circulars/mar-2026/slug_100491.html">Title</a></td>
        </tr>

    Each row links to an intermediate /legal/circulars/ HTML page.
    crawl_sebi() follows each link to extract the embedded PDF URL.

    Returns list of dicts with keys: url, title, date, out_of_range
    """
    links: list[dict] = []

    try:
        row_pattern = re.compile(
            r'<tr[^>]*>.*?</tr>', re.DOTALL | re.IGNORECASE
        )
        # Match /legal/circulars/, /legal/master-circulars/, or direct .pdf
        link_pattern = re.compile(
            r'href=["\']((https?://(?:www\.)?sebi\.gov\.in)?'
            r'/(?:legal/(?:circulars|master-circulars|advisories)|'
            r'sebi_data/commondocs)/[^"\']+)["\']',
            re.IGNORECASE,
        )
        # SEBI listing uses "Mon DD, YYYY" format e.g. "Mar 20, 2026"
        date_pattern = re.compile(
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+20[12]\d)\b',
            re.IGNORECASE,
        )
        title_pattern = re.compile(
            r'<a[^>]+href=["\'][^"\']*legal[^"\']*["\'][^>]*>([^<]{5,300})</a>',
            re.IGNORECASE,
        )

        for row in row_pattern.finditer(html):
            row_html   = row.group(0)
            link_match = link_pattern.search(row_html)
            date_match = date_pattern.search(row_html)

            if not link_match or not link_match.group(1):
                continue

            # Parse and range-check date
            doc_date  = None
            date_str  = ""
            if date_match:
                date_str = date_match.group(1)
                doc_date = _parse_rss_date(date_str)

            out_of_range = bool(
                doc_date and (doc_date < start_dt or doc_date > end_dt)
            )

            href = link_match.group(1)
            if not href.startswith("http"):
                href = urljoin(SEBI_BASE_URL, href)

            title = ""
            title_match = title_pattern.search(row_html)
            if title_match:
                title = _clean_html(title_match.group(1))

            links.append({
                "url":          href,
                "title":        title,
                "date":         date_str,
                "out_of_range": out_of_range,
                "needs_follow": not href.lower().endswith(".pdf"),
            })

    except Exception as e:
        logger.warning(f"[CRAWLER] SEBI link parsing error: {e}")

    return links


def _extract_pdf_from_sebi_page(html: str) -> Optional[str]:
    """
    Extracts the direct PDF download URL from a SEBI /legal/circulars/ page.
    These pages embed the PDF either as a direct href or in an <iframe src=...>.
    Returns the absolute PDF URL or None if not found.
    """
    patterns = [
        re.compile(r'href=["\']([^"\']*\.pdf)["\']', re.IGNORECASE),
        re.compile(r'src=["\']([^"\']*\.pdf)["\']',  re.IGNORECASE),
        re.compile(r'data=["\']([^"\']*\.pdf)["\']', re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(html)
        if m:
            url = m.group(1)
            if not url.startswith("http"):
                url = urljoin(SEBI_BASE_URL, url)
            return url
    return None



def _extract_pdf_from_rbi_page(html: str) -> Optional[str]:
    """
    Extracts a direct PDF URL from an RBI document landing page.
    RBI notification/circular pages embed the PDF in an <a href> or <iframe>.
    Tries both the /rdocs/ CDN path and generic .pdf patterns.
    Returns the absolute PDF URL or None if not found.
    """
    patterns = [
        # RBI CDN direct links (most reliable)
        re.compile(
            r'href=["\']([^"\']*(?:/rdocs/|/upload/|/content/)[^"\']*\.pdf)["\']',
            re.IGNORECASE,
        ),
        # Generic PDF href
        re.compile(r'href=["\']([^"\']*\.pdf)["\']', re.IGNORECASE),
        # iframe src
        re.compile(r'src=["\']([^"\']*\.pdf)["\']',  re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(html)
        if m:
            url = m.group(1)
            if not url.startswith("http"):
                url = urljoin(RBI_BASE_URL, url)
            return url
    return None
def _parse_rbi_links(
    html: str,
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict]:
    """
    Parses RBI listing page HTML for document links.

    RBI uses two URL structures:
      - Legacy: /scripts/NotificationUser.aspx?Id=XXXX&Mode=0
                /scripts/PressReleaseDisplay.aspx?prid=XXXXX
      - New (2024+): /en/web/rbi/-/notifications/{slug}
                     /en/web/rbi/-/press-releases/{slug}
                     direct .pdf hrefs under /rdocs/ or /upload/
    The listing pages load their tables via JS on modern browsers, but the
    static HTML still includes <a href> rows in a hidden table or noscript block.
    We cast a wide net across all RBI link patterns.
    """
    links: list[dict] = []

    try:
        pdf_pattern = re.compile(
            r'href=["\']('
            r'[^"\']*\.pdf'
            r'|[^"\']*(?:NotificationUser|PressReleaseDisplay|BS_ViewMasDirections)'
            r'[^"\']*'
            r'|[^"\']*(?:/en/web/rbi/-/(?:notifications|press-releases|publications)/'
            r'[^"\']+)'
            r')["\']',
            re.IGNORECASE,
        )
        date_pattern = re.compile(
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
            r'\s+\d{1,2},?\s+20[12]\d'
            r'|\d{1,2}[/-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
            r'[/-]20[12]\d'
            r'|\d{4}-\d{2}-\d{2})\b',
            re.IGNORECASE,
        )
        title_pattern = re.compile(
            r'<a[^>]*href=[^>]*>([^<]{10,300})</a>', re.IGNORECASE
        )

        seen_urls: set[str] = set()

        for match in pdf_pattern.finditer(html):
            url = match.group(1)
            if not url.startswith("http"):
                url = urljoin(RBI_BASE_URL, url)

            # Skip nav/header links that are not document links
            if any(skip in url for skip in [
                "/home", "/about", "/contact", "/sitemap",
                "javascript:", "#", "login", "register",
            ]):
                continue

            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Widen context window — RBI rows are sometimes large
            start_pos  = max(0, match.start() - 400)
            context    = html[start_pos: match.end() + 400]
            date_match  = date_pattern.search(context)
            title_match = title_pattern.search(context)

            doc_date = None
            if date_match:
                doc_date = _parse_rss_date(date_match.group(1))

            if doc_date and (doc_date < start_dt or doc_date > end_dt):
                continue

            links.append({
                "url":   url,
                "title": _clean_html(title_match.group(1)) if title_match else "",
                "date":  date_match.group(1) if date_match else "",
            })

    except Exception as e:
        logger.warning(f"[CRAWLER] RBI link parsing error: {e}")

    return links

import requests, io
from backend.ingestion.pdf_parser import PDFParser

def fetch_sebi_pdf_text(pdf_url: str) -> str:
    """Download and extract text from SEBI PDF circular."""
    resp = requests.get(pdf_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    parser = PDFParser()
    return parser.parse_bytes(io.BytesIO(resp.content))  # adjust to your PDFParser API

def _parse_bse_links(
    html: str,
    company_code: str,
    year: int,
) -> list[dict]:
    """Parses BSE filing page for annual report PDF links."""
    links: list[dict] = []

    try:
        pdf_pattern = re.compile(
            r'href=["\']([^"\']*(?:AnnualReports?|annrep)[^"\']*\.pdf)["\']',
            re.IGNORECASE,
        )
        for match in pdf_pattern.finditer(html):
            url = match.group(1)
            if not url.startswith("http"):
                url = urljoin(BSE_BASE_URL, url)

            links.append({
                "url":   url,
                "title": f"Annual Report {year} - {company_code}",
                "date":  str(year),
            })

        # Also try direct BSE CDN URL pattern
        cdn_url = (
            f"https://www.bseindia.com/AnnualReports/{company_code}/"
            f"AnnualReport{year}.pdf"
        )
        links.append({
            "url":   cdn_url,
            "title": f"Annual Report {year} - {company_code}",
            "date":  str(year),
        })

    except Exception as e:
        logger.warning(f"[CRAWLER] BSE link parsing error: {e}")

    return links


def _xml_text(element, tag_options: list[str]) -> Optional[str]:
    """
    Extracts text from first matching XML tag.
    Tries multiple tag names for RSS 2.0 / Atom compatibility.
    """
    for tag in tag_options:
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        # For Atom <link href="...">
        if child is not None:
            href = child.get("href")
            if href:
                return href
    return None


def _clean_html(html_str: str) -> str:
    """Strips HTML tags from string."""
    if not html_str:
        return ""
    text = re.sub(r'<[^>]+>', ' ', html_str)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _save_news_article(article: dict) -> Optional[str]:
    """Saves news article as JSON file to data/raw/news/"""
    try:
        filename = f"news_{article.get('source', 'unknown')}_{article['url_hash'][:12]}.json"
        file_path = NEWS_DIR / filename
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(article, f, indent=2, ensure_ascii=False)
        return str(file_path)
    except Exception as e:
        logger.warning(f"[CRAWLER] Failed to save news article: {e}")
        return None


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

crawler = Crawler()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "crawler",
    "Crawler",
    "DownloadResult",
    "NewsArticle",
    "CrawlerError",
    "DownloadError",
    "RateLimitError",
    "NEWS_RSS_FEEDS",
    "USER_AGENTS",
]