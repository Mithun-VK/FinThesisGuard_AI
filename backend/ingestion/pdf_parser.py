# backend/ingestion/pdf_parser.py
"""
FinThesisGuard AI — PDF Parser
ArkAngel Financial Solutions

Extracts clean, structured text from financial PDFs.
Handles annual reports, SEBI circulars, RBI notifications,
earnings transcripts, and broker research reports.

Strategy:
    1. Primary:  PyPDF2/pypdf  — fast, clean for digital PDFs
    2. Secondary: PyMuPDF (fitz) — better layout handling
    3. Fallback:  PaddleOCR    — for scanned/image-based PDFs

Table extraction uses PyMuPDF's block detection for financial tables.
"""

import io
import os
import re
import time
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.config import settings
from backend.utils.logger import logger, log_metric


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Minimum chars per page to consider it text-based (not scanned)
MIN_CHARS_PER_PAGE_THRESHOLD: int = 50

# OCR confidence target
OCR_ACCURACY_TARGET: float = 0.90

# Max pages to process (very large PDFs are truncated)
MAX_PAGES: int = 500

# Financial table detection keywords
TABLE_HEADER_KEYWORDS: set[str] = {
    "particulars", "description", "item", "quarter", "year",
    "fy", "q1", "q2", "q3", "q4", "total", "amount", "crore",
    "lakh", "million", "billion", "net", "gross", "ratio",
    "%", "rs.", "inr", "usd", "revenue", "profit", "loss",
    "asset", "liability", "equity", "income", "expense",
    "nim", "gnpa", "nnpa", "casa", "crar", "roe", "eps",
}

# Boilerplate legal phrases to strip from financial documents
BOILERPLATE_PATTERNS: list[str] = [
    r"this\s+document\s+is\s+for\s+(?:information|informational)\s+purposes?\s+only",
    r"past\s+performance\s+(?:is\s+not|does\s+not\s+guarantee)\s+(?:a\s+)?(?:guarantee|indicator)\s+of\s+future",
    r"this\s+(?:report|document|material)\s+(?:has\s+been\s+)?prepared\s+by",
    r"(?:please\s+)?(?:read|refer\s+to)\s+the\s+(?:important\s+)?(?:disclosures?|disclaimers?)",
    r"(?:sebi\s+)?registration\s+(?:number|no\.?)\s*[:\-]?\s*[A-Z0-9]+",
    r"analyst\s+certification",
    r"conflicts?\s+of\s+interest",
    r"all\s+rights?\s+reserved",
    r"confidential(?:ity)?\s+notice",
    r"not\s+for\s+(?:public\s+)?distribution",
    r"for\s+(?:private\s+)?(?:circulation|use)\s+only",
    r"investment\s+in\s+(?:securities|mutual\s+funds?)\s+(?:is\s+)?subject\s+to\s+market\s+risk",
    r"mutual\s+fund\s+investments?\s+are\s+subject\s+to\s+market\s+risks?",
    r"page\s+\d+\s+of\s+\d+",
    r"^\s*\d+\s*$",                   # Lone page numbers
    r"(?:cin|llpin)\s*:\s*[A-Z0-9]+", # Company identification numbers in headers
]

# Header/footer detection — lines appearing on many pages are likely headers/footers
HEADER_FOOTER_MAX_OCCURRENCES: int = 5


# ─────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────

@dataclass
class ParsedPage:
    """Text and metadata extracted from a single PDF page."""
    page_number: int
    text: str
    char_count: int
    word_count: int
    is_scanned: bool = False
    ocr_confidence: float = 1.0
    has_tables: bool = False


@dataclass
class ExtractedTable:
    """
    A financial table extracted from a PDF page.
    Stored alongside text chunks for structured data retrieval.
    """
    page_number: int
    headers: list[str]
    rows: list[list[str]]
    raw_text: str
    title: str = ""

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "title":       self.title,
            "headers":     self.headers,
            "rows":        self.rows,
            "raw_text":    self.raw_text,
            "row_count":   len(self.rows),
            "col_count":   len(self.headers),
        }

    def to_text(self) -> str:
        """Converts table to readable text for embedding."""
        lines = []
        if self.title:
            lines.append(self.title)
        if self.headers:
            lines.append(" | ".join(self.headers))
            lines.append("-" * 40)
        for row in self.rows:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)


@dataclass
class PDFParseResult:
    """
    Complete result of parsing a single PDF file.
    Returned by parse_pdf_with_metadata().
    """
    text: str
    pages: list[ParsedPage]
    tables: list[ExtractedTable]
    page_count: int
    file_size_bytes: int
    file_hash: str
    has_tables: bool
    is_scanned: bool
    ocr_used: bool
    avg_ocr_confidence: float
    parse_method: str             # "pypdf", "pymupdf", "paddleocr"
    duration_ms: float
    char_count: int
    word_count: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "page_count":        self.page_count,
            "file_size_bytes":   self.file_size_bytes,
            "file_hash":         self.file_hash,
            "has_tables":        self.has_tables,
            "is_scanned":        self.is_scanned,
            "ocr_used":          self.ocr_used,
            "avg_ocr_confidence": self.avg_ocr_confidence,
            "parse_method":      self.parse_method,
            "duration_ms":       self.duration_ms,
            "char_count":        self.char_count,
            "word_count":        self.word_count,
            "table_count":       len(self.tables),
        }


# ─────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────

class PDFParserError(Exception):
    """Base exception for all PDF parsing failures."""
    def __init__(
        self,
        message: str,
        file_path: str = "",
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.file_path     = file_path
        self.original_error = original_error


class PDFNotFoundError(PDFParserError):
    """Raised when the PDF file does not exist."""
    pass


class PDFCorruptError(PDFParserError):
    """Raised when the PDF is corrupt or unreadable."""
    pass


class PDFEmptyError(PDFParserError):
    """Raised when no text can be extracted from the PDF."""
    pass


class OCRError(PDFParserError):
    """Raised when OCR processing fails."""
    pass


# ─────────────────────────────────────────────
# PDF PARSER CLASS
# ─────────────────────────────────────────────

class PDFParser:
    """
    Multi-strategy PDF parser for financial documents.

    Strategy waterfall:
        1. pypdf     → Fast, works for most digital PDFs
        2. PyMuPDF   → Better layout, table-aware extraction
        3. PaddleOCR → Scanned/image PDFs (slower, GPU-accelerated)

    Usage:
        from backend.ingestion.pdf_parser import pdf_parser

        # Simple text extraction
        text = pdf_parser.parse_pdf("HDFC_Annual_Report_2026.pdf")

        # Full metadata + tables
        result = pdf_parser.parse_pdf_with_metadata("report.pdf")
        print(result.page_count, result.has_tables)

        # Tables only
        tables = pdf_parser.extract_tables("report.pdf")
        for table in tables:
            print(table.headers, table.rows[:3])
    """

    def __init__(self):
        self._ocr_model = None
        self._ocr_loaded: bool = False
        self._total_parsed: int = 0
        self._total_pages: int = 0
        self._total_ocr_calls: int = 0
        self._parse_time_ms: float = 0.0

    # ─────────────────────────────────────────
    # PRIMARY: parse_pdf
    # ─────────────────────────────────────────

    def parse_pdf(
        self,
        file_path: str,
        max_pages: int = MAX_PAGES,
    ) -> str:
        """
        Extracts clean text from a PDF file.
        Automatically chooses best parsing strategy.

        Args:
            file_path: Absolute or relative path to PDF file
            max_pages: Maximum pages to parse (default 500)

        Returns:
            Clean text string with all pages joined

        Raises:
            PDFNotFoundError, PDFCorruptError, PDFEmptyError

        Examples:
            text = pdf_parser.parse_pdf("reports/HDFC_AR_2026.pdf")
        """
        result = self.parse_pdf_with_metadata(
            file_path=file_path,
            max_pages=max_pages,
        )
        return result.text

    # ─────────────────────────────────────────
    # FULL: parse_pdf_with_metadata
    # ─────────────────────────────────────────

    def parse_pdf_with_metadata(
        self,
        file_path: str,
        max_pages: int = MAX_PAGES,
        extract_tables: bool = True,
        metadata: Optional[dict] = None,
    ) -> PDFParseResult:
        """
        Parses PDF and returns full result with text, tables, and metadata.

        Args:
            file_path:      Path to PDF file
            max_pages:      Maximum pages to process
            extract_tables: Whether to detect and extract tables
            metadata:       Additional metadata to attach to result

        Returns:
            PDFParseResult with all extracted content

        Raises:
            PDFNotFoundError, PDFCorruptError, PDFEmptyError

        Examples:
            result = pdf_parser.parse_pdf_with_metadata("rbi_circular.pdf")
            print(result.page_count)    # 12
            print(result.has_tables)    # True
            print(len(result.tables))   # 3
        """
        path = Path(file_path)
        _validate_file(path)

        parse_start = time.perf_counter()
        file_size   = path.stat().st_size
        file_hash   = _compute_file_hash(path)

        logger.info(
            f"[PDF PARSER] Parsing | "
            f"file={path.name} | "
            f"size={file_size/1024:.1f}KB"
        )

        # ── Strategy 1: PyMuPDF (preferred — best layout handling) ────────────
        pages, method, ocr_used = [], "pymupdf", False
        try:
            pages = self._parse_with_pymupdf(path, max_pages)
        except ImportError:
            logger.debug("[PDF PARSER] PyMuPDF not available — trying pypdf")
            try:
                pages = self._parse_with_pypdf(path, max_pages)
                method = "pypdf"
            except ImportError:
                logger.warning("[PDF PARSER] pypdf not available — trying PaddleOCR")
                pages, ocr_used = self._parse_with_ocr(path, max_pages), True
                method = "paddleocr"
        except Exception as e:
            logger.warning(f"[PDF PARSER] PyMuPDF failed ({e}) — falling back to pypdf")
            try:
                pages = self._parse_with_pypdf(path, max_pages)
                method = "pypdf"
            except Exception as e2:
                raise PDFCorruptError(
                    f"All parse strategies failed: {e2}",
                    file_path=str(path),
                    original_error=e2,
                )

        # ── Detect scanned pages and trigger OCR ──────────────────────────────
        scanned_pages = [p for p in pages if p.is_scanned]
        if scanned_pages and not ocr_used:
            logger.info(
                f"[PDF PARSER] {len(scanned_pages)} scanned pages detected — "
                f"applying OCR fallback"
            )
            pages = self._apply_ocr_to_scanned(path, pages)
            if any(p.ocr_used_flag for p in pages if hasattr(p, 'ocr_used_flag')):
                ocr_used = True
                method = f"{method}+paddleocr"

        if not pages:
            raise PDFEmptyError(
                f"No text extracted from '{path.name}'. "
                f"File may be corrupt or image-only without OCR.",
                file_path=str(path),
            )

        # ── Remove headers/footers ────────────────────────────────────────────
        pages = _remove_headers_footers(pages)

        # ── Extract tables ────────────────────────────────────────────────────
        tables: list[ExtractedTable] = []
        if extract_tables:
            try:
                tables = self.extract_tables(str(path))
            except Exception as e:
                logger.warning(f"[PDF PARSER] Table extraction failed (non-fatal): {e}")

        # ── Assemble final text ───────────────────────────────────────────────
        full_text = _assemble_text(pages)
        full_text = clean_text(full_text)

        if not full_text.strip():
            raise PDFEmptyError(
                f"PDF '{path.name}' produced empty text after cleaning.",
                file_path=str(path),
            )

        # ── Compute stats ─────────────────────────────────────────────────────
        duration_ms = (time.perf_counter() - parse_start) * 1000
        is_scanned  = len(scanned_pages) > len(pages) * 0.5   # >50% scanned
        avg_conf    = (
            sum(p.ocr_confidence for p in pages) / len(pages)
            if pages else 1.0
        )
        char_count  = len(full_text)
        word_count  = len(full_text.split())

        self._total_parsed += 1
        self._total_pages  += len(pages)
        self._parse_time_ms += duration_ms

        logger.info(
            f"[PDF PARSER] Complete | "
            f"file={path.name} | "
            f"pages={len(pages)} | "
            f"chars={char_count:,} | "
            f"tables={len(tables)} | "
            f"method={method} | "
            f"ocr={ocr_used} | "
            f"{duration_ms:.0f}ms"
        )
        log_metric("pdf_parse_duration_ms", duration_ms, unit="ms")
        log_metric("pdf_page_count", len(pages), unit="count")

        if ocr_used:
            log_metric("pdf_ocr_confidence", avg_conf * 100, unit="pct")
            if avg_conf < OCR_ACCURACY_TARGET:
                logger.warning(
                    f"[PDF PARSER] OCR accuracy {avg_conf:.1%} "
                    f"below target {OCR_ACCURACY_TARGET:.0%} | "
                    f"file={path.name}"
                )

        return PDFParseResult(
            text             = full_text,
            pages            = pages,
            tables           = tables,
            page_count       = len(pages),
            file_size_bytes  = file_size,
            file_hash        = file_hash,
            has_tables       = len(tables) > 0,
            is_scanned       = is_scanned,
            ocr_used         = ocr_used,
            avg_ocr_confidence = round(avg_conf, 4),
            parse_method     = method,
            duration_ms      = round(duration_ms, 1),
            char_count       = char_count,
            word_count       = word_count,
            metadata         = metadata or {},
        )

    # ─────────────────────────────────────────
    # TABLE EXTRACTION
    # ─────────────────────────────────────────

    def extract_tables(self, file_path: str) -> list[ExtractedTable]:
        """
        Detects and extracts financial tables from a PDF.
        Uses PyMuPDF's block structure analysis.

        Args:
            file_path: Path to PDF file

        Returns:
            List of ExtractedTable objects

        Examples:
            tables = pdf_parser.extract_tables("HDFC_AR_2026.pdf")
            for t in tables:
                print(t.headers)
                print(t.rows[:3])
                print(t.to_text())
        """
        path = Path(file_path)
        _validate_file(path)

        tables: list[ExtractedTable] = []

        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning(
                "[PDF PARSER] PyMuPDF not available — "
                "table extraction skipped"
            )
            return tables

        try:
            doc = fitz.open(str(path))
            for page_num, page in enumerate(doc, 1):
                if page_num > MAX_PAGES:
                    break
                page_tables = _extract_page_tables_pymupdf(page, page_num)
                tables.extend(page_tables)
            doc.close()

            logger.info(
                f"[PDF PARSER] Tables extracted | "
                f"file={path.name} | "
                f"count={len(tables)}"
            )
        except Exception as e:
            logger.warning(
                f"[PDF PARSER] Table extraction error | "
                f"file={path.name} | {e}"
            )

        return tables

    # ─────────────────────────────────────────
    # PRIVATE: PyMuPDF parsing
    # ─────────────────────────────────────────

    def _parse_with_pymupdf(
        self,
        path: Path,
        max_pages: int,
    ) -> list[ParsedPage]:
        """Primary parser — PyMuPDF for layout-aware extraction."""
        import fitz

        pages: list[ParsedPage] = []
        doc = fitz.open(str(path))

        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]

            # Extract text with layout preservation
            text = page.get_text("text")         # type: ignore[attr-defined]

            char_count = len(text.strip())
            is_scanned = char_count < MIN_CHARS_PER_PAGE_THRESHOLD

            pages.append(ParsedPage(
                page_number    = page_num + 1,
                text           = text,
                char_count     = char_count,
                word_count     = len(text.split()),
                is_scanned     = is_scanned,
                ocr_confidence = 1.0 if not is_scanned else 0.0,
            ))

        doc.close()
        logger.debug(
            f"[PDF PARSER] PyMuPDF | "
            f"pages={len(pages)} | "
            f"scanned={sum(1 for p in pages if p.is_scanned)}"
        )
        return pages

    # ─────────────────────────────────────────
    # PRIVATE: pypdf parsing
    # ─────────────────────────────────────────

    def _parse_with_pypdf(
        self,
        path: Path,
        max_pages: int,
    ) -> list[ParsedPage]:
        """Secondary parser — pypdf for clean digital PDFs."""
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader      # type: ignore[no-redef]

        pages: list[ParsedPage] = []
        reader = PdfReader(str(path))

        for page_num, page in enumerate(reader.pages[:max_pages]):
            text = page.extract_text() or ""
            char_count = len(text.strip())
            is_scanned = char_count < MIN_CHARS_PER_PAGE_THRESHOLD

            pages.append(ParsedPage(
                page_number    = page_num + 1,
                text           = text,
                char_count     = char_count,
                word_count     = len(text.split()),
                is_scanned     = is_scanned,
                ocr_confidence = 1.0 if not is_scanned else 0.0,
            ))

        logger.debug(
            f"[PDF PARSER] pypdf | "
            f"pages={len(pages)} | "
            f"scanned={sum(1 for p in pages if p.is_scanned)}"
        )
        return pages

    # ─────────────────────────────────────────
    # PRIVATE: PaddleOCR parsing
    # ─────────────────────────────────────────

    def _parse_with_ocr(
        self,
        path: Path,
        max_pages: int,
    ) -> list[ParsedPage]:
        """
        OCR fallback using PaddleOCR for scanned PDFs.
        Converts pages to images, then runs OCR on each.
        """
        ocr = self._get_ocr_model()
        pages: list[ParsedPage] = []

        try:
            import fitz
            doc = fitz.open(str(path))
            self._total_ocr_calls += 1

            for page_num in range(min(len(doc), max_pages)):
                page = doc[page_num]

                # Render page as image at 2x resolution for better OCR
                mat  = fitz.Matrix(2.0, 2.0)
                clip = page.get_pixmap(matrix=mat)  # type: ignore[attr-defined]

                img_bytes = clip.tobytes("png")
                page_text, confidence = self._ocr_image(ocr, img_bytes)

                pages.append(ParsedPage(
                    page_number    = page_num + 1,
                    text           = page_text,
                    char_count     = len(page_text),
                    word_count     = len(page_text.split()),
                    is_scanned     = True,
                    ocr_confidence = confidence,
                ))

                logger.debug(
                    f"[PDF PARSER] OCR page {page_num+1} | "
                    f"chars={len(page_text)} | "
                    f"confidence={confidence:.2f}"
                )
            doc.close()

        except Exception as e:
            raise OCRError(
                f"PaddleOCR processing failed: {e}",
                file_path=str(path),
                original_error=e,
            )

        return pages

    def _apply_ocr_to_scanned(
        self,
        path: Path,
        pages: list[ParsedPage],
    ) -> list[ParsedPage]:
        """Applies OCR selectively to scanned pages only."""
        try:
            ocr = self._get_ocr_model()
            import fitz
            doc = fitz.open(str(path))

            for i, parsed_page in enumerate(pages):
                if not parsed_page.is_scanned:
                    continue

                page_idx = parsed_page.page_number - 1
                if page_idx >= len(doc):
                    continue

                page = doc[page_idx]
                mat  = fitz.Matrix(2.0, 2.0)
                clip = page.get_pixmap(matrix=mat)  # type: ignore[attr-defined]
                img_bytes = clip.tobytes("png")

                page_text, confidence = self._ocr_image(ocr, img_bytes)
                pages[i].text           = page_text
                pages[i].char_count     = len(page_text)
                pages[i].word_count     = len(page_text.split())
                pages[i].ocr_confidence = confidence

            doc.close()
        except Exception as e:
            logger.warning(f"[PDF PARSER] Selective OCR failed (non-fatal): {e}")

        return pages

    def _get_ocr_model(self):
        """Lazy-loads PaddleOCR model on first OCR call."""
        if self._ocr_loaded and self._ocr_model is not None:
            return self._ocr_model

        try:
            from paddleocr import PaddleOCR
            self._ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                use_gpu=_is_gpu_available(),
            )
            self._ocr_loaded = True
            logger.info(
                f"[PDF PARSER] PaddleOCR loaded | "
                f"gpu={_is_gpu_available()}"
            )
            return self._ocr_model
        except ImportError:
            raise OCRError(
                "PaddleOCR not installed. Run: pip install paddleocr paddlepaddle"
            )

    def _ocr_image(
        self,
        ocr_model,
        img_bytes: bytes,
    ) -> tuple[str, float]:
        """
        Runs PaddleOCR on an image and returns text + confidence.

        Returns:
            Tuple of (extracted_text, avg_confidence_score)
        """
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)

        result = ocr_model.ocr(img_array, cls=True)

        if not result or not result[0]:
            return "", 0.0

        lines: list[str] = []
        confidences: list[float] = []

        for line in result[0]:
            if line and len(line) >= 2:
                text_conf = line[1]
                if text_conf and len(text_conf) >= 2:
                    text = str(text_conf[0])
                    conf = float(text_conf[1])
                    lines.append(text)
                    confidences.append(conf)

        full_text   = " ".join(lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return full_text, avg_confidence

    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        avg_time = (
            round(self._parse_time_ms / self._total_parsed, 1)
            if self._total_parsed > 0 else 0.0
        )
        return {
            "total_parsed":      self._total_parsed,
            "total_pages":       self._total_pages,
            "total_ocr_calls":   self._total_ocr_calls,
            "total_parse_ms":    round(self._parse_time_ms, 1),
            "avg_parse_ms":      avg_time,
        }


# ─────────────────────────────────────────────
# STANDALONE: clean_text
# ─────────────────────────────────────────────

def clean_text(raw_text: str) -> str:
    """
    Cleans raw extracted PDF text for ingestion.

    Removes:
    - Page numbers (standalone digits, "Page N of M")
    - PDF headers/footers artifacts
    - Watermarks and boilerplate legal text
    - Excessive whitespace (3+ blank lines → 1)
    - Special control characters
    - Hyphenated word breaks from PDF layout
    - Repeated punctuation
    - Non-printable characters

    Preserves:
    - Financial figures and percentages
    - Table structures (where possible)
    - Paragraph breaks
    - Sentence endings

    Args:
        raw_text: Raw text string from PDF extraction

    Returns:
        Cleaned text string

    Examples:
        text = clean_text(raw_pdf_output)
    """
    if not raw_text:
        return ""

    text = raw_text

    # ── Remove non-printable / control characters ─────────────────────────────
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)

    # ── Normalize unicode dashes and quotes ───────────────────────────────────
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2022', '-')    # Bullet points → dash
    text = text.replace('\u00a0', ' ')    # Non-breaking space → regular space

    # ── Fix hyphenated word breaks (PDF layout artifact) ─────────────────────
    # "bank-\ning" → "banking"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # ── Remove boilerplate legal text ─────────────────────────────────────────
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE | re.MULTILINE)

    # ── Remove standalone page numbers ───────────────────────────────────────
    text = re.sub(r'(?m)^\s*\d{1,4}\s*$', '', text)

    # ── Remove "Page N of M" / "Page N" patterns ─────────────────────────────
    text = re.sub(
        r'\bpage\s+\d+\s*(?:of\s+\d+)?\b', '',
        text, flags=re.IGNORECASE
    )

    # ── Remove PDF artifact characters ───────────────────────────────────────
    text = re.sub(r'[•◦▪▫►◄→←↑↓]', ' ', text)
    text = re.sub(r'[_]{3,}', ' ', text)           # Long underlines
    text = re.sub(r'[-]{3,}', ' ', text)           # Long dashes (not financial)
    text = re.sub(r'[=]{3,}', ' ', text)           # Section separators
    text = re.sub(r'[*]{2,}', ' ', text)           # Asterisk markers

    # ── Fix OCR spacing artifacts ─────────────────────────────────────────────
    # "NI M" → "NIM", "H D F C" → "HDFC"
    text = re.sub(r'\b([A-Z])\s([A-Z])\s([A-Z])\s([A-Z])\b', r'\1\2\3\4', text)
    text = re.sub(r'\b([A-Z])\s([A-Z])\s([A-Z])\b', r'\1\2\3', text)

    # ── Normalize whitespace ──────────────────────────────────────────────────
    text = re.sub(r'[ \t]+', ' ', text)               # Multiple spaces → single
    text = re.sub(r'\n{3,}', '\n\n', text)            # 3+ newlines → 2
    text = re.sub(r' \n', '\n', text)                 # Space before newline
    text = re.sub(r'\n ', '\n', text)                 # Space after newline

    # ── Remove very short orphan lines (likely header/footer artifacts) ───────
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep: blank lines, financial data lines, normal sentences
        if not stripped:
            cleaned_lines.append('')
            continue
        # Drop: very short non-numeric, non-sentence lines (likely headers)
        if len(stripped) < 3 and not re.match(r'^\d', stripped):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # ── Final whitespace normalization ────────────────────────────────────────
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)    # Final pass on triple newlines

    return text


# ─────────────────────────────────────────────
# STANDALONE: extract_metadata_from_filename
# ─────────────────────────────────────────────

def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extracts document metadata from filename patterns.
    Handles common Indian financial document naming conventions.

    Detected fields:
        company:     HDFC, ICICI, TCS, Infosys, etc.
        year:        2020–2030
        quarter:     Q1, Q2, Q3, Q4
        doc_type:    annual_report, earnings_transcript,
                     broker_research, rbi, sebi, etc.
        financial_year: FY24, FY25, FY26

    Args:
        filename: Filename string (with or without path/extension)

    Returns:
        Dict with detected metadata fields

    Examples:
        extract_metadata_from_filename("HDFC_Bank_Annual_Report_2026.pdf")
        # → {'company': 'HDFC Bank', 'year': '2026', 'doc_type': 'annual_report'}

        extract_metadata_from_filename("ICICI_Q3_FY26_Earnings_Transcript.pdf")
        # → {'company': 'ICICI Bank', 'quarter': 'Q3', 'financial_year': 'FY26', ...}

        extract_metadata_from_filename("RBI_Monetary_Policy_Statement_Feb_2026.pdf")
        # → {'doc_type': 'rbi', 'authority': 'RBI', 'year': '2026'}
    """
    # Strip path and extension
    name = Path(filename).stem
    name_lower = name.lower()
    name_normalized = re.sub(r'[_\-\s]+', ' ', name_lower).strip()

    metadata: dict = {
        "company":        None,
        "year":           None,
        "quarter":        None,
        "financial_year": None,
        "doc_type":       "unknown",
        "authority":      1.0,
        "source":         name,
    }

    # ── Year detection ────────────────────────────────────────────────────────
    year_match = re.search(r'\b(20[12]\d)\b', name)
    if year_match:
        metadata["year"] = year_match.group(1)

    # ── Financial year detection (FY24, FY25, FY2026) ────────────────────────
    fy_match = re.search(r'\bfy\s*(\d{2,4})\b', name_lower)
    if fy_match:
        fy_raw = fy_match.group(1)
        if len(fy_raw) == 2:
            fy_raw = "20" + fy_raw
        metadata["financial_year"] = f"FY{fy_raw}"

    # ── Quarter detection ─────────────────────────────────────────────────────
    quarter_match = re.search(r'\b(q[1-4])\b', name_lower)
    if quarter_match:
        metadata["quarter"] = quarter_match.group(1).upper()

    # ── Document type detection ───────────────────────────────────────────────
    doc_type_patterns: list[tuple[str, str, float]] = [
        # Pattern, doc_type, authority_weight
        (r'\b(annual[\s_]?report|ar\b)',             "annual_report",        0.75),
        (r'\b(earnings?[\s_]?call|transcript)',       "earnings_transcript",  0.65),
        (r'\b(investor[\s_]?presentation)',           "annual_report",        0.70),
        (r'\b(rbi|reserve[\s_]?bank)',                "rbi",                  1.00),
        (r'\b(sebi[\s_]?circular|sebi)',              "sebi",                 1.00),
        (r'\b(mca|ministry[\s_]?of[\s_]?corporate)', "mca",                  0.95),
        (r'\b(budget|union[\s_]?budget)',             "regulatory",           0.90),
        (r'\b(monetary[\s_]?policy|mpc[\s_]?statement)', "rbi",              1.00),
        (r'\b(broker[\s_]?report|research[\s_]?note|initiating[\s_]?coverage)', "broker_research", 0.40),
        (r'\b(concall|conference[\s_]?call)',         "earnings_transcript",  0.65),
        (r'\b(drhp|red[\s_]?herring)',                "annual_report",        0.70),
        (r'\b(prospectus)',                           "annual_report",        0.75),
        (r'\b(notice|circular|notification)',         "regulatory",           0.85),
        (r'\b(press[\s_]?release)',                   "news",                 0.30),
    ]
    for pattern, doc_type, authority in doc_type_patterns:
        if re.search(pattern, name_lower):
            metadata["doc_type"]  = doc_type
            metadata["authority"] = authority
            break

    # ── Company detection ─────────────────────────────────────────────────────
    company_patterns: list[tuple[str, str]] = [
        # Pattern,  Clean name
        (r'\bhdfc[\s_]?bank\b',         "HDFC Bank"),
        (r'\bhdfc[\s_]?ltd\b|\bhdfc\b', "HDFC Ltd"),
        (r'\bicici[\s_]?bank\b',        "ICICI Bank"),
        (r'\bicici[\s_]?prudential\b',  "ICICI Prudential"),
        (r'\bsbi\b|state[\s_]?bank',    "SBI"),
        (r'\baxis[\s_]?bank\b',         "Axis Bank"),
        (r'\bkotak\b',                  "Kotak Mahindra Bank"),
        (r'\bindusind\b',               "IndusInd Bank"),
        (r'\bbandhan\b',                "Bandhan Bank"),
        (r'\bidfc[\s_]?first\b',        "IDFC First Bank"),
        (r'\btcs\b|tata[\s_]?consultancy', "TCS"),
        (r'\binfosys\b',                "Infosys"),
        (r'\bwipro\b',                  "Wipro"),
        (r'\bhcl[\s_]?tech\b',          "HCL Technologies"),
        (r'\btech[\s_]?mahindra\b',     "Tech Mahindra"),
        (r'\breliance\b',               "Reliance Industries"),
        (r'\bjio\b',                    "Reliance Jio"),
        (r'\bbajaj[\s_]?finance\b',     "Bajaj Finance"),
        (r'\bbajaj[\s_]?finserv\b',     "Bajaj Finserv"),
        (r'\bmuthoot\b',                "Muthoot Finance"),
        (r'\bmanappuram\b',             "Manappuram Finance"),
        (r'\blic\b|life[\s_]?insurance[\s_]?corp', "LIC"),
        (r'\badani\b',                  "Adani Group"),
        (r'\btata[\s_]?motors\b',       "Tata Motors"),
        (r'\bmahindra\b',               "Mahindra & Mahindra"),
        (r'\bmaruti\b',                 "Maruti Suzuki"),
        (r'\bzomato\b',                 "Zomato"),
        (r'\bpaytm\b',                  "Paytm"),
        (r'\bnsdl\b',                   "NSDL"),
        (r'\bcdsl\b',                   "CDSL"),
        (r'\bnse\b',                    "NSE"),
        (r'\bbse\b',                    "BSE"),
    ]
    for pattern, company_name in company_patterns:
        if re.search(pattern, name_lower):
            metadata["company"] = company_name
            break

    # ── Authority source detection (for regulatory docs) ──────────────────────
    if re.search(r'\brbi\b|reserve[\s_]?bank', name_lower):
        metadata["authority_name"] = "RBI"
    elif re.search(r'\bsebi\b', name_lower):
        metadata["authority_name"] = "SEBI"
    elif re.search(r'\bmca\b|ministry[\s_]?corporate', name_lower):
        metadata["authority_name"] = "MCA"
    elif re.search(r'\bpfrda\b', name_lower):
        metadata["authority_name"] = "PFRDA"
    elif re.search(r'\birdai\b', name_lower):
        metadata["authority_name"] = "IRDAI"

    logger.debug(
        f"[PDF PARSER] Metadata from filename | "
        f"file={filename} | "
        f"detected={metadata}"
    )
    return metadata


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _validate_file(path: Path) -> None:
    """Validates file exists, is readable, and is a PDF."""
    if not path.exists():
        raise PDFNotFoundError(
            f"PDF file not found: '{path}'",
            file_path=str(path),
        )
    if not path.is_file():
        raise PDFParserError(
            f"Path is not a file: '{path}'",
            file_path=str(path),
        )
    if path.suffix.lower() not in {".pdf"}:
        raise PDFParserError(
            f"File is not a PDF: '{path.name}'. "
            f"Only .pdf files are supported.",
            file_path=str(path),
        )
    if path.stat().st_size == 0:
        raise PDFCorruptError(
            f"PDF file is empty (0 bytes): '{path.name}'",
            file_path=str(path),
        )


def _compute_file_hash(path: Path) -> str:
    """Computes MD5 hash of file for deduplication."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _assemble_text(pages: list[ParsedPage]) -> str:
    """Joins page texts with double newlines between pages."""
    parts = []
    for page in pages:
        text = page.text.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _remove_headers_footers(pages: list[ParsedPage]) -> list[ParsedPage]:
    """
    Detects repeated lines across pages (likely headers/footers)
    and removes them from all pages.
    """
    if len(pages) < 3:
        return pages

    # Count line occurrences across all pages
    line_counts: dict[str, int] = {}
    for page in pages:
        seen_on_page: set[str] = set()
        for line in page.text.split('\n'):
            stripped = line.strip()
            if len(stripped) < 5:
                continue
            if stripped not in seen_on_page:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
                seen_on_page.add(stripped)

    # Lines appearing on many pages = header/footer
    repeated = {
        line for line, count in line_counts.items()
        if count >= HEADER_FOOTER_MAX_OCCURRENCES
    }

    if not repeated:
        return pages

    logger.debug(
        f"[PDF PARSER] Removing {len(repeated)} repeated header/footer lines"
    )

    for page in pages:
        cleaned_lines = [
            line for line in page.text.split('\n')
            if line.strip() not in repeated
        ]
        page.text      = '\n'.join(cleaned_lines)
        page.char_count = len(page.text)
        page.word_count = len(page.text.split())

    return pages


def _extract_page_tables_pymupdf(
    page,
    page_number: int,
) -> list[ExtractedTable]:
    """
    Extracts tables from a single PyMuPDF page using block analysis.
    Identifies tables by detecting grid-like text block arrangements.
    """
    tables: list[ExtractedTable] = []

    try:
        # Use PyMuPDF's built-in table finder (available in fitz >= 1.23)
        import fitz
        tab_finder = page.find_tables()
        if tab_finder and tab_finder.tables:
            for i, tab in enumerate(tab_finder.tables):
                try:
                    df = tab.to_pandas()
                    headers = [str(c) for c in df.columns.tolist()]
                    rows    = df.values.tolist()
                    rows    = [[str(cell) for cell in row] for row in rows]

                    # Only keep if it looks like a financial table
                    if _is_financial_table(headers, rows):
                        tables.append(ExtractedTable(
                            page_number = page_number,
                            headers     = headers,
                            rows        = rows,
                            raw_text    = tab.extract() or "",
                            title       = f"Table {i+1} (Page {page_number})",
                        ))
                except Exception:
                    pass
    except (AttributeError, ImportError):
        # Older PyMuPDF or pandas not installed — skip table extraction
        pass

    return tables


def _is_financial_table(headers: list, rows: list) -> bool:
    """
    Heuristic: checks if a detected table contains financial data.
    Avoids extracting navigation menus, TOCs, and decorative boxes.
    """
    if not headers or len(headers) < 2:
        return False
    if not rows or len(rows) < 2:
        return False

    # Check if any header matches financial table keywords
    headers_lower = " ".join(str(h) for h in headers).lower()
    if any(kw in headers_lower for kw in TABLE_HEADER_KEYWORDS):
        return True

    # Check if any cell contains numbers (financial tables have numbers)
    for row in rows[:3]:
        for cell in row:
            if re.search(r'\d+\.?\d*\s*%?', str(cell)):
                return True

    return False


def _is_gpu_available() -> bool:
    """Checks if GPU is available for OCR acceleration."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        import paddle
        return paddle.device.get_device().startswith("gpu")
    except Exception:
        return False


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

pdf_parser = PDFParser()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "pdf_parser",
    "PDFParser",
    "ParsedPage",
    "ExtractedTable",
    "PDFParseResult",
    "clean_text",
    "extract_metadata_from_filename",
    "PDFParserError",
    "PDFNotFoundError",
    "PDFCorruptError",
    "PDFEmptyError",
    "OCRError",
]