# backend/ingestion/chunker.py
"""
FinThesisGuard AI — Document Chunker
ArkAngel Financial Solutions

Splits cleaned financial documents into semantically coherent,
overlapping chunks for embedding and Pinecone upsert.

Chunking strategy (in priority order):
    1. Section boundary detection  — split at "Risk Factors", "Financial
                                     Highlights" etc. (natural boundaries)
    2. Sentence boundary awareness — never cut mid-sentence
    3. Table boundary awareness    — never cut mid-table
    4. Sliding window overlap      — configurable overlap for context continuity
    5. Adaptive chunk size         — different sizes per document type

Deduplication:
    Each chunk gets an MD5 hash. Duplicate chunks (same hash) are
    skipped — prevents re-ingesting the same content twice.
"""

import re
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from backend.config import settings
from backend.utils.logger import logger, log_metric
from backend.ingestion.pdf_parser import clean_text


# ─────────────────────────────────────────────
# ADAPTIVE CHUNK SIZE CONFIG
# Tuned per document type for optimal retrieval
# ─────────────────────────────────────────────

CHUNK_SIZES: dict[str, int] = {
    "annual_report":        1200,   # Dense disclosures — larger chunks
    "earnings_transcript":  1000,   # Conversational — medium
    "rbi":                   800,   # Regulatory — precise, medium
    "sebi":                  800,   # Regulatory — precise, medium
    "mca":                   800,   # Regulatory
    "tax_law":               900,   # Legal text — medium-large
    "broker_research":       700,   # Analysis — medium
    "news":                  600,   # Short form — smaller chunks
    "blog":                  500,   # Informal — smallest
    "unknown":              1000,   # Default
}

OVERLAP_SIZES: dict[str, int] = {
    "annual_report":        200,
    "earnings_transcript":  150,
    "rbi":                  120,
    "sebi":                 120,
    "mca":                  120,
    "tax_law":              150,
    "broker_research":      100,
    "news":                  80,
    "blog":                  60,
    "unknown":              150,
}

# Hard limits
MIN_CHUNK_SIZE:  int = 200
MAX_CHUNK_SIZE:  int = 2000
MIN_WORDS_PER_CHUNK: int = 20    # Chunks below this are filtered out
MIN_CHARS_PER_CHUNK: int = 80    # Absolute minimum — very short chunks

# Default values when doc_type is not specified
DEFAULT_CHUNK_SIZE:   int = 1000
DEFAULT_OVERLAP_SIZE: int = 150


# ─────────────────────────────────────────────
# SECTION HEADER PATTERNS
# Financial document section headers used as natural chunk boundaries
# ─────────────────────────────────────────────

SECTION_HEADER_PATTERNS: list[str] = [
    # Financial results
    r"(?:financial\s+)?highlights?",
    r"financial\s+(?:results?|performance|summary)",
    r"key\s+(?:financial\s+)?(?:metrics?|indicators?|ratios?|highlights?)",
    r"profit\s+(?:and\s+loss|&\s+loss)",
    r"balance\s+sheet",
    r"cash\s+flow\s+statement",
    r"statement\s+of\s+(?:profit|income)",
    r"income\s+statement",
    r"notes?\s+to\s+(?:the\s+)?(?:financial\s+)?accounts?",

    # Business sections
    r"management(?:'s?)?\s+discussion\s+(?:and\s+analysis|&\s+analysis)",
    r"md\s*&\s*a",
    r"business\s+(?:overview|review|summary|description|segments?)",
    r"operating\s+(?:performance|review|highlights?)",
    r"segment(?:al)?\s+(?:performance|results?|review)",
    r"geographic\s+(?:performance|breakdown)",

    # Risk sections
    r"risk\s+(?:factors?|management|framework|governance)",
    r"credit\s+risk",
    r"market\s+risk",
    r"liquidity\s+risk",
    r"operational\s+risk",
    r"concentration\s+risk",

    # Banking specific
    r"net\s+interest\s+(?:margin|income)",
    r"(?:gross|net)\s+npa",
    r"capital\s+adequacy",
    r"asset\s+quality",
    r"casa\s+(?:ratio|deposits?)",
    r"loan\s+(?:book|portfolio|growth)",
    r"deposit\s+(?:growth|mix|base)",
    r"provisioning\s+(?:coverage|policy)",

    # Corporate governance
    r"corporate\s+governance",
    r"board\s+of\s+directors",
    r"director(?:'s?)?\s+report",
    r"auditor(?:'s?)?\s+report",
    r"statutory\s+(?:report|compliance)",
    r"related\s+party\s+transactions?",
    r"dividend\s+(?:policy|distribution)",

    # Forward-looking
    r"outlook\s+(?:and\s+guidance|for)?",
    r"guidance\s+(?:and\s+outlook)?",
    r"future\s+(?:plans?|prospects?|strategy)",
    r"strategic\s+(?:priorities?|direction|initiatives?)",
    r"growth\s+(?:strategy|drivers?|levers?)",

    # Regulatory
    r"regulatory\s+(?:compliance|framework|updates?|changes?)",
    r"(?:sebi|rbi|mca)\s+(?:guidelines?|circulars?|regulations?)",
    r"statutory\s+information",

    # Standard report sections
    r"executive\s+summary",
    r"chairman(?:'s?)?\s+(?:message|letter|statement)",
    r"ceo(?:'s?)?\s+(?:message|letter|statement)",
    r"(?:managing\s+)?director(?:'s?)?\s+(?:message|statement)",
    r"about\s+(?:the\s+)?(?:company|bank|us)",
    r"company\s+overview",
    r"products?\s+(?:and\s+)?services?",
    r"distribution\s+network",
    r"human\s+resources?|people\s+initiatives?",
    r"technology\s+(?:and\s+digital)?(?:\s+initiatives?)?",
    r"esg|sustainability|corporate\s+social\s+responsibility|csr",
    r"awards?\s+(?:and\s+)?recognition",
    r"glossary|abbreviations?|definitions?",
    r"annexure|appendix|schedule",
]

# Compiled pattern for fast matching
_SECTION_HEADER_RE = re.compile(
    r'(?im)^(?:\s*)(?:\d+[\.\d]*\s+)?('
    + '|'.join(f'(?:{p})' for p in SECTION_HEADER_PATTERNS)
    + r')(?:\s*:)?\s*$'
)

# Table block detection (for boundary preservation)
_TABLE_START_RE = re.compile(
    r'(?:particulars?|description|item)\s*\|',
    re.IGNORECASE,
)
_TABLE_ROW_RE = re.compile(r'^.+\|.+$', re.MULTILINE)


# ─────────────────────────────────────────────
# CHUNK DATACLASS
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single text chunk ready for embedding and Pinecone upsert.

    Fields:
        id:          Deterministic ID: "{doc_id}_chunk_{index}"
        text:        Chunk text content
        metadata:    Dict with source, date, source_type, company,
                     authority, chunk_index, doc_id, section, etc.
        doc_id:      Parent document ID
        chunk_index: Position within document (0-based)
        char_count:  Length in characters
        word_count:  Length in words
        hash:        MD5 hash for deduplication
        section:     Section header this chunk belongs to (if detected)
        embedding:   384-dim vector (populated after encoding, None initially)
    """
    text:        str
    metadata:    dict
    doc_id:      str
    chunk_index: int
    id:          str = field(default="")
    char_count:  int = field(default=0)
    word_count:  int = field(default=0)
    hash:        str = field(default="")
    section:     str = field(default="")
    embedding:   Optional[list[float]] = field(default=None, repr=False)

    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())
        self.hash       = _md5(self.text)
        if not self.id:
            self.id = f"{self.doc_id}_chunk_{self.chunk_index}"

    def to_pinecone_dict(self) -> dict:
        """
        Converts chunk to Pinecone-ready upsert format.
        Requires embedding to be populated first.
        """
        if self.embedding is None:
            raise ValueError(
                f"Chunk '{self.id}' has no embedding. "
                "Call embedder.encode_batch() before upsert."
            )
        return {
            "id":       self.id,
            "values":   self.embedding,
            "metadata": {
                **self.metadata,
                "text":        self.text,
                "chunk_index": self.chunk_index,
                "doc_id":      self.doc_id,
                "section":     self.section,
                "char_count":  self.char_count,
                "word_count":  self.word_count,
            },
        }

    def to_dict(self) -> dict:
        """Full dict representation (without embedding for serialization)."""
        return {
            "id":          self.id,
            "doc_id":      self.doc_id,
            "chunk_index": self.chunk_index,
            "text":        self.text,
            "section":     self.section,
            "char_count":  self.char_count,
            "word_count":  self.word_count,
            "hash":        self.hash,
            "metadata":    self.metadata,
            "has_embedding": self.embedding is not None,
        }


# ─────────────────────────────────────────────
# CHUNKER CLASS
# ─────────────────────────────────────────────

class Chunker:
    """
    Adaptive financial document chunker.

    Usage:
        from backend.ingestion.chunker import chunker

        # From raw text + metadata
        chunks = chunker.chunk_document(
            text=cleaned_text,
            metadata={
                "source":      "HDFC Bank Annual Report 2026",
                "source_type": "annual_report",
                "company":     "HDFC Bank",
                "date":        "2026-01-15",
                "authority":   0.75,
            },
            doc_id="hdfc_ar_2026",
        )

        # Just text chunks (no metadata)
        text_chunks = chunker.chunk_text(
            text=cleaned_text,
            chunk_size=1000,
            overlap=150,
        )

        # Convert to Pinecone-ready dicts (after embedding)
        pinecone_records = [c.to_pinecone_dict() for c in chunks]
    """

    def __init__(self):
        self._total_docs:   int = 0
        self._total_chunks: int = 0
        self._total_duplicates_skipped: int = 0
        self._chunk_time_ms: float = 0.0

    # ─────────────────────────────────────────
    # MAIN: chunk_document
    # ─────────────────────────────────────────

    def chunk_document(
        self,
        text: str,
        metadata: dict,
        doc_id: Optional[str] = None,
        clean: bool = True,
        deduplicate: bool = True,
    ) -> list[Chunk]:
        """
        Full pipeline: clean → section-split → chunk → deduplicate.

        Args:
            text:        Raw or cleaned document text
            metadata:    Document metadata dict (source, date, etc.)
            doc_id:      Unique document identifier (auto-generated if None)
            clean:       Run clean_text() before chunking (default True)
            deduplicate: Skip duplicate chunks via MD5 hash (default True)

        Returns:
            List of Chunk objects with metadata and deterministic IDs

        Raises:
            ValueError if text is empty after cleaning

        Examples:
            chunks = chunker.chunk_document(
                text=pdf_parser.parse_pdf("report.pdf"),
                metadata={
                    "source":      "HDFC Annual Report 2026",
                    "source_type": "annual_report",
                    "company":     "HDFC Bank",
                    "date":        "2026-01-15",
                    "authority":   0.75,
                },
                doc_id="hdfc_ar_2026",
            )
            len(chunks)          # → 80-200 depending on doc size
            chunks[0].id         # → "hdfc_ar_2026_chunk_0"
            chunks[0].section    # → "Financial Highlights"
        """
        chunk_start = time.perf_counter()

        if not doc_id:
            doc_id = _generate_doc_id(metadata)

        # ── Step 1: Clean text ────────────────────────────────────────────────
        working_text = clean_text(text) if clean else text

        if not working_text.strip():
            raise ValueError(
                f"Document '{doc_id}' produced empty text after cleaning."
            )

        # ── Step 2: Determine adaptive chunk size ─────────────────────────────
        source_type = metadata.get("source_type", "unknown")
        chunk_size  = adaptive_chunk_size(source_type)
        overlap     = OVERLAP_SIZES.get(source_type, DEFAULT_OVERLAP_SIZE)

        # ── Step 3: Detect section boundaries ────────────────────────────────
        sections = detect_section_headers(working_text)
        section_splits = _split_by_sections(working_text, sections)

        logger.debug(
            f"[CHUNKER] Sections detected | "
            f"doc={doc_id} | "
            f"sections={len(sections)} | "
            f"splits={len(section_splits)}"
        )

        # ── Step 4: Chunk each section ────────────────────────────────────────
        all_text_chunks: list[tuple[str, str]] = []  # (text, section_name)

        for section_name, section_text in section_splits:
            if not section_text.strip():
                continue
            text_chunks = self.chunk_text(
                text=section_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            for tc in text_chunks:
                all_text_chunks.append((tc, section_name))

        if not all_text_chunks:
            raise ValueError(
                f"No chunks produced for document '{doc_id}'. "
                f"Text may be too short or entirely boilerplate."
            )

        # ── Step 5: Build Chunk objects ───────────────────────────────────────
        chunks: list[Chunk] = []
        seen_hashes: set[str] = set()
        chunk_index = 0

        for text_chunk, section_name in all_text_chunks:
            chunk = Chunk(
                text        = text_chunk,
                metadata    = {**metadata, "source_type": source_type},
                doc_id      = doc_id,
                chunk_index = chunk_index,
                section     = section_name,
            )

            # Deduplication
            if deduplicate and chunk.hash in seen_hashes:
                self._total_duplicates_skipped += 1
                logger.debug(
                    f"[CHUNKER] Duplicate skipped | "
                    f"doc={doc_id} | chunk_index={chunk_index}"
                )
                continue

            seen_hashes.add(chunk.hash)
            chunks.append(chunk)
            chunk_index += 1

        # ── Step 6: Filter short/low-quality chunks ───────────────────────────
        chunks = self.filter_short_chunks(chunks, min_words=MIN_WORDS_PER_CHUNK)

        # ── Step 7: Re-assign sequential IDs after filtering ─────────────────
        chunks = self.assign_chunk_ids(chunks, doc_id)

        duration_ms = (time.perf_counter() - chunk_start) * 1000
        self._total_docs   += 1
        self._total_chunks += len(chunks)
        self._chunk_time_ms += duration_ms

        avg_chars = (
            sum(c.char_count for c in chunks) / len(chunks)
            if chunks else 0
        )
        logger.info(
            f"[CHUNKER] Complete | "
            f"doc={doc_id} | "
            f"source_type={source_type} | "
            f"chunk_size={chunk_size} | "
            f"overlap={overlap} | "
            f"total_chunks={len(chunks)} | "
            f"avg_chars={avg_chars:.0f} | "
            f"sections={len(sections)} | "
            f"deduped={self._total_duplicates_skipped} | "
            f"{duration_ms:.0f}ms"
        )
        log_metric("chunker_chunk_count", len(chunks), unit="count")
        log_metric("chunker_avg_chunk_chars", avg_chars, unit="chars")
        log_metric("chunker_duration_ms", duration_ms, unit="ms")

        return chunks

    # ─────────────────────────────────────────
    # CORE: chunk_text
    # ─────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP_SIZE,
    ) -> list[str]:
        """
        Splits text into overlapping chunks using sentence-boundary-aware
        sliding window. Never cuts mid-sentence or mid-table.

        Algorithm:
            1. Split text into semantic units (sentences/table rows/paragraphs)
            2. Greedily accumulate units until chunk_size reached
            3. Roll back to last sentence boundary if mid-sentence
            4. Slide forward by (chunk_size - overlap) for next chunk
            5. Preserve table blocks as atomic units (no splitting)

        Args:
            text:       Input text string
            chunk_size: Target chunk size in characters (default 1000)
            overlap:    Overlap in characters between adjacent chunks (default 150)

        Returns:
            List of text chunk strings

        Examples:
            chunks = chunker.chunk_text(
                text=long_document_text,
                chunk_size=1000,
                overlap=150,
            )
        """
        if not text or not text.strip():
            return []

        chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, MAX_CHUNK_SIZE))
        overlap    = max(0, min(overlap, chunk_size // 2))

        # ── Detect and isolate table blocks ───────────────────────────────────
        # Tables are wrapped as atomic units — never split across chunks
        text_with_markers, table_registry = _protect_tables(text)

        # ── Split into semantic units ──────────────────────────────────────────
        units = _split_into_units(text_with_markers)
        if not units:
            return []

        chunks: list[str] = []
        current_chars = 0
        current_units: list[str] = []

        i = 0
        while i < len(units):
            unit = units[i]
            unit_len = len(unit)

            # If single unit exceeds chunk_size, force-split it
            if unit_len > chunk_size and not _is_table_marker(unit):
                if current_units:
                    chunk_text_str = _restore_tables(
                        " ".join(current_units).strip(), table_registry
                    )
                    if chunk_text_str.strip():
                        chunks.append(chunk_text_str)
                    current_units = _overlap_units(current_units, overlap)
                    current_chars = sum(len(u) for u in current_units)

                # Force-split the oversized unit at sentence boundaries
                forced = _force_split_unit(unit, chunk_size)
                for part in forced:
                    restored = _restore_tables(part, table_registry)
                    if restored.strip():
                        chunks.append(restored)
                i += 1
                continue

            # Adding this unit would exceed chunk_size → emit current chunk
            if current_chars + unit_len > chunk_size and current_units:
                chunk_text_str = _restore_tables(
                    " ".join(current_units).strip(), table_registry
                )
                if chunk_text_str.strip():
                    chunks.append(chunk_text_str)

                # Keep overlap units for next chunk
                current_units = _overlap_units(current_units, overlap)
                current_chars = sum(len(u) for u in current_units)

            current_units.append(unit)
            current_chars += unit_len
            i += 1

        # Emit final chunk
        if current_units:
            chunk_text_str = _restore_tables(
                " ".join(current_units).strip(), table_registry
            )
            if chunk_text_str.strip():
                chunks.append(chunk_text_str)

        # Filter empty/too-short chunks
        chunks = [c for c in chunks if len(c.strip()) >= MIN_CHARS_PER_CHUNK]

        return chunks

    # ─────────────────────────────────────────
    # ASSIGN CHUNK IDs
    # ─────────────────────────────────────────

    def assign_chunk_ids(
        self,
        chunks: list[Chunk],
        doc_id: str,
    ) -> list[Chunk]:
        """
        Assigns deterministic sequential IDs to chunks.
        IDs follow format: "{doc_id}_chunk_{index}"

        Args:
            chunks: List of Chunk objects
            doc_id: Parent document ID

        Returns:
            Same chunks with updated IDs and chunk_index fields

        Examples:
            chunks = chunker.assign_chunk_ids(chunks, "hdfc_ar_2026")
            chunks[0].id  # → "hdfc_ar_2026_chunk_0"
            chunks[5].id  # → "hdfc_ar_2026_chunk_5"
        """
        for idx, chunk in enumerate(chunks):
            chunk.chunk_index = idx
            chunk.id          = f"{doc_id}_chunk_{idx}"
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["doc_id"]      = doc_id
        return chunks

    # ─────────────────────────────────────────
    # FILTER SHORT CHUNKS
    # ─────────────────────────────────────────

    def filter_short_chunks(
        self,
        chunks: list[Chunk],
        min_words: int = MIN_WORDS_PER_CHUNK,
    ) -> list[Chunk]:
        """
        Removes low-quality chunks that are too short to be useful.
        Short chunks hurt retrieval quality — they lack context.

        Args:
            chunks:    List of Chunk objects
            min_words: Minimum word count to keep (default 20)

        Returns:
            Filtered list of chunks above word threshold

        Examples:
            filtered = chunker.filter_short_chunks(chunks, min_words=20)
        """
        before = len(chunks)
        filtered = [
            c for c in chunks
            if c.word_count >= min_words
            and c.char_count >= MIN_CHARS_PER_CHUNK
        ]
        removed = before - len(filtered)
        if removed > 0:
            logger.debug(
                f"[CHUNKER] Short chunks removed | "
                f"before={before} | removed={removed} | kept={len(filtered)}"
            )
        return filtered

    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns chunker operational statistics."""
        avg_chunks = (
            round(self._total_chunks / self._total_docs, 1)
            if self._total_docs > 0 else 0.0
        )
        avg_time = (
            round(self._chunk_time_ms / self._total_docs, 1)
            if self._total_docs > 0 else 0.0
        )
        return {
            "total_docs":              self._total_docs,
            "total_chunks":            self._total_chunks,
            "total_duplicates_skipped": self._total_duplicates_skipped,
            "avg_chunks_per_doc":      avg_chunks,
            "total_chunk_ms":          round(self._chunk_time_ms, 1),
            "avg_chunk_ms":            avg_time,
        }


# ─────────────────────────────────────────────
# STANDALONE FUNCTIONS (exported for direct use)
# ─────────────────────────────────────────────

def adaptive_chunk_size(text_type: str) -> int:
    """
    Returns optimal chunk size in characters for a given document type.

    Args:
        text_type: Document source type string

    Returns:
        Chunk size in characters

    Examples:
        adaptive_chunk_size("annual_report")   # → 1200
        adaptive_chunk_size("rbi")             # → 800
        adaptive_chunk_size("news")            # → 600
        adaptive_chunk_size("unknown")         # → 1000
    """
    return CHUNK_SIZES.get(text_type.lower().strip(), DEFAULT_CHUNK_SIZE)


def detect_section_headers(text: str) -> list[str]:
    """
    Detects financial document section headers in text.
    Used as natural chunk boundaries to preserve semantic coherence.

    Detects common patterns:
        "Financial Highlights", "Risk Factors", "Management Discussion",
        "Asset Quality", "Capital Adequacy", "Director's Report", etc.

    Args:
        text: Document text to scan

    Returns:
        List of detected section header strings (in order of appearance)

    Examples:
        headers = detect_section_headers(annual_report_text)
        # → ["Financial Highlights", "Risk Factors", "Capital Adequacy", ...]
    """
    headers: list[str] = []
    seen: set[str] = set()

    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            continue

        # Must look like a header: title-case or uppercase, short line
        if _is_likely_header(stripped):
            normalized = stripped.lower()
            if _SECTION_HEADER_RE.match(stripped) or _matches_header_heuristic(stripped):
                if normalized not in seen:
                    headers.append(stripped)
                    seen.add(normalized)

    return headers


def estimate_tokens(text: str) -> int:
    """
    Fast token count estimate without loading a tokenizer.
    Formula: chars / 4 (approximate for English financial text).

    Accuracy: ±15% vs exact BPE tokenizer count.
    Good enough for chunk size planning.

    Args:
        text: Input text string

    Returns:
        Estimated token count (integer)

    Examples:
        estimate_tokens("Net Interest Margin stood at 4.2%")  # → ~9
        estimate_tokens(long_document_text)                    # → ~2500
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _split_by_sections(
    text: str,
    sections: list[str],
) -> list[tuple[str, str]]:
    """
    Splits document text at detected section headers.

    Returns:
        List of (section_name, section_text) tuples.
        First tuple has section_name="" for pre-header content.
    """
    if not sections:
        return [("", text)]

    result: list[tuple[str, str]] = []
    remaining = text
    current_section = ""

    for header in sections:
        # Escape for regex
        escaped = re.escape(header)
        pattern = re.compile(
            r'(?im)(?:^|\n)\s*' + escaped + r'\s*(?::)?\s*(?:\n|$)'
        )
        match = pattern.search(remaining)
        if not match:
            continue

        # Text before this header
        before = remaining[:match.start()].strip()
        if before:
            result.append((current_section, before))

        remaining = remaining[match.end():]
        current_section = header

    # Remaining text after last header
    if remaining.strip():
        result.append((current_section, remaining.strip()))

    # Fallback: no sections matched
    if not result:
        return [("", text)]

    return result


def _split_into_units(text: str) -> list[str]:
    """
    Splits text into atomic semantic units for chunking.
    Priority order:
        1. Table blocks (kept atomic)
        2. Paragraphs (double newline)
        3. Sentences (., !, ?, ;)
        4. Lines (single newline)

    Never returns empty strings.
    """
    if not text.strip():
        return []

    units: list[str] = []

    # Split by double newlines (paragraph breaks)
    paragraphs = re.split(r'\n{2,}', text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Table markers — keep atomic
        if _is_table_marker(para):
            units.append(para)
            continue

        # Multi-line paragraphs — try sentence splitting
        if len(para) > 200:
            sentences = _split_into_sentences(para)
            units.extend(s for s in sentences if s.strip())
        else:
            units.append(para)

    return [u for u in units if u.strip()]


def _split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using a two-pass strategy.

    Pass 1: Tokenize on sentence-ending punctuation followed by
            whitespace + capital letter.
    Pass 2: Re-join any spurious splits caused by:
            - Decimal numbers  (4.2%, Rs.45)
            - Common abbreviations (Ltd., Inc., Dr., Mr.)
            - Single capital letters (initials, section refs)
            - Financial terms (FY26., Q3.)
    """
    if not text or not text.strip():
        return []

    # Abbreviations that end with a period but are NOT sentence endings
    ABBREVS: frozenset[str] = frozenset({
        "rs", "ltd", "inc", "corp", "dr", "mr", "mrs", "ms", "prof",
        "vs", "etc", "approx", "est", "viz", "no", "vol", "fig",
        "dept", "govt", "mgmt", "q1", "q2", "q3", "q4",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec",
    })

    # Step 1: Split at [.!?;] followed by whitespace + capital / newline
    raw_splits = re.split(r'([.!?;])\s+(?=[A-Z"\'\n])', text)

    # re.split with capturing group returns: [text, sep, text, sep, ...]
    # Reconstruct sentences by joining text+sep pairs
    parts: list[str] = []
    i = 0
    while i < len(raw_splits):
        piece = raw_splits[i]
        if i + 1 < len(raw_splits):
            sep = raw_splits[i + 1]
            parts.append(piece + sep)
            i += 2
        else:
            parts.append(piece)
            i += 1

    # Step 2: Re-join spurious splits
    sentences: list[str] = []
    buffer = ""

    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue

        if buffer:
            # Check if the PREVIOUS part ended with an abbreviation
            # e.g. "Ltd." — last word before the period
            last_word_match = re.search(r'\b(\w+)[.!?;]$', buffer.rstrip())
            if last_word_match:
                last_word = last_word_match.group(1).lower()
                # Rejoin if abbreviation or single char (initial) or digit
                if (
                    last_word in ABBREVS
                    or len(last_word) == 1              # Single initial: "A."
                    or last_word.isdigit()              # "2." in numbered lists
                    or re.match(r'^fy\d{2,4}$', last_word)   # "FY26."
                    or re.match(r'^q[1-4]$', last_word)       # "Q3."
                ):
                    buffer += " " + part
                    continue

            # Check if this part starts with a lowercase letter
            # (continuation of previous sentence broken by decimal)
            if part_stripped and part_stripped[0].islower():
                buffer += " " + part
                continue

            sentences.append(buffer.strip())
            buffer = part
        else:
            buffer = part

    if buffer.strip():
        sentences.append(buffer.strip())

    return [s for s in sentences if s.strip()] or [text]



def _overlap_units(units: list[str], overlap: int) -> list[str]:
    """
    Returns the trailing units whose combined length ≈ overlap chars.
    Used to initialize the next chunk with overlap context.
    """
    if overlap <= 0 or not units:
        return []

    selected: list[str] = []
    total = 0
    for unit in reversed(units):
        if total + len(unit) <= overlap:
            selected.insert(0, unit)
            total += len(unit)
        else:
            break

    return selected


def _protect_tables(text: str) -> tuple[str, dict[str, str]]:
    """
    Replaces table blocks with unique placeholder markers.
    Tables are multi-line blocks containing | characters.

    Returns:
        (text_with_markers, {marker: original_table_text})
    """
    table_registry: dict[str, str] = {}
    result = text
    table_num = 0

    # Find table blocks: 2+ consecutive lines with | characters
    table_pattern = re.compile(
        r'(?:^.+\|.+\n){2,}',
        re.MULTILINE,
    )

    def replace_table(match: re.Match) -> str:
        nonlocal table_num
        marker = f"__TABLE_BLOCK_{table_num}__"
        table_registry[marker] = match.group(0)
        table_num += 1
        return f"\n{marker}\n"

    result = table_pattern.sub(replace_table, result)
    return result, table_registry


def _restore_tables(text: str, table_registry: dict[str, str]) -> str:
    """Restores table placeholders back to original table text."""
    for marker, table_text in table_registry.items():
        text = text.replace(marker, table_text)
    return text


def _is_table_marker(text: str) -> bool:
    """Returns True if text is a table block placeholder."""
    return bool(re.match(r'^__TABLE_BLOCK_\d+__$', text.strip()))


def _force_split_unit(unit: str, chunk_size: int) -> list[str]:
    """
    Force-splits an oversized text unit at sentence boundaries.
    Used when a single unit exceeds chunk_size.
    """
    sentences = _split_into_sentences(unit)
    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            parts.append(" ".join(current).strip())
            current = []
            current_len = 0
        current.append(sentence)
        current_len += len(sentence)

    if current:
        parts.append(" ".join(current).strip())

    return [p for p in parts if p.strip()]


def _is_likely_header(line: str) -> bool:
    """
    Heuristic check: is this line likely a section header?
    Headers are typically:
    - Short (< 80 chars)
    - Title case or ALL CAPS
    - No trailing period (not a sentence)
    - Not a number-only line
    """
    if len(line) > 80 or len(line) < 3:
        return False
    if line.endswith('.') and len(line) > 40:
        return False
    if re.match(r'^\d+\.?\d*$', line):
        return False

    words = line.split()
    if not words:
        return False

    # Title case check: most words capitalized
    cap_words = sum(1 for w in words if w and w[0].isupper())
    if cap_words / len(words) >= 0.6:
        return True

    # ALL CAPS check
    if line.isupper() and len(line) > 3:
        return True

    return False


def _matches_header_heuristic(line: str) -> bool:
    """
    Secondary header heuristic for lines not caught by the main regex.
    Catches numbered sections like "1. Risk Factors", "2.1 Asset Quality"
    """
    numbered = re.match(r'^\d+(?:\.\d+)*\s+(.+)$', line)
    if numbered:
        content = numbered.group(1)
        if len(content) > 3 and not content.endswith('.'):
            return True
    return False


def _md5(text: str) -> str:
    """Computes MD5 hash of text for deduplication."""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def _generate_doc_id(metadata: dict) -> str:
    """
    Generates a deterministic doc_id from metadata.
    Falls back to UUID4 if no useful metadata available.
    """
    source  = metadata.get("source", "")
    company = metadata.get("company", "")
    date    = metadata.get("date", "")

    if source:
        # Slugify the source name
        slug = re.sub(r'[^a-z0-9]+', '_', source.lower()).strip('_')[:60]
        return slug

    if company and date:
        slug = re.sub(r'[^a-z0-9]+', '_', company.lower()).strip('_')
        date_slug = re.sub(r'[^0-9]', '', date)[:8]
        return f"{slug}_{date_slug}"

    return str(uuid.uuid4()).replace("-", "")[:16]


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

chunker = Chunker()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "chunker",
    "Chunker",
    "Chunk",
    "adaptive_chunk_size",
    "detect_section_headers",
    "estimate_tokens",
    "CHUNK_SIZES",
    "OVERLAP_SIZES",
]
