# backend/agents/agent3_reranker.py
"""
FinThesisGuard AI — Agent 3: Re-Ranker
ArkAngel Financial Solutions

Scores and re-ranks retrieved Pinecone chunks before they are passed
to the LLM synthesis layer. Replaces raw cosine similarity ordering
with a multi-signal composite score that accounts for recency,
source authority, and financial term relevance.

Scoring formula (from config):
    final_score = (
        similarity                          ← Pinecone cosine score
        × (1 + RECENCY_WEIGHT × recency)   ← 0.3 × [0.0, 1.0]
        × authority                         ← [0.7, 1.5] from AUTHORITY_WEIGHTS
        × (1 + OVERLAP_WEIGHT × overlap)    ← 0.2 × [0.0, 1.0]
    )

Component ranges:
    similarity:       [0.0,  1.0]  — Pinecone cosine similarity
    recency_boost:    [0.0,  1.0]  — Linear decay over 365 days
    authority_weight: [0.7,  1.5]  — Per source_type multiplier
    term_overlap:     [0.0,  1.0]  — Jaccard on bidirectionally-normalized
                                     financial term tokens

Score example:
    similarity=0.85, recency=0.9 (last month), authority=1.3 (annual_report),
    overlap=0.6
    → 0.85 × (1 + 0.3×0.9) × 1.3 × (1 + 0.2×0.6)
    → 0.85 × 1.27 × 1.3 × 1.12
    ≈ 1.574

Pipeline position:
    Pinecone retrieval → [Agent 3: Re-Ranker] → LLM synthesis
    Top 10 raw chunks → filter → score → top 6 reranked chunks

Key design:
    - Synchronous (no async needed — pure computation)
    - Score breakdown attached to every chunk for explainability
    - Handles missing metadata gracefully (defaults to safe fallbacks)
    - No external API calls — latency target 5-15ms for 10 chunks
    - Financial term vocabulary loaded once at import from FINANCIAL_TERMS
    - Bidirectional acronym normalization: "NIM" ↔ "Net Interest Margin"
      ensures acronyms in queries correctly match expansions in chunk text
      and vice-versa (fixes Jaccard = 0.0 bug for acronym↔expansion pairs)
"""

import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, date
from typing import Optional

from backend.config import (
    AUTHORITY_WEIGHTS,
    RECENCY_WEIGHT,
    OVERLAP_WEIGHT,
    RECENCY_DECAY_DAYS,
    MIN_SIMILARITY_THRESHOLD,
    TOP_K_RERANKED,
)
from backend.utils.financial_terms import FINANCIAL_TERMS
from backend.utils.logger           import logger, log_metric


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

AGENT_NAME = "agent3_reranker"

# Default authority for unrecognised source_types
_DEFAULT_AUTHORITY: float = AUTHORITY_WEIGHTS.get("unknown", 0.7)

# Minimum composite score to keep a chunk after full re-ranking.
# Separate from MIN_SIMILARITY_THRESHOLD which gates raw cosine pre-filter.
MIN_RERANK_SCORE: float = 0.10

# Date formats tried in order when parsing chunk metadata["date"].
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",     # 2026-01-15  ← most common in metadata
    "%d-%m-%Y",     # 15-01-2026
    "%d/%m/%Y",     # 15/01/2026
    "%Y/%m/%d",     # 2026/01/15
    "%B %d, %Y",    # January 15, 2026
    "%b %d, %Y",    # Jan 15, 2026
    "%d %B %Y",     # 15 January 2026
    "%Y%m%d",       # 20260115
)

# Common English stopwords excluded from term overlap to reduce noise.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "not", "but", "for", "nor",
    "so", "yet", "in", "on", "at", "to", "up", "as", "is", "it",
    "be", "do", "go", "if", "by", "my", "we", "he", "she", "they",
    "you", "me", "us", "him", "her", "our", "its", "was", "are",
    "has", "had", "did", "can", "may", "will", "of", "that", "this",
    "with", "from", "into", "than", "then", "when", "what", "which",
    "how", "why", "who", "all", "any", "few", "more", "most", "per",
    "due", "via", "vs", "etc", "inc", "ltd", "pvt", "fig", "table",
    "page", "note", "source", "ibid",
})


# ─────────────────────────────────────────────
# FINANCIAL VOCABULARY (built once at import)
# ─────────────────────────────────────────────
# FINANCIAL_TERMS: dict[str, str]
#   e.g. {"NIM": "Net Interest Margin", "GNPA": "Gross NPA", ...}
#
# We build three structures from this dict:
#
#   _FINANCIAL_TERM_TOKENS    — flat frozenset of every individual token
#                               (acronym tokens + expansion word tokens)
#                               Used to quickly check if a word belongs to
#                               the financial domain.
#
#   _ACRONYM_TO_EXPANSION_TOKENS — maps each lowercase acronym key to the
#                               frozenset of its expansion word tokens.
#                               e.g. "nim" → {"net","interest","margin"}
#                               Used in Direction A of normalization.
#
#   _EXPANSION_PHRASE_TO_ACRONYM — maps each full expansion phrase (lowercase)
#                               to its acronym.
#                               e.g. "net interest margin" → "nim"
#                               Used in Direction B of normalization.
#
# Together A+B give bidirectional equivalence so Jaccard never returns 0.0
# when the query uses an acronym and the chunk uses its expansion (or v/v).
# ─────────────────────────────────────────────

def _build_fin_token(s: str) -> frozenset[str]:
    """Splits a string into cleaned tokens, excluding stopwords."""
    return frozenset(
        t for t in s.lower().replace("-", " ").split()
        if len(t) >= 2 and t not in _STOPWORDS
    )

_ACRONYM_TO_EXPANSION_TOKENS: dict[str, frozenset[str]] = {
    acr.lower().strip(): _build_fin_token(exp)
    for acr, exp in FINANCIAL_TERMS.items()
}

_EXPANSION_PHRASE_TO_ACRONYM: dict[str, str] = {
    exp.lower().strip(): acr.lower().strip()
    for acr, exp in FINANCIAL_TERMS.items()
}

# Flat set of every token that appears in any acronym or expansion.
# Used to determine whether a raw token belongs to financial vocabulary.
_FINANCIAL_TERM_TOKENS: frozenset[str] = frozenset(
    token
    for acr, exp_tokens in _ACRONYM_TO_EXPANSION_TOKENS.items()
    for token in list(exp_tokens) + [acr]
    if len(token) >= 2
)


# ─────────────────────────────────────────────
# SCORE BREAKDOWN DATACLASS
# ─────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    """
    Detailed breakdown of all signals contributing to final_score.
    Attached to every reranked chunk for explainability.

    Fields:
        similarity:         Raw Pinecone cosine similarity [0.0, 1.0]
        recency_boost:      Linear recency decay [0.0, 1.0] (1.0 = today)
        authority_weight:   Source authority multiplier [0.7, 1.5]
        term_overlap:       Jaccard on bidirectionally-normalized fin terms [0.0, 1.0]
        final_score:        Composite re-ranking score (may exceed 1.0)
        rank:               1-based position after re-ranking (1 = best)
        date_parsed:        ISO date string if successfully parsed, else ""
        source_type:        Source type used for authority lookup
        query_fin_tokens:   Financial tokens extracted from query (for debug)
        chunk_fin_tokens:   Financial tokens extracted from chunk (for debug)
    """
    similarity:       float
    recency_boost:    float
    authority_weight: float
    term_overlap:     float
    final_score:      float
    rank:             int            = 0
    date_parsed:      str            = ""
    source_type:      str            = "unknown"
    query_fin_tokens: frozenset[str] = frozenset()
    chunk_fin_tokens: frozenset[str] = frozenset()

    def to_dict(self) -> dict:
        d = asdict(self)
        # frozensets are not JSON-serialisable — convert to sorted lists
        d["query_fin_tokens"] = sorted(self.query_fin_tokens)
        d["chunk_fin_tokens"] = sorted(self.chunk_fin_tokens)
        return d

    def explanation(self) -> str:
        """Human-readable one-liner for debugging and frontend display."""
        return (
            f"score={self.final_score:.4f} | "
            f"sim={self.similarity:.3f} | "
            f"recency={self.recency_boost:.2f} | "
            f"authority={self.authority_weight:.1f} | "
            f"overlap={self.term_overlap:.2f} | "
            f"source={self.source_type}"
        )


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _tokenize(text: str) -> frozenset[str]:
    """
    Tokenizes text into a frozenset of lowercase alphanumeric tokens.
    Removes stopwords and pure-digit tokens. Minimum token length: 2.

    Normalization applied before splitting:
        hyphens → spaces   ("non-performing" → "non performing")
        slashes → spaces   ("NIM/NPA"        → "nim npa")
        ampersands → spaces ("F&O"           → "f o")
        underscores → spaces

    Examples:
        _tokenize("HDFC Bank NIM Q3 FY26")
        → frozenset({"hdfc", "bank", "nim", "q3", "fy26"})

        _tokenize("Net Interest Margin (NIM) stood at 4.2%")
        → frozenset({"net", "interest", "margin", "nim", "stood"})
    """
    if not text:
        return frozenset()

    normalized = (
        text.lower()
            .replace("-", " ")
            .replace("/", " ")
            .replace("&", " ")
            .replace("_", " ")
    )
    tokens = re.findall(r'\b[a-z0-9][a-z0-9]*\b', normalized)
    return frozenset(
        t for t in tokens
        if len(t) >= 2
        and t not in _STOPWORDS
        and not t.isdigit()
    )


def _get_financial_token_set(
    text:   str,
    tokens: Optional[frozenset[str]] = None,
) -> frozenset[str]:
    """
    Produces a canonical financial token set with bidirectional
    acronym ↔ expansion normalization, so that Jaccard similarity
    is non-zero whenever query and chunk share any term regardless of
    whether it appears as an acronym in one and as its expansion in the other.

    Two-directional normalization:

        Direction A — acronym found in text → also add its expansion tokens:
            text contains "NIM"
            → adds "nim" (acronym itself) + {"net","interest","margin"}

        Direction B — expansion phrase found in text → also add its acronym:
            text contains "Net Interest Margin"
            → adds "nim" + {"net","interest","margin"}

    Effect on Jaccard:
        query  = "HDFC NIM Q3 FY26"
            query_fin = {"nim","net","interest","margin","q3","fy26"}

        chunk  = "Net Interest Margin stood at 4.2% for Q3 FY26"
            chunk_fin = {"nim","net","interest","margin","q3","fy26"}

        intersection = {"nim","net","interest","margin","q3","fy26"} → 6
        union        = same                                           → 6
        Jaccard = 6/6 = 1.0   (was 0.0 before fix)

    Args:
        text:   Raw text string (needed for multi-word phrase scan)
        tokens: Pre-tokenized frozenset; computed internally if None.
                Pass pre-computed set for batch efficiency.

    Returns:
        frozenset of canonical, lowercased financial tokens.
    """
    if tokens is None:
        tokens = _tokenize(text)

    result: set[str] = set()
    text_lower = text.lower()

    # ── Direction A: token is a known acronym key ─────────────────────────
    for token in tokens:
        if token in _ACRONYM_TO_EXPANSION_TOKENS:
            result.add(token)                               # keep acronym
            result |= _ACRONYM_TO_EXPANSION_TOKENS[token]  # add expansion
        elif token in _FINANCIAL_TERM_TOKENS:
            result.add(token)                               # plain fin token

    # ── Direction B: expansion phrase appears as substring ────────────────
    # Sort by phrase length descending so longer phrases match first,
    # preventing "npa" from matching before "gross npa" etc.
    for phrase in sorted(_EXPANSION_PHRASE_TO_ACRONYM, key=len, reverse=True):
        if phrase in text_lower:
            acr = _EXPANSION_PHRASE_TO_ACRONYM[phrase]
            result.add(acr)
            if acr in _ACRONYM_TO_EXPANSION_TOKENS:
                result |= _ACRONYM_TO_EXPANSION_TOKENS[acr]

    return frozenset(result)


# ─────────────────────────────────────────────
# RERANKER CLASS
# ─────────────────────────────────────────────

class Reranker:
    """
    Agent 3: Scores and re-ranks retrieved chunks using a composite
    multi-signal formula.

    Usage:
        from backend.agents.agent3_reranker import reranker

        reranked_chunks = reranker.rerank(
            chunks = pinecone_results,
            query  = "HDFC Bank NIM Q3 FY26",
            top_k  = 6,
        )
        for chunk in reranked_chunks:
            print(chunk["_score_breakdown"].explanation())
            print(chunk["text"][:200])

    Chunk dict expected schema (from Pinecone retriever):
        {
            "id":         "doc_abc123_chunk_0042",
            "text":       "Net Interest Margin for Q3 FY26 stood at 4.2%...",
            "similarity": 0.87,
            "metadata": {
                "source_type": "annual_report",
                "source":      "HDFC Bank Annual Report FY26",
                "date":        "2026-01-15",
                "company":     "HDFC Bank",
                "chunk_index": 42,
                "doc_id":      "doc_abc123",
            }
        }
    """

    def __init__(self):
        self._total_reranked: int   = 0
        self._total_filtered: int   = 0
        self._total_input:    int   = 0
        self._total_ms:       float = 0.0
        self._vocab_size:     int   = len(_FINANCIAL_TERM_TOKENS)

        logger.info(
            f"[RERANKER] Agent 3 ready | "
            f"vocab_size={self._vocab_size} financial tokens | "
            f"acronyms={len(_ACRONYM_TO_EXPANSION_TOKENS)} | "
            f"min_threshold={MIN_SIMILARITY_THRESHOLD} | "
            f"top_k={TOP_K_RERANKED}"
        )

    # ─────────────────────────────────────────
    # PUBLIC: rerank
    # ─────────────────────────────────────────

    def rerank(
        self,
        chunks:    list[dict],
        query:     str,
        top_k:     int   = TOP_K_RERANKED,
        min_score: float = MIN_SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        """
        Main entry point. Scores and re-ranks a list of retrieved chunks.

        Pipeline:
            1. Guard: empty inputs → early return
            2. Pre-filter: discard chunks below min_score (raw cosine)
            3. Score: calculate_final_score() for each eligible chunk
            4. Post-filter: discard chunks below MIN_RERANK_SCORE
            5. Sort descending by final_score
            6. Assign ranks, return top_k with _score_breakdown attached

        Args:
            chunks:    list[dict] from Pinecone retriever.
                       Each must have: id, text, similarity, metadata.
            query:     Search query string (used for term overlap).
            top_k:     Max chunks to return (default: TOP_K_RERANKED=6).
            min_score: Min raw cosine similarity to pass pre-filter
                       (default: MIN_SIMILARITY_THRESHOLD=0.3).

        Returns:
            List of chunk dicts sorted by final_score descending.
            Each chunk gains two keys:
                "_score_breakdown"  → ScoreBreakdown instance
                "_rank"             → int (1 = highest score)

        Raises:
            Nothing — all errors handled gracefully with fallback scores.

        Latency:
            10 chunks: ~5ms
            50 chunks: ~18ms

        Examples:
            chunks = [
                {"id": "c1", "text": "NIM at 4.2%...", "similarity": 0.87,
                 "metadata": {"source_type": "annual_report", "date": "2026-01-15"}},
                {"id": "c2", "text": "RBI circular...", "similarity": 0.72,
                 "metadata": {"source_type": "rbi", "date": "2025-03-01"}},
            ]
            results = reranker.rerank(chunks, "HDFC Bank NIM Q3 FY26")
            # c2 (authority=1.5) may outrank c1 (authority=1.3)
            # despite lower raw similarity
        """
        if not chunks:
            logger.debug("[RERANKER] Empty chunk list — returning []")
            return []

        if not query or not query.strip():
            logger.warning(
                "[RERANKER] Empty query — returning chunks sorted by similarity"
            )
            return sorted(
                chunks, key=lambda c: c.get("similarity", 0.0), reverse=True
            )[:top_k]

        start_ms = time.perf_counter()
        self._total_input += len(chunks)

        # Pre-compute query canonical financial token set once.
        # Shared across all chunk scorings in this call — O(n) not O(n²).
        query_raw_tokens   = _tokenize(query)
        query_fin_tokens   = _get_financial_token_set(query, query_raw_tokens)

        # ── Step 1: Pre-filter on raw similarity ──────────────────────────
        eligible = [
    c for c in chunks
    if max(
        c.get("similarity", 0.0),
        c.get("score", 0.0),
        c.get("final_score", 0.0),
    ) >= min_score
]
        filtered_pre = len(chunks) - len(eligible)
        self._total_filtered += filtered_pre

        if filtered_pre > 0:
            logger.debug(
                f"[RERANKER] Pre-filtered {filtered_pre}/{len(chunks)} chunks "
                f"below similarity threshold {min_score}"
            )

        if not eligible:
            logger.warning(
                f"[RERANKER] All {len(chunks)} chunks filtered "
                f"(similarity < {min_score}) — returning []"
            )
            return []

        # ── Step 2: Score all eligible chunks ─────────────────────────────
        scored: list[tuple[float, dict]] = []

        for chunk in eligible:
            try:
                breakdown = self.calculate_final_score(
                    chunk            = chunk,
                    query            = query,
                    query_raw_tokens = query_raw_tokens,
                    query_fin_tokens = query_fin_tokens,
                )
                chunk_copy = dict(chunk)
                chunk_copy["_score_breakdown"] = breakdown
                chunk_copy["_final_score"]     = breakdown.final_score

                if breakdown.final_score >= MIN_RERANK_SCORE:
                    scored.append((breakdown.final_score, chunk_copy))

            except Exception as e:
                logger.warning(
                    f"[RERANKER] Scoring failed for chunk {chunk.get('id','?')} | "
                    f"error={type(e).__name__}: {e} — using raw similarity fallback"
                )
                fallback = chunk.get("similarity", 0.0)
                chunk_copy = dict(chunk)
                chunk_copy["_score_breakdown"] = ScoreBreakdown(
                    similarity       = fallback,
                    recency_boost    = 0.5,
                    authority_weight = _DEFAULT_AUTHORITY,
                    term_overlap     = 0.0,
                    final_score      = fallback,
                    source_type      = "unknown",
                )
                chunk_copy["_final_score"] = fallback
                scored.append((fallback, chunk_copy))

        # ── Step 3: Sort descending by composite score ────────────────────
        scored.sort(key=lambda x: x[0], reverse=True)

        # ── Step 4: Assign ranks and slice top_k ─────────────────────────
        result: list[dict] = []
        for rank, (_, chunk) in enumerate(scored[:top_k], start=1):
            chunk["_score_breakdown"].rank = rank
            chunk["_rank"] = rank
            result.append(chunk)

        # ── Step 5: Stats and logging ─────────────────────────────────────
        duration_ms       = (time.perf_counter() - start_ms) * 1000
        self._total_ms   += duration_ms
        self._total_reranked += len(result)

        logger.info(
            f"[RERANKER] Complete | "
            f"input={len(chunks)} | eligible={len(eligible)} | "
            f"returned={len(result)} | {duration_ms:.1f}ms"
        )
        if result:
            logger.debug(
                f"[RERANKER] Top chunk: "
                f"{result[0]['_score_breakdown'].explanation()}"
            )

        log_metric("reranker_duration_ms",  duration_ms,  unit="ms")
        log_metric("reranker_chunks_in",    len(chunks),  unit="count")
        log_metric("reranker_chunks_out",   len(result),  unit="count")

        return result

    # ─────────────────────────────────────────
    # PUBLIC: calculate_final_score
    # ─────────────────────────────────────────

    def calculate_final_score(
        self,
        chunk:            dict,
        query:            str,
        query_raw_tokens: Optional[frozenset[str]] = None,
        query_fin_tokens: Optional[frozenset[str]] = None,
    ) -> ScoreBreakdown:
        """
        Computes the full multi-signal composite score for one chunk.

        Formula (from config.py):
            final_score = (
                similarity
                × (1 + RECENCY_WEIGHT  × recency_boost)
                × authority_weight
                × (1 + OVERLAP_WEIGHT × term_overlap)
            )

        Args:
            chunk:            Chunk dict with 'similarity', 'text', 'metadata'.
            query:            Search query string.
            query_raw_tokens: Pre-tokenized frozenset from _tokenize(query).
                              Computed internally if None.
            query_fin_tokens: Pre-computed canonical financial token set from
                              _get_financial_token_set(query). Computed if None.
                              Pass both for maximum batch efficiency.

        Returns:
            ScoreBreakdown with all component scores and final_score.
            Also carries query_fin_tokens and chunk_fin_tokens for debug.

        Examples:
            breakdown = reranker.calculate_final_score(
                chunk={
                    "similarity": 0.85,
                    "text": "RBI MPC kept repo rate unchanged at 6.5%...",
                    "metadata": {"source_type": "rbi", "date": "2026-03-01"},
                },
                query="RBI repo rate decision March 2026"
            )
            # similarity=0.85, recency≈0.98, authority=1.5, overlap>0
            # final_score ≈ 1.88
        """
        metadata    = chunk.get("metadata", {}) or {}
        similarity = float(chunk.get("similarity") or chunk.get("score")or chunk.get("final_score")or 0.0)
        source_type = str(metadata.get("source_type", "unknown")).lower().strip()
        date_str    = str(metadata.get("date", ""))
        text        = str(chunk.get("text", ""))

        # Component 1 — Recency boost
        recency_boost, date_parsed = self.calculate_recency_boost(date_str)

        # Component 2 — Authority weight
        authority_weight = self.get_authority_weight(source_type)

        # Component 3 — Term overlap (bidirectional Jaccard)
        if query_raw_tokens is None:
            query_raw_tokens = _tokenize(query)
        if query_fin_tokens is None:
            query_fin_tokens = _get_financial_token_set(query, query_raw_tokens)

        # Use private detailed version to capture chunk_fin_tokens for
        # ScoreBreakdown explainability — public calculate_term_overlap()
        # returns float only (preserving the public API contract).
        term_overlap, chunk_fin_tokens = self._calculate_term_overlap_detailed(
            query            = query,
            chunk_text       = text,
            query_fin_tokens = query_fin_tokens,
        )

        # Composite formula
        final_score = (
            similarity
            * (1.0 + RECENCY_WEIGHT  * recency_boost)
            * authority_weight
            * (1.0 + OVERLAP_WEIGHT * term_overlap)
        )

        return ScoreBreakdown(
            similarity        = round(similarity,       4),
            recency_boost     = round(recency_boost,    4),
            authority_weight  = round(authority_weight, 4),
            term_overlap      = round(term_overlap,     4),
            final_score       = round(final_score,      6),
            date_parsed       = date_parsed,
            source_type       = source_type,
            query_fin_tokens  = query_fin_tokens,
            chunk_fin_tokens  = chunk_fin_tokens,
        )

    # ─────────────────────────────────────────
    # PUBLIC: calculate_recency_boost
    # ─────────────────────────────────────────

    def calculate_recency_boost(self, date_str: str) -> tuple[float, str]:
        """
        Computes a linear recency decay score from a date string.

        Decay model:
            days_ago = 0    → boost = 1.0   (published today)
            days_ago = 182  → boost = 0.5   (6 months ago)
            days_ago = 365  → boost = 0.0   (1 year ago)
            days_ago > 365  → boost = 0.0   (clamped at 0)

        Formula:
            boost = max(0.0, 1.0 - days_ago / RECENCY_DECAY_DAYS)

        Args:
            date_str: Date string in any supported format.
                      Supported: "2026-01-15", "15-01-2026", "15/01/2026",
                      "January 15, 2026", "Jan 15, 2026", "15 January 2026",
                      "20260115", or any ISO datetime with YYYY-MM-DD prefix.

        Returns:
            (recency_boost: float, date_parsed_iso: str)
            On parse failure → (0.5, "") — neutral, neither penalizes
            nor rewards undated documents.

        Examples:
            calculate_recency_boost("2026-03-19")   → (1.000, "2026-03-19")
            calculate_recency_boost("2025-09-19")   → (0.501, "2025-09-19")
            calculate_recency_boost("2025-03-19")   → (0.000, "2025-03-19")
            calculate_recency_boost("")             → (0.500, "")
            calculate_recency_boost("bad_date")     → (0.500, "")
        """
        if not date_str or date_str.strip() in ("", "None", "null", "N/A", "nan"):
            return 0.5, ""

        date_str      = date_str.strip()
        parsed_date: Optional[date] = None

        # Try known format strings in order
        for fmt in _DATE_FORMATS:
            try:
                parsed_date = datetime.strptime(date_str, fmt).date()
                break
            except ValueError:
                continue

        # Last resort: extract YYYY-MM-DD substring from ISO datetimes
        if parsed_date is None:
            m = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
            if m:
                try:
                    parsed_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
                except ValueError:
                    pass

        if parsed_date is None:
            logger.debug(
                f"[RERANKER] Could not parse date '{date_str}' — "
                f"returning neutral recency=0.5"
            )
            return 0.5, ""

        today    = datetime.now(timezone.utc).date()
        days_ago = max(0, (today - parsed_date).days)   # clamp future dates to 0
        boost    = max(0.0, 1.0 - days_ago / RECENCY_DECAY_DAYS)
        return round(boost, 4), parsed_date.isoformat()

    # ─────────────────────────────────────────
    # PUBLIC: get_authority_weight
    # ─────────────────────────────────────────

    def get_authority_weight(self, source_type: str) -> float:
        """
        Returns the authority multiplier for a given source type.
        Looks up AUTHORITY_WEIGHTS from config.py. Case-insensitive.

        Authority hierarchy (high → low):
            rbi / sebi:          1.5  ← Regulatory primary sources
            mca / tax_law:       1.4  ← Government legal documents
            annual_report:       1.3  ← Primary company disclosure
            earnings_transcript: 1.2  ← Management commentary
            broker_research:     1.1  ← Analyst secondary research
            news:                1.0  ← Baseline
            blog:                0.8  ← Informal / opinion
            unknown:             0.7  ← Lowest trust

        Args:
            source_type: String like "rbi", "ANNUAL_REPORT", "news", etc.

        Returns:
            Float from AUTHORITY_WEIGHTS, or _DEFAULT_AUTHORITY (0.7) if unknown.

        Examples:
            get_authority_weight("rbi")           → 1.5
            get_authority_weight("ANNUAL_REPORT") → 1.3
            get_authority_weight("twitter")       → 0.7
        """
        key    = source_type.lower().strip()
        weight = AUTHORITY_WEIGHTS.get(key)
        if weight is None:
            logger.debug(
                f"[RERANKER] Unknown source_type '{source_type}' — "
                f"using default authority {_DEFAULT_AUTHORITY}"
            )
            return _DEFAULT_AUTHORITY
        return weight

    # ─────────────────────────────────────────
    # PUBLIC: calculate_term_overlap
    # ─────────────────────────────────────────
    def _calculate_term_overlap_detailed(
        self,
        query:            str,
        chunk_text:       str,
        query_fin_tokens: Optional[frozenset[str]] = None,
    ) -> tuple[float, frozenset[str]]:
        """
        Internal version used by calculate_final_score().
        Returns both the Jaccard score AND the chunk's financial token set
        so ScoreBreakdown can store it for explain(verbose=True).

        Public callers use calculate_term_overlap() which returns float only.
        """
        if query_fin_tokens is None:
            query_fin_tokens = _get_financial_token_set(query)

        chunk_fin_tokens = _get_financial_token_set(chunk_text)

        if not query_fin_tokens or not chunk_fin_tokens:
            return 0.0, chunk_fin_tokens

        intersection = query_fin_tokens & chunk_fin_tokens
        union        = query_fin_tokens | chunk_fin_tokens

        jaccard = round(len(intersection) / len(union), 4) if union else 0.0
        return jaccard, chunk_fin_tokens

    def calculate_term_overlap(
        self,
        query:            str,
        chunk_text:       str,
        query_fin_tokens: Optional[frozenset[str]] = None,
    ) -> float:
        """
        Computes Jaccard similarity between query and chunk using
        bidirectionally-normalized canonical financial token sets.

        Public API — always returns float in [0.0, 1.0].
        Internal scoring uses _calculate_term_overlap_detailed() which also
        returns the chunk token set for ScoreBreakdown explainability.

        Bidirectional normalization (the core fix):
            "NIM" in query   → expands to {"nim","net","interest","margin"}
            "Net Interest Margin" in chunk → same canonical set
            → Jaccard > 0 instead of 0.0

        Args:
            query:            Search query string.
            chunk_text:       Full text of the retrieved chunk.
            query_fin_tokens: Pre-computed canonical set (batch optimization).
                              Computed internally if None.

        Returns:
            Jaccard similarity float in [0.0, 1.0].

        Examples:
            # Acronym in query ↔ expansion in chunk — FIXED ✓
            calculate_term_overlap(
                "HDFC NIM Q3 FY26",
                "Net Interest Margin stood at 4.2% for Q3 FY26"
            ) → ~0.7

            # Expansion in query ↔ acronym in chunk — FIXED ✓
            calculate_term_overlap(
                "Net Interest Margin for HDFC",
                "HDFC NIM was 4.2% in Q3 FY26"
            ) → ~0.7

            # No financial terms — returns 0.0 ✓
            calculate_term_overlap("hello world", "lorem ipsum") → 0.0
        """
        score, _ = self._calculate_term_overlap_detailed(
            query            = query,
            chunk_text       = chunk_text,
            query_fin_tokens = query_fin_tokens,
        )
        return score
    # ─────────────────────────────────────────
    # PUBLIC: explain
    # ─────────────────────────────────────────

    def explain(self, chunk: dict, verbose: bool = False) -> str:
        """
        Returns a human-readable explanation of a chunk's re-ranking score.
        Useful for debugging and frontend score transparency.

        Args:
            chunk:   A chunk dict already processed by rerank().
                     Must have "_score_breakdown" key.
            verbose: If True, also prints the financial token sets used
                     in Jaccard calculation (useful for debugging overlap=0).

        Returns:
            Multi-line formatted explanation string.

        Example output:
            Re-Rank Score: 1.2743  [rank #1]
            ────────────────────────────────────
            Cosine Similarity : 0.872  (raw Pinecone match quality)
            Recency Boost     : 0.950  (published 2026-02-28)
            Authority Weight  : 1.300  (source: annual_report)
            Term Overlap      : 0.450  (Jaccard on financial vocab)
            ────────────────────────────────────
            Formula:
              0.872 × (1 + 0.3×0.95) × 1.30 × (1 + 0.2×0.45)
            = 0.872 × 1.285 × 1.300 × 1.090
            = 1.2743
        """
        bd = chunk.get("_score_breakdown")
        if not isinstance(bd, ScoreBreakdown):
            return "No score breakdown available — chunk was not processed by reranker."

        rank_str  = f"  [rank #{bd.rank}]" if bd.rank else ""
        date_info = f"(published {bd.date_parsed})" if bd.date_parsed else "(date unknown)"

        lines = [
            f"Re-Rank Score: {bd.final_score:.4f}{rank_str}",
            "─" * 38,
            f"Cosine Similarity : {bd.similarity:.3f}  (raw Pinecone match quality)",
            f"Recency Boost     : {bd.recency_boost:.3f}  {date_info}",
            f"Authority Weight  : {bd.authority_weight:.3f}  (source: {bd.source_type})",
            f"Term Overlap      : {bd.term_overlap:.3f}  (Jaccard on financial vocab)",
            "─" * 38,
            "Formula:",
            f"  {bd.similarity:.3f} "
            f"× (1 + {RECENCY_WEIGHT}×{bd.recency_boost:.3f}) "
            f"× {bd.authority_weight:.3f} "
            f"× (1 + {OVERLAP_WEIGHT}×{bd.term_overlap:.3f})",
            f"= {bd.similarity:.3f} "
            f"× {1 + RECENCY_WEIGHT * bd.recency_boost:.3f} "
            f"× {bd.authority_weight:.3f} "
            f"× {1 + OVERLAP_WEIGHT * bd.term_overlap:.3f}",
            f"= {bd.final_score:.4f}",
        ]

        if verbose:
            q_toks = sorted(bd.query_fin_tokens) if bd.query_fin_tokens else []
            c_toks = sorted(bd.chunk_fin_tokens) if bd.chunk_fin_tokens else []
            shared = sorted(bd.query_fin_tokens & bd.chunk_fin_tokens)
            lines += [
                "─" * 38,
                f"Query  financial tokens ({len(q_toks)}): {q_toks}",
                f"Chunk  financial tokens ({len(c_toks)}): {c_toks}",
                f"Shared tokens          ({len(shared)}): {shared}",
            ]

        return "\n".join(lines)

    # ─────────────────────────────────────────
    # PUBLIC: get_stats
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns runtime statistics for monitoring dashboards."""
        avg_ms = round(
            self._total_ms / max(1, self._total_reranked), 2
        )
        filter_rate = round(
            self._total_filtered / max(1, self._total_input) * 100, 1
        )
        return {
            "total_input_chunks":    self._total_input,
            "total_filtered_chunks": self._total_filtered,
            "total_reranked_chunks": self._total_reranked,
            "filter_rate_pct":       filter_rate,
            "total_ms":              round(self._total_ms, 1),
            "avg_ms_per_chunk":      avg_ms,
            "vocab_size":            self._vocab_size,
            "acronym_count":         len(_ACRONYM_TO_EXPANSION_TOKENS),
        }


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

reranker = Reranker()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "reranker",
    "Reranker",
    "ScoreBreakdown",
]
