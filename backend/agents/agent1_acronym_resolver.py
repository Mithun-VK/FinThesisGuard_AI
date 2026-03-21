# backend/agents/agent1_acronym_resolver.py
"""
FinThesisGuard AI — Agent 1: Acronym Resolver
ArkAngel Financial Solutions

Expands financial acronyms in user queries BEFORE retrieval so that
embedding-based search finds semantically richer matches.

Example transformations:
    "What is HDFC Bank's NIM and GNPA trend in Q3 FY26?"
    → "What is HDFC Bank's Net Interest Margin (NIM) and
       Gross Non-Performing Assets (GNPA) trend in Q3 FY26?"

    "Explain RBI's stance on FLDG"
    → "Explain Reserve Bank of India (RBI)'s stance on
       First Loss Default Guarantee (FLDG)"

Resolution strategy (in priority order):
    1. Exact match in ACRONYM_DICT             → instant dict lookup  (~0ms)
    2. Case-insensitive match in ACRONYM_DICT  → dict lookup          (~0ms)
    3. Context-aware Groq disambiguation       → LLM call             (~150ms)
    4. Unknown — returned as-is with a warning log

Design principles:
    - Non-destructive: only APPENDS "(expansion)" after the acronym,
      never rewrites surrounding words
    - Idempotent: running twice on same query produces identical output
    - Thread-safe: no mutable state, safe for concurrent FastAPI requests
    - Graceful degradation: if Groq call fails, returns original query
"""

import asyncio
import re
import time
from typing import Optional

from backend.utils.financial_terms import FINANCIAL_TERMS
from backend.utils.llm_client      import llm_client
from backend.utils.logger          import logger, log_metric


# ─────────────────────────────────────────────
# ACRONYM DICTIONARY
# Canonical Indian financial market acronyms.
# Format: { "ACRONYM": "Full Form" }
# Merged with FINANCIAL_TERMS from financial_terms.py at runtime.
# ─────────────────────────────────────────────

_BUILTIN_ACRONYM_DICT: dict[str, str] = {

    # ── Banking / Credit metrics ──────────────────────────────────────────
    "NIM":    "Net Interest Margin",
    "NII":    "Net Interest Income",
    "NIE":    "Non-Interest Expense",
    "NPA":    "Non-Performing Asset",
    "GNPA":   "Gross Non-Performing Asset",
    "NNPA":   "Net Non-Performing Asset",
    "CASA":   "Current Account Savings Account",
    "CRAR":   "Capital to Risk-weighted Asset Ratio",
    "CAR":    "Capital Adequacy Ratio",
    "PCR":    "Provision Coverage Ratio",
    "LCR":    "Liquidity Coverage Ratio",
    "NSFR":   "Net Stable Funding Ratio",
    "SLR":    "Statutory Liquidity Ratio",
    "CRR":    "Cash Reserve Ratio",
    "LTV":    "Loan to Value Ratio",
    "DSCR":   "Debt Service Coverage Ratio",
    "ECL":    "Expected Credit Loss",
    "LGD":    "Loss Given Default",
    "EAD":    "Exposure at Default",
    "PD":     "Probability of Default",
    "RWA":    "Risk-Weighted Asset",
    "ALM":    "Asset Liability Management",
    "NHB":    "National Housing Bank",
    "PSL":    "Priority Sector Lending",
    "MSME":   "Micro Small and Medium Enterprise",
    "KCC":    "Kisan Credit Card",
    "SMA":    "Special Mention Account",
    "OTS":    "One Time Settlement",
    "DRT":    "Debt Recovery Tribunal",
    "SARFAESI": "Securitisation and Reconstruction of Financial Assets and Enforcement of Security Interest",
    "IBC":    "Insolvency and Bankruptcy Code",
    "NCLT":   "National Company Law Tribunal",

    # ── RBI / Monetary Policy ─────────────────────────────────────────────
    "RBI":    "Reserve Bank of India",
    "MPC":    "Monetary Policy Committee",
    "MSF":    "Marginal Standing Facility",
    "LAF":    "Liquidity Adjustment Facility",
    "OMO":    "Open Market Operations",
    "SDF":    "Standing Deposit Facility",
    "VRRR":   "Variable Rate Reverse Repo",
    "VRR":    "Variable Rate Repo",
    "CPI":    "Consumer Price Index",
    "WPI":    "Wholesale Price Index",
    "IIP":    "Index of Industrial Production",
    "FPI":    "Foreign Portfolio Investment",
    "FDI":    "Foreign Direct Investment",
    "ECB":    "External Commercial Borrowing",
    "FCNR":   "Foreign Currency Non-Resident",
    "NRE":    "Non-Resident External",
    "NRO":    "Non-Resident Ordinary",
    "FEMA":   "Foreign Exchange Management Act",
    "LERMS":  "Liberalised Exchange Rate Management System",

    # ── SEBI / Capital Markets ────────────────────────────────────────────
    "SEBI":   "Securities and Exchange Board of India",
    "NSE":    "National Stock Exchange",
    "BSE":    "Bombay Stock Exchange",
    "NSDL":   "National Securities Depository Limited",
    "CDSL":   "Central Depository Services Limited",
    "AMFI":   "Association of Mutual Funds in India",
    "AIF":    "Alternative Investment Fund",
    "PMS":    "Portfolio Management Service",
    "PIPE":   "Private Investment in Public Equity",
    "QIB":    "Qualified Institutional Buyer",
    "HNI":    "High Net-worth Individual",
    "NII":    "Non-Institutional Investor",
    "DRHP":   "Draft Red Herring Prospectus",
    "RHP":    "Red Herring Prospectus",
    "IPO":    "Initial Public Offering",
    "FPO":    "Follow-on Public Offer",
    "OFS":    "Offer for Sale",
    "ESOP":   "Employee Stock Ownership Plan",
    "ESPS":   "Employee Stock Purchase Scheme",
    "LODR":   "Listing Obligations and Disclosure Requirements",
    "SAST":   "Substantial Acquisition of Shares and Takeovers",
    "PIT":    "Prevention of Insider Trading",
    "UPSI":   "Unpublished Price Sensitive Information",
    "SDI":    "Structured Debt Instrument",
    "InvIT":  "Infrastructure Investment Trust",
    "REIT":   "Real Estate Investment Trust",

    # ── Valuation / Returns ───────────────────────────────────────────────
    "PE":     "Price to Earnings",
    "PB":     "Price to Book",
    "PS":     "Price to Sales",
    "EV":     "Enterprise Value",
    "EBITDA": "Earnings Before Interest Tax Depreciation and Amortisation",
    "EBIT":   "Earnings Before Interest and Tax",
    "PAT":    "Profit After Tax",
    "PBT":    "Profit Before Tax",
    "EPS":    "Earnings Per Share",
    "DPS":    "Dividend Per Share",
    "BV":     "Book Value",
    "NAV":    "Net Asset Value",
    "FCF":    "Free Cash Flow",
    "OCF":    "Operating Cash Flow",
    "ROCE":   "Return on Capital Employed",
    "ROE":    "Return on Equity",
    "ROA":    "Return on Assets",
    "ROIC":   "Return on Invested Capital",
    "CAGR":   "Compound Annual Growth Rate",
    "YoY":    "Year on Year",
    "QoQ":    "Quarter on Quarter",
    "MoM":    "Month on Month",
    "TTM":    "Trailing Twelve Months",
    "LTM":    "Last Twelve Months",
    "NTM":    "Next Twelve Months",

    # ── Insurance ─────────────────────────────────────────────────────────
    "VNB":    "Value of New Business",
    "NBM":    "New Business Margin",
    "EV":     "Embedded Value",
    "APE":    "Annualised Premium Equivalent",
    "GWP":    "Gross Written Premium",
    "NWP":    "Net Written Premium",
    "IRDA":   "Insurance Regulatory and Development Authority",
    "IRDAI":  "Insurance Regulatory and Development Authority of India",
    "LIC":    "Life Insurance Corporation",
    "ULIPs":  "Unit Linked Insurance Plans",
    "ULIP":   "Unit Linked Insurance Plan",

    # ── NBFC / Fintech ────────────────────────────────────────────────────
    "NBFC":   "Non-Banking Financial Company",
    "HFC":    "Housing Finance Company",
    "MFI":    "Microfinance Institution",
    "BC":     "Business Correspondent",
    "PPI":    "Prepaid Payment Instrument",
    "UPI":    "Unified Payments Interface",
    "NACH":   "National Automated Clearing House",
    "IMPS":   "Immediate Payment Service",
    "NEFT":   "National Electronic Funds Transfer",
    "RTGS":   "Real Time Gross Settlement",
    "AML":    "Anti Money Laundering",
    "KYC":    "Know Your Customer",
    "CKYC":   "Central Know Your Customer",
    "FLDG":   "First Loss Default Guarantee",
    "DLG":    "Default Loss Guarantee",
    "LSP":    "Lending Service Provider",
    "AA":     "Account Aggregator",
    "OCEN":   "Open Credit Enablement Network",
    "ONDC":   "Open Network for Digital Commerce",

    # ── Government / Macro ───────────────────────────────────────────────
    "GDP":    "Gross Domestic Product",
    "GVA":    "Gross Value Added",
    "CAD":    "Current Account Deficit",
    "FAD":    "Fiscal Account Deficit",
    "GFD":    "Gross Fiscal Deficit",
    "FRBM":   "Fiscal Responsibility and Budget Management",
    "GST":    "Goods and Services Tax",
    "CGST":   "Central Goods and Services Tax",
    "SGST":   "State Goods and Services Tax",
    "IGST":   "Integrated Goods and Services Tax",
    "TDS":    "Tax Deducted at Source",
    "TCS":    "Tax Collected at Source",  # also Tata Consultancy — context-dependent
    "MAT":    "Minimum Alternate Tax",
    "STT":    "Securities Transaction Tax",
    "LTCG":   "Long Term Capital Gains",
    "STCG":   "Short Term Capital Gains",
    "MCA":    "Ministry of Corporate Affairs",
    "ROC":    "Registrar of Companies",
    "CIN":    "Corporate Identification Number",

    # ── Technology / IT ───────────────────────────────────────────────────
    "TCV":    "Total Contract Value",
    "ACV":    "Annual Contract Value",
    "ARR":    "Annual Recurring Revenue",
    "MRR":    "Monthly Recurring Revenue",
    "SaaS":   "Software as a Service",
    "BPO":    "Business Process Outsourcing",
    "KPO":    "Knowledge Process Outsourcing",
    "GCC":    "Global Capability Centre",
    "BFSI":   "Banking Financial Services and Insurance",

    # ── Mutual Funds ──────────────────────────────────────────────────────
    "AUM":    "Assets Under Management",
    "SIP":    "Systematic Investment Plan",
    "SWP":    "Systematic Withdrawal Plan",
    "STP":    "Systematic Transfer Plan",
    "ELSS":   "Equity Linked Savings Scheme",
    "FOF":    "Fund of Funds",
    "ETF":    "Exchange Traded Fund",

    # ── Fiscal Calendar ───────────────────────────────────────────────────
    "FY":     "Financial Year",
    "H1":     "First Half of Financial Year",
    "H2":     "Second Half of Financial Year",
    "Q1":     "First Quarter",
    "Q2":     "Second Quarter",
    "Q3":     "Third Quarter",
    "Q4":     "Fourth Quarter",

    # ── Commodities / Energy ──────────────────────────────────────────────
    "OPEC":   "Organization of the Petroleum Exporting Countries",
    "LNG":    "Liquefied Natural Gas",
    "CNG":    "Compressed Natural Gas",
    "PNG":    "Piped Natural Gas",
    "ONGC":   "Oil and Natural Gas Corporation",
    "IOC":    "Indian Oil Corporation",
    "BPCL":   "Bharat Petroleum Corporation Limited",

    # ── Pharma / Healthcare ───────────────────────────────────────────────
    "ANDA":   "Abbreviated New Drug Application",
    "NDA":    "New Drug Application",
    "API":    "Active Pharmaceutical Ingredient",
    "USFDA":  "United States Food and Drug Administration",
    "WHO":    "World Health Organization",
    "GMP":    "Good Manufacturing Practice",

    # ── Risk / Compliance ─────────────────────────────────────────────────
    "VaR":    "Value at Risk",
    "CVaR":   "Conditional Value at Risk",
    "ES":     "Expected Shortfall",
    "ICAAP":  "Internal Capital Adequacy Assessment Process",
    "SREP":   "Supervisory Review and Evaluation Process",
    "BASEL":  "Basel Capital Accord",
    "ESG":    "Environmental Social and Governance",
    "CSR":    "Corporate Social Responsibility",
}

# Context-ambiguous acronyms that need LLM disambiguation
# Maps acronym → list of possible expansions
_AMBIGUOUS_ACRONYMS: dict[str, list[str]] = {
    "TCS":  ["Tax Collected at Source", "Tata Consultancy Services"],
    "NII":  ["Net Interest Income", "Non-Institutional Investor"],
    "EV":   ["Enterprise Value", "Electric Vehicle", "Embedded Value"],
    "ECB":  ["External Commercial Borrowing", "European Central Bank"],
    "PE":   ["Price to Earnings", "Private Equity"],
    "BC":   ["Business Correspondent", "Before Christ"],
    "PS":   ["Price to Sales", "Persistent Systems", "Press Statement"],
    "MCA":  ["Ministry of Corporate Affairs", "Master of Computer Applications"],
    "NDA":  ["New Drug Application", "Non-Disclosure Agreement",
             "National Democratic Alliance"],
    "SBI":  ["State Bank of India", "Small Business Innovation"],
    "LIC":  ["Life Insurance Corporation", "Letter of Intent to Claim"],
    "ES":   ["Expected Shortfall", "Earnings Surprise"],
    "PIT":  ["Prevention of Insider Trading", "Pit Stop"],
    "AA":   ["Account Aggregator", "Alcoholics Anonymous"],
    "FPO":  ["Follow-on Public Offer", "Farmer Producer Organisation"],
    "ARR":  ["Annual Recurring Revenue", "Annualised Rate of Return"],
}

# Pattern to detect candidate acronyms:
# - ALL CAPS, 2-6 chars, word boundary
# - Optionally followed by digits (e.g., Q3, H1, FY26)
_ACRONYM_PATTERN = re.compile(
    r'\b([A-Z]{2,6}(?:\d{0,4})?)\b'
)

# Pattern to detect already-expanded acronyms like "NIM (Net Interest Margin)"
# or "Net Interest Margin (NIM)" — skip these to stay idempotent
_ALREADY_EXPANDED_PATTERN = re.compile(
    r'\b[A-Z]{2,6}\s*\([^)]{5,60}\)'        # NIM (Net Interest Margin)
    r'|\b[A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+){1,6}\s*\([A-Z]{2,6}\)',  # Net Interest Margin (NIM)
)

# Groq system prompt for disambiguation
_GROQ_SYSTEM_PROMPT = """You are a financial acronym expert specializing in Indian 
capital markets, banking regulation, and corporate finance.

Your task: Given an acronym and the sentence it appears in, return ONLY the single 
most appropriate full form — nothing else. No explanation, no punctuation, just 
the expansion.

Context: Indian financial markets (NSE, BSE, RBI, SEBI regulated entities).
If genuinely ambiguous, prefer the banking/finance interpretation.

Examples:
  Query: expand "TCS" in "TCS reported strong Q3 results"
  Response: Tata Consultancy Services

  Query: expand "TCS" in "TCS at 0.1% on equity gains"
  Response: Tax Collected at Source

  Query: expand "EV" in "EV/EBITDA multiple is 12x"
  Response: Enterprise Value

  Query: expand "EV" in "EV adoption in India growing 40% YoY"
  Response: Electric Vehicle"""


# ─────────────────────────────────────────────
# ACRONYM RESOLVER CLASS
# ─────────────────────────────────────────────

class AcronymResolver:
    """
    Agent 1: Expands financial acronyms in queries before retrieval.

    Instantiation:
        from backend.agents.agent1_acronym_resolver import acronym_resolver

    Usage:
        expanded = await acronym_resolver.resolve(
            "What is HDFC Bank's NIM and GNPA in Q3 FY26?"
        )
        # → "What is HDFC Bank's Net Interest Margin (NIM) and
        #    Gross Non-Performing Asset (GNPA) in Q3 FY26?"

        acronyms = acronym_resolver.detect_acronyms("NIM rose 20bps QoQ")
        # → ["NIM", "QoQ"]
    """

    def __init__(self):
        # Merge built-in dict with financial_terms.py FINANCIAL_TERMS
        self._dict: dict[str, str] = {
            **_BUILTIN_ACRONYM_DICT,
            **{k.upper(): v for k, v in FINANCIAL_TERMS.items()},
        }
        # Build case-insensitive lookup
        self._dict_lower: dict[str, str] = {
            k.lower(): v for k, v in self._dict.items()
        }

        # Runtime stats
        self._total_resolved: int   = 0
        self._total_groq:     int   = 0
        self._total_unknown:  int   = 0
        self._total_ms:       float = 0.0

        logger.info(
            f"[ACRONYM] AcronymResolver ready | "
            f"dict_size={len(self._dict)} acronyms"
        )

    # ─────────────────────────────────────────
    # PUBLIC: resolve
    # ─────────────────────────────────────────

    async def resolve(
        self,
        query:          str,
        use_llm:        bool = True,
        max_expansions: int  = 8,
    ) -> str:
        """
        Expands all resolvable acronyms in a query string.
        Non-destructive: only appends "(expansion)" after each acronym.

        Args:
            query:          The raw user query string
            use_llm:        Allow Groq fallback for ambiguous acronyms
                            (default True; set False for strict offline mode)
            max_expansions: Cap on expansions per query to control latency
                            (default 8)

        Returns:
            Expanded query string with inline acronym definitions.
            Returns original query unchanged on any failure.

        Latency targets:
            Dict-only:     < 5ms
            With Groq:     < 250ms (one LLM call per ambiguous acronym)

        Examples:
            await resolver.resolve("What is RBI's view on NIM compression?")
            → "What is Reserve Bank of India (RBI)'s view on
               Net Interest Margin (NIM) compression?"

            await resolver.resolve("TCS Q3 FY26 EBITDA margins?")
            → "Tata Consultancy Services (TCS) Q3 FY26 Earnings Before
               Interest Tax Depreciation and Amortisation (EBITDA) margins?"
        """
        if not query or not query.strip():
            return query

        start_ms = time.perf_counter()

        # Find already-expanded spans to skip them
        already_expanded = set()
        for m in _ALREADY_EXPANDED_PATTERN.finditer(query):
            # Extract the uppercase acronym from already-expanded span
            caps = re.findall(r'\b[A-Z]{2,6}\b', m.group(0))
            already_expanded.update(caps)

        # Detect candidate acronyms
        candidates = self.detect_acronyms(query)

        # Filter: remove already expanded, limit count
        to_expand = [
            a for a in candidates
            if a not in already_expanded
        ][:max_expansions]

        if not to_expand:
            logger.debug(
                f"[ACRONYM] No expandable acronyms found | "
                f"query='{query[:60]}'"
            )
            return query

        logger.debug(
            f"[ACRONYM] Candidates: {to_expand} | "
            f"query='{query[:60]}'"
        )

        # Build expansion map for this query
        expansion_map: dict[str, str] = {}
        groq_needed:   list[str]      = []

        for acronym in to_expand:
            expansion = self._lookup(acronym)

            if expansion:
                expansion_map[acronym] = expansion
                self._total_resolved  += 1

            elif acronym in _AMBIGUOUS_ACRONYMS and use_llm:
                groq_needed.append(acronym)

            else:
                self._total_unknown += 1
                logger.debug(f"[ACRONYM] Unknown acronym: {acronym}")

        # Resolve ambiguous ones via Groq (all in parallel)
        if groq_needed:
            groq_results = await self._resolve_ambiguous_batch(
                acronyms = groq_needed,
                context  = query,
            )
            expansion_map.update(groq_results)
            self._total_groq   += len(groq_results)
            self._total_resolved += len(groq_results)

        if not expansion_map:
            return query

        # Apply expansions to query text
        expanded_query = self._apply_expansions(query, expansion_map)

        duration_ms     = (time.perf_counter() - start_ms) * 1000
        self._total_ms += duration_ms

        logger.info(
            f"[ACRONYM] Resolved | "
            f"found={len(to_expand)} | "
            f"expanded={len(expansion_map)} | "
            f"groq={len(groq_needed)} | "
            f"{duration_ms:.1f}ms | "
            f"expansions={list(expansion_map.keys())}"
        )
        log_metric("acronym_resolve_ms", duration_ms, unit="ms")

        return expanded_query

    # ─────────────────────────────────────────
    # PUBLIC: detect_acronyms
    # ─────────────────────────────────────────

    def detect_acronyms(self, query: str) -> list[str]:
        """
        Detects candidate acronyms in query text.
        Returns unique list preserving order of first appearance.

        Detection rules:
            - All-caps words, 2–6 characters
            - Optionally followed by 1-4 digits (Q3, FY26, H1)
            - Word boundary on both sides
            - Excludes: single letters, common English words
            - Deduplicated (preserves first occurrence order)

        Args:
            query: Raw query string

        Returns:
            List of detected acronym strings (uppercase)

        Examples:
            detect_acronyms("HDFC Bank NIM vs GNPA in Q3 FY26")
            → ["NIM", "GNPA", "FY"]
            # Note: "HDFC" is in the dict as company name → included
            # "Q3" matched as Q + digit suffix
        """
        if not query:
            return []

        seen:   set[str]  = set()
        result: list[str] = []

        for match in _ACRONYM_PATTERN.finditer(query):
            candidate = match.group(1)

            # Skip pure numbers
            if candidate.isdigit():
                continue

            # Skip single-letter matches caught by boundary
            if len(candidate) < 2:
                continue

            # Skip common English words that happen to be all-caps
            # (e.g., query typed in caps lock)
            if candidate.lower() in _COMMON_ENGLISH_WORDS:
                continue

            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)

        return result

    # ─────────────────────────────────────────
    # PUBLIC: expand_with_context
    # ─────────────────────────────────────────

    async def expand_with_context(
        self,
        acronym: str,
        context: str,
    ) -> str:
        """
        Uses Groq to disambiguate a single ambiguous acronym in context.

        Args:
            acronym: The acronym to expand (e.g., "TCS")
            context: The sentence containing it (e.g., "TCS Q3 revenue...")

        Returns:
            Expansion string (e.g., "Tata Consultancy Services")
            Falls back to acronym itself on error.

        Examples:
            await resolver.expand_with_context("TCS", "TCS Q3 FY26 margins rose")
            → "Tata Consultancy Services"

            await resolver.expand_with_context("EV", "EV/EBITDA is 14x")
            → "Enterprise Value"
        """
        if not acronym or not context:
            return acronym

        start_ms = time.perf_counter()

        prompt = (
            f'expand "{acronym}" in this sentence: "{context}"'
        )

        try:
            response = await llm_client.chat(
                system_prompt = _GROQ_SYSTEM_PROMPT,
                user_message  = prompt,
                model         = "groq/llama-3.1-8b-instant",
                max_tokens    = 20,
                temperature   = 0.0,
            )

            expansion = response.content.strip().strip('"\'.,')

            # Sanity check: expansion should be longer than the acronym
            if len(expansion) <= len(acronym):
                logger.debug(
                    f"[ACRONYM] Groq returned suspicious expansion: "
                    f"'{acronym}' → '{expansion}' — using fallback"
                )
                return self._lookup(acronym) or acronym

            duration_ms = (time.perf_counter() - start_ms) * 1000
            logger.debug(
                f"[ACRONYM] Groq expansion | "
                f"'{acronym}' → '{expansion}' | "
                f"{duration_ms:.0f}ms"
            )
            return expansion

        except Exception as e:
            logger.warning(
                f"[ACRONYM] Groq expansion failed for '{acronym}': {e} "
                f"— falling back to dict"
            )
            return self._lookup(acronym) or acronym

    # ─────────────────────────────────────────
    # PUBLIC: add_custom_acronym
    # ─────────────────────────────────────────

    def add_custom_acronym(self, acronym: str, expansion: str) -> None:
        """
        Adds a custom acronym to the runtime dictionary.
        Useful for domain-specific or client-specific terms.

        Args:
            acronym:   The acronym (will be uppercased)
            expansion: Full form

        Examples:
            resolver.add_custom_acronym("FTHG", "FinThesisGuard")
            resolver.add_custom_acronym("AARK", "ArkAngel Financial Solutions")
        """
        acronym_upper = acronym.upper()
        self._dict[acronym_upper]       = expansion
        self._dict_lower[acronym.lower()] = expansion
        logger.debug(
            f"[ACRONYM] Custom acronym added: '{acronym_upper}' → '{expansion}'"
        )

    # ─────────────────────────────────────────
    # PUBLIC: get_stats
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns runtime statistics for monitoring."""
        avg_ms = (
            round(self._total_ms / self._total_resolved, 2)
            if self._total_resolved > 0 else 0.0
        )
        return {
            "dict_size":       len(self._dict),
            "total_resolved":  self._total_resolved,
            "total_groq_calls": self._total_groq,
            "total_unknown":   self._total_unknown,
            "total_ms":        round(self._total_ms, 1),
            "avg_ms_per_call": avg_ms,
        }

    # ─────────────────────────────────────────
    # PRIVATE: lookup
    # ─────────────────────────────────────────

    def _lookup(self, acronym: str) -> Optional[str]:
        """
        Looks up acronym in dictionary.
        Tries exact match first, then case-insensitive fallback.
        Returns None if not found.
        """
        # Exact match (all-caps)
        if acronym in self._dict:
            return self._dict[acronym]

        # Case-insensitive fallback (e.g., "Nim" or "nim")
        lower = acronym.lower()
        if lower in self._dict_lower:
            return self._dict_lower[lower]

        # Strip trailing digits for FY-style: "FY26" → "FY"
        stripped = re.sub(r'\d+$', '', acronym)
        if stripped and stripped != acronym:
            if stripped in self._dict:
                return self._dict[stripped]

        return None

    # ─────────────────────────────────────────
    # PRIVATE: resolve ambiguous batch
    # ─────────────────────────────────────────

    async def _resolve_ambiguous_batch(
        self,
        acronyms: list[str],
        context:  str,
    ) -> dict[str, str]:
        """
        Resolves multiple ambiguous acronyms in parallel Groq calls.
        Returns {acronym: expansion} dict.
        """
        tasks   = [
            self.expand_with_context(acr, context)
            for acr in acronyms
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        expansion_map: dict[str, str] = {}
        for acr, result in zip(acronyms, results):
            if isinstance(result, Exception):
                logger.warning(
                    f"[ACRONYM] Batch Groq failed for '{acr}': {result}"
                )
                # Fall back to dict or first candidate
                fallback = (
                    self._lookup(acr)
                    or _AMBIGUOUS_ACRONYMS.get(acr, [acr])[0]
                )
                expansion_map[acr] = fallback
            else:
                expansion_map[acr] = result

        return expansion_map

    # ─────────────────────────────────────────
    # PRIVATE: apply expansions
    # ─────────────────────────────────────────

    def _apply_expansions(
        self,
        query:         str,
        expansion_map: dict[str, str],
    ) -> str:
        """
        Applies expansion_map to query text.
        Inserts "(Full Form)" immediately after each acronym.

        Handles:
            - Word boundary matching to avoid partial replacements
            - Ordinal suffixes: "Q3" → "Q3 (Third Quarter)"
            - Possessives: "RBI's" → "Reserve Bank of India (RBI)'s"
            - Multiple occurrences: only expands FIRST occurrence
              of each acronym to avoid clutter

        Examples:
            _apply_expansions(
                "NIM and GNPA both improved",
                {"NIM": "Net Interest Margin", "GNPA": "Gross Non-Performing Asset"}
            )
            → "Net Interest Margin (NIM) and Gross Non-Performing Asset (GNPA) both improved"
        """
        result   = query
        expanded = set()   # Track which acronyms already expanded (first-only)

        # Sort by length descending to handle longer acronyms first
        # (prevents "NPA" matching inside "GNPA")
        sorted_acronyms = sorted(
            expansion_map.keys(),
            key=len,
            reverse=True,
        )

        for acronym in sorted_acronyms:
            if acronym in expanded:
                continue

            expansion = expansion_map[acronym]

            # Pattern: word boundary + acronym + optional possessive
            # Handles: NIM, NIM's, NIM), NIM.
            pattern = re.compile(
                rf'\b({re.escape(acronym)})(\'s|s\b)?\b',
                re.IGNORECASE,
            )

            def _replacer(m: re.Match, _acr=acronym, _exp=expansion) -> str:
                matched_acr = m.group(1)
                suffix      = m.group(2) or ""
                return f"{_exp} ({matched_acr}){suffix}"

            new_result, count = pattern.subn(_replacer, result, count=1)

            if count > 0:
                result   = new_result
                expanded.add(acronym)

        return result


# ─────────────────────────────────────────────
# COMMON ENGLISH WORDS TO SKIP
# These all-caps strings are NOT acronyms in financial context
# ─────────────────────────────────────────────

_COMMON_ENGLISH_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "not", "but", "for", "nor",
    "so", "yet", "in", "on", "at", "to", "up", "as", "is",
    "it", "be", "do", "go", "if", "by", "my", "we", "he",
    "she", "they", "you", "me", "us", "him", "her", "our",
    "its", "was", "are", "has", "had", "did", "can", "may",
    "will", "new", "old", "big", "low", "high", "top", "key",
    "all", "any", "few", "more", "most", "per", "due", "via",
    "vs", "etc", "inc", "ltd", "pvt",
})


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

acronym_resolver = AcronymResolver()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "acronym_resolver",
    "AcronymResolver",
    "_BUILTIN_ACRONYM_DICT",
    "_AMBIGUOUS_ACRONYMS",
]
