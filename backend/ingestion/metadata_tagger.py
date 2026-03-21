# backend/ingestion/metadata_tagger.py
"""
FinThesisGuard AI — Metadata Tagger
ArkAngel Financial Solutions

Auto-tags every document chunk with rich, structured metadata
before Pinecone upsert. Metadata powers filtered retrieval,
authority-weighted re-ranking, and conflict detection.

Pipeline position:
    pdf_parser → chunker → [metadata_tagger] → embedder → pinecone

Every Pinecone vector metadata dict produced here contains:
    source_type, company, sector, date, quarter, financial_year,
    authority_weight, doc_type, extracted figures (NIM, NPA, etc.)
"""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from backend.config import settings
from backend.utils.logger import logger, log_metric
from backend.utils.financial_terms import FINANCIAL_TERMS


# ─────────────────────────────────────────────
# AUTHORITY WEIGHTS (source → weight)
# Mirrors retriever.py for consistency
# ─────────────────────────────────────────────

AUTHORITY_WEIGHTS: dict[str, float] = {
    "rbi":                 1.00,
    "sebi":                1.00,
    "mca":                 0.95,
    "tax_law":             0.90,
    "annual_report":       0.75,
    "earnings_transcript": 0.65,
    "broker_research":     0.40,
    "news":                0.25,
    "blog":                0.15,
    "unknown":             0.20,
}


# ─────────────────────────────────────────────
# SOURCE TYPE DETECTION RULES
# (pattern, source_type) — evaluated in order
# ─────────────────────────────────────────────

# Filename-based detection patterns
# AFTER (fixed):
# backend/ingestion/metadata_tagger.py
# ✅ PRODUCTION-READY FIX — Copy-paste entire blocks

FILENAME_SOURCE_PATTERNS: list[tuple[str, str]] = [
    (r'sebi',                                          "sebi"),     # ✅ #1 PRIORITY
    (r'rbi|reserve[\s_]?bank',                         "rbi"),
    (r'mca|ministry[\s_]?of[\s_]?corporate',           "mca"),
    (r'income[\s_]?tax|incometax|cbdt|section\s*1',    "income_tax"),  # ✅ Tax
    (r'irdai|insurance[\s_]?regulatory',               "irdai"),      # ✅ Insurance
    (r'budget|union[\s_]?budget',                      "tax_law"),
    (r'monetary[\s_]?policy|mpc[\s_]?statement',       "rbi"),
    (r'annual[\s_]?report|ar\b|\bdrhp',                "annual_report"),
    (r'earnings?|concall|transcript|q[1-4][\s_]?fy',   "earnings_transcript"),
    (r'investor[\s_]?presentation',                    "annual_report"),
    (r'broker|research[\s_]?note|initiating|coverage', "broker_research"),
    (r'press[\s_]?release',                            "news"),
    (r'hindu|businessline|economictimes|mint|livemint|timesofindia|ndtv', "news"),  # ✅ Media
    (r'news|article|headline',                         "news"),
    (r'blog|opinion|editorial',                        "blog"),
]

CONTENT_SOURCE_PATTERNS: list[tuple[str, str]] = [
    (r'reserve[\s_]?bank[\s_]?of[\s_]?india|rbi[\s_]?circular|rbi[\s_]?notification', "rbi"),
    (r'sebi[\s_]?circular|sebi[\s_]?notification|securities[\s_]+and[\s_]+exchange[\s_]+board|sebi\.gov\.in|sebi\.sebi', "sebi"),  # ✅ Robust
    (r'ministry[\s_]+of[\s_]+corporate[\s_]+affairs|companies[\s_]+act', "mca"),
    (r'income\.tax[\s_]+act|section[\s_]+195|cbdt|income\.tax[\s_]+department', "income_tax"),
    (r'irdai|insurance[\s_]+regulatory[\s_]+and[\s_]+development', "irdai"),
    (r'monetary[\s_]+policy[\s_]+(?:committee|statement|report)', "rbi"),
    (r'we[\s_]+hereby[\s_]+inform|pursuant[\s_]+to[\s_]+regulation', "sebi"),
    (r'management[\s_]+discussion|md&a|annual[\s_]+report', "annual_report"),
    (r'earnings[\s_]+call|concall|q&a[\s_]+session', "earnings_transcript"),
    (r'we[\s_]+recommend|target[\s_]+price|rating:[\s_]*(?:buy|sell|hold)', "broker_research"),
    (r'reports?\.?\s+(?:that|said|stated)', "news"),
]


# ─────────────────────────────────────────────
# COMPANY DETECTION
# ─────────────────────────────────────────────

# (pattern, canonical_name, sector)
COMPANY_PATTERNS: list[tuple[str, str, str]] = [
    # Banking
    (r'\bhdfc\s+bank\b',                          "HDFC Bank",              "banking"),
    (r'\bhdfc\s+ltd\b|\bhdfc\s+limited\b',        "HDFC Ltd",               "banking"),
    (r'\bicici\s+bank\b',                          "ICICI Bank",             "banking"),
    (r'\bicici\s+prudential\b',                    "ICICI Prudential",       "insurance"),
    (r'\bstate\s+bank\s+of\s+india\b|\bsbi\b',    "SBI",                    "banking"),
    (r'\baxis\s+bank\b',                           "Axis Bank",              "banking"),
    (r'\bkotak\s+mahindra\s+bank\b|\bkotak\s+bank\b|\bkotak\b', "Kotak Mahindra Bank", "banking"),
    (r'\bindusind\s+bank\b',                       "IndusInd Bank",          "banking"),
    (r'\bbandhan\s+bank\b',                        "Bandhan Bank",           "banking"),
    (r'\bidfc\s+first\s+bank\b|\bidfc\s+bank\b',  "IDFC First Bank",        "banking"),
    (r'\byes\s+bank\b',                            "Yes Bank",               "banking"),
    (r'\bfederal\s+bank\b',                        "Federal Bank",           "banking"),
    (r'\bpunjab\s+national\s+bank\b|\bpnb\b',     "PNB",                    "banking"),
    (r'\bbank\s+of\s+baroda\b|\bbob\b',            "Bank of Baroda",         "banking"),
    (r'\bcanara\s+bank\b',                         "Canara Bank",            "banking"),
    (r'\bunion\s+bank\b',                          "Union Bank",             "banking"),
    (r'\bau\s+small\s+finance\b',                  "AU Small Finance Bank",  "banking"),
    (r'\bequitas\b',                               "Equitas Small Finance",  "banking"),
    (r'\butkarsh\b',                               "Utkarsh Small Finance",  "banking"),

    # NBFCs
    (r'\bbajaj\s+finance\b',                       "Bajaj Finance",          "nbfc"),
    (r'\bbajaj\s+finserv\b',                       "Bajaj Finserv",          "nbfc"),
    (r'\bmuthoot\s+finance\b',                     "Muthoot Finance",        "nbfc"),
    (r'\bmanappuram\s+finance\b',                  "Manappuram Finance",     "nbfc"),
    (r'\bchola(?:mandalam)?\b',                    "Cholamandalam Finance",  "nbfc"),
    (r'\bpiramal\s+(?:enterprises|finance)\b',     "Piramal Finance",        "nbfc"),
    (r'\bm&m\s+financial\b|\bmahindra\s+finance\b', "M&M Financial",        "nbfc"),
    (r'\bshriram\s+(?:transport|finance)\b',       "Shriram Finance",        "nbfc"),
    (r'\blic\s+housing\b',                         "LIC Housing Finance",    "nbfc"),

    # Insurance
    (r'\blic\b|life\s+insurance\s+corporation',    "LIC",                    "insurance"),
    (r'\bsbi\s+life\b',                            "SBI Life",               "insurance"),
    (r'\bhdfc\s+life\b',                           "HDFC Life",              "insurance"),
    (r'\bmax\s+life\b',                            "Max Life",               "insurance"),
    (r'\bstar\s+health\b',                         "Star Health",            "insurance"),

    # IT
    (r'\btata\s+consultancy\b|\btcs\b',            "TCS",                    "technology"),
    (r'\binfosys\b',                               "Infosys",                "technology"),
    (r'\bwipro\b',                                 "Wipro",                  "technology"),
    (r'\bhcl\s+tech(?:nologies)?\b',               "HCL Technologies",       "technology"),
    (r'\btech\s+mahindra\b',                       "Tech Mahindra",          "technology"),
    (r'\bmphasis\b',                               "Mphasis",                "technology"),
    (r'\bltimindtree\b|\bl&t\s+infotech\b',        "LTIMindtree",            "technology"),
    (r'\bpersistent\s+systems\b',                  "Persistent Systems",     "technology"),
    (r'\bcoforge\b',                               "Coforge",                "technology"),
    (r'\bzomato\b',                                "Zomato",                 "technology"),
    (r'\bswiggy\b',                                "Swiggy",                 "technology"),
    (r'\bpaytm\b|\bone97\b',                       "Paytm",                  "fintech"),
    (r'\bnaukri\b|\binfo\s*edge\b',                "Info Edge",              "technology"),
    (r'\bindiamart\b',                             "IndiaMART",              "technology"),

    # Energy / Oil & Gas
    (r'\breliance\s+industries\b|\bril\b',         "Reliance Industries",    "energy"),
    (r'\breliance\s+jio\b|\bjio\b',                "Reliance Jio",           "telecom"),
    (r'\bontc\b|\boil\s+and\s+natural\s+gas\b',   "ONGC",                   "energy"),
    (r'\bcoal\s+india\b',                          "Coal India",             "energy"),
    (r'\bntpc\b',                                  "NTPC",                   "energy"),
    (r'\bpower\s+grid\b',                          "Power Grid",             "energy"),
    (r'\badani\s+(?:green|power|energy)\b',        "Adani Energy",           "energy"),
    (r'\badani\s+(?:enterprises|ports|total)\b',   "Adani Enterprises",      "conglomerate"),
    (r'\bbpcl\b|bharat\s+petroleum\b',             "BPCL",                   "energy"),
    (r'\biocl\b|indian\s+oil\b',                   "Indian Oil",             "energy"),

    # Auto
    (r'\btata\s+motors\b',                         "Tata Motors",            "automobile"),
    (r'\bmahindra\s+(?:&|and)\s+mahindra\b|\bm&m\b', "Mahindra & Mahindra", "automobile"),
    (r'\bmaruti\s+suzuki\b|\bmaruti\b',            "Maruti Suzuki",          "automobile"),
    (r'\bhero\s+(?:motocorp|honda)\b',             "Hero MotoCorp",          "automobile"),
    (r'\bbajaj\s+auto\b',                          "Bajaj Auto",             "automobile"),
    (r'\btvs\s+motor\b',                           "TVS Motor",              "automobile"),
    (r'\beicher\s+motors\b',                       "Eicher Motors",          "automobile"),

    # Pharma
    (r'\bsun\s+pharma(?:ceutical)?\b',             "Sun Pharma",             "pharma"),
    (r'\bdr\.\s*reddy\b|\bdr\s+reddy\b',           "Dr. Reddy's",            "pharma"),
    (r'\bcipl[ao]\b',                              "Cipla",                  "pharma"),
    (r'\bdivi(?:s)?\s+lab\b',                      "Divi's Labs",            "pharma"),
    (r'\bbiocon\b',                                "Biocon",                 "pharma"),
    (r'\baurobindo\b',                             "Aurobindo Pharma",       "pharma"),

    # FMCG / Consumer
    (r'\bhul\b|hindustan\s+unilever\b',            "HUL",                    "fmcg"),
    (r'\bitc\s+limited\b|\bitc\b',                 "ITC",                    "fmcg"),
    (r'\bnestl[eé]\s+india\b',                     "Nestlé India",           "fmcg"),
    (r'\bbritannia\b',                             "Britannia",              "fmcg"),
    (r'\bdabur\b',                                 "Dabur",                  "fmcg"),
    (r'\bgodrej\s+consumer\b|\bgodrej\b',          "Godrej Consumer",        "fmcg"),
    (r'\bmarico\b',                                "Marico",                 "fmcg"),
    (r'\bcolgate\b',                               "Colgate",                "fmcg"),

    # Conglomerate / Infra
    (r'\btata\s+steel\b',                          "Tata Steel",             "metals"),
    (r'\bjsw\s+steel\b',                           "JSW Steel",              "metals"),
    (r'\bhindal(?:co)?\b',                         "Hindalco",               "metals"),
    (r'\bvedanta\b',                               "Vedanta",                "metals"),
    (r'\blarsen\s+(?:&|and)\s+toubro\b|\bl&t\b',  "L&T",                    "infrastructure"),
    (r'\babbb?\s+india\b',                         "ABB India",              "capital_goods"),
    (r'\bsiemens\s+india\b',                       "Siemens India",          "capital_goods"),
    (r'\basian\s+paints\b',                        "Asian Paints",           "chemicals"),
    (r'\bpidilite\b',                              "Pidilite",               "chemicals"),
    (r'\bsrf\s+limited\b|\bsrf\b',                "SRF",                    "chemicals"),

    # Exchanges / Infra
    (r'\bnse\b|national\s+stock\s+exchange',       "NSE",                    "exchange"),
    (r'\bbse\b|bombay\s+stock\s+exchange',         "BSE",                    "exchange"),
    (r'\bnsdl\b',                                  "NSDL",                   "exchange"),
    (r'\bcdsl\b',                                  "CDSL",                   "exchange"),
    (r'\bcams\b',                                  "CAMS",                   "fintech"),
    (r'\bkfintech\b',                              "KFin Technologies",      "fintech"),
]

# Precompile company patterns
_COMPILED_COMPANY: list[tuple[re.Pattern, str, str]] = [
    (re.compile(p, re.IGNORECASE), name, sector)
    for p, name, sector in COMPANY_PATTERNS
]


# ─────────────────────────────────────────────
# SECTOR KEYWORD MAP
# ─────────────────────────────────────────────

SECTOR_KEYWORDS: dict[str, list[str]] = {
    "banking": [
        "nim", "casa", "gnpa", "nnpa", "crar", "npa", "deposit",
        "credit growth", "lending", "net interest", "provisioning",
        "slippage", "recovery", "restructured", "pcr", "llr",
        "current account", "savings account", "loan book",
    ],
    "technology": [
        "attrition", "headcount", "utilization", "deal wins",
        "total contract value", "tcv", "revenue per employee",
        "digital transformation", "cloud", "ai", "saas", "paas",
        "offshore", "onsite", "billing rate", "pipeline",
    ],
    "pharma": [
        "anda", "nda", "usfda", "api", "formulation", "r&d",
        "generic", "biosimilar", "clinical trial", "patent cliff",
        "abbreviated new drug application", "dossier",
    ],
    "nbfc": [
        "aum", "disbursement", "yield on assets", "cost of borrowing",
        "net interest spread", "stage 2", "stage 3", "ecl",
        "microfinance", "gold loan", "vehicle finance",
    ],
    "energy": [
        "crude", "refinery", "upstream", "downstream", "brent",
        "wti", "pipeline", "lng", "natural gas", "capacity addition",
        "renewable", "solar", "wind", "gw",
    ],
    "fmcg": [
        "volume growth", "price realization", "distribution",
        "rural", "urban", "market share", "brand", "skus",
        "gross margin", "ad spends",
    ],
    "automobile": [
        "ev", "electric vehicle", "wholesale", "retail", "dispatch",
        "penetration", "suv", "two wheeler", "three wheeler",
        "passenger vehicle", "commercial vehicle", "order book",
        "chip shortage", "blended asp",
    ],
    "infrastructure": [
        "order book", "l1", "ebitda margin", "capex", "execution",
        "toll", "concession", "bid pipeline", "hydro", "metro",
    ],
    "insurance": [
        "premium", "gdp", "vnb", "apefye", "renewal premium",
        "surrender", "persistency", "combined ratio", "loss ratio",
        "embedded value", "new business margin",
    ],
    "metals": [
        "steel", "aluminium", "copper", "zinc", "ebitda per tonne",
        "coking coal", "iron ore", "realisation", "volume",
    ],
    "regulatory": [
        "circular", "notification", "regulation", "compliance",
        "penalty", "enforcement", "directive", "guideline",
    ],
}


# ─────────────────────────────────────────────
# FINANCIAL FIGURE EXTRACTION PATTERNS
# ─────────────────────────────────────────────

# (metric_key, regex_pattern)
# Captures value + optional unit
FINANCIAL_FIGURE_PATTERNS: list[tuple[str, str]] = [
    # Net Interest Margin
    ("nim_pct",
     r'\bnim\b[\s\S]{0,30}?(\d{1,2}(?:\.\d{1,2})?)\s*%'
     r'|net\s+interest\s+margin[\s\S]{0,30}?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    # GNPA / NPA
    ("gnpa_pct",
     r'\bgnpa\b[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'
     r'|gross\s+npa[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    ("nnpa_pct",
     r'\bnnpa\b[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'
     r'|net\s+npa[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    # CASA ratio
    ("casa_ratio_pct",
     r'\bcasa\s+ratio\b[\s\S]{0,20}?(\d{2}(?:\.\d{1,2})?)\s*%'
     r'|\bcasa\b[\s\S]{0,20}?(\d{2}(?:\.\d{1,2})?)\s*%'),

    # CRAR / Capital Adequacy
    ("crar_pct",
     r'\bcrar\b[\s\S]{0,20}?(\d{2}(?:\.\d{1,2})?)\s*%'
     r'|capital\s+adequacy[\s\S]{0,20}?(\d{2}(?:\.\d{1,2})?)\s*%'),

    # Tier 1 Capital
    ("tier1_capital_pct",
     r'tier[\s\-]?i\s+(?:capital\s+)?ratio[\s\S]{0,20}?(\d{2}(?:\.\d{1,2})?)\s*%'),

    # PCR - Provision Coverage Ratio
    ("pcr_pct",
     r'\bpcr\b[\s\S]{0,20}?(\d{2}(?:\.\d{1,2})?)\s*%'
     r'|provision\s+coverage[\s\S]{0,20}?(\d{2}(?:\.\d{1,2})?)\s*%'),

    # ROE / ROA
    ("roe_pct",
     r'\broe\b[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'
     r'|return\s+on\s+equity[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    ("roa_pct",
     r'\broa\b[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'
     r'|return\s+on\s+assets?[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    # PE ratio
    ("pe_ratio",
     r'\bp(?:/|-)e\s+(?:ratio|multiple)[\s\S]{0,20}?(\d{1,3}(?:\.\d{1,2})?)\s*x?'
     r'|\d{1,3}(?:\.\d{1,2})?\s*x\s+(?:fy\d{2,4}|trailing|forward)'),

    # EPS
    ("eps",
     r'\beps\b[\s\S]{0,20}?(?:rs\.?|₹|inr)?\s*(\d{1,5}(?:\.\d{1,2})?)'
     r'|earnings?\s+per\s+share[\s\S]{0,20}?(?:rs\.?|₹)?\s*(\d{1,5}(?:\.\d{1,2})?)'),

    # Revenue / Revenue growth
    ("revenue_crore",
     r'revenue[\s\S]{0,30}?(?:rs\.?|₹|inr)?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?)\s*cr(?:ore)?'),

    ("revenue_growth_pct",
     r'revenue[\s\S]{0,30}?(?:grew?|growth|increased?|up)\s+(?:by\s+)?(\d{1,3}(?:\.\d{1,2})?)\s*%'),

    # PAT (Profit After Tax)
    ("pat_crore",
     r'\bpat\b[\s\S]{0,30}?(?:rs\.?|₹)?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?)\s*cr(?:ore)?'
     r'|profit\s+after\s+tax[\s\S]{0,30}?(?:rs\.?|₹)?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?)\s*cr(?:ore)?'),

    # EBITDA margin
    ("ebitda_margin_pct",
     r'\bebitda\s+margin\b[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    # AUM (for NBFCs/MFs)
    ("aum_crore",
     r'\baum\b[\s\S]{0,30}?(?:rs\.?|₹)?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?)\s*cr(?:ore)?'
     r'|assets?\s+under\s+management[\s\S]{0,30}?(?:rs\.?|₹)?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?)\s*cr(?:ore)?'),

    # CAGR
    ("cagr_pct",
     r'\bcagr\b[\s\S]{0,30}?(\d{1,2}(?:\.\d{1,2})?)\s*%'
     r'|compound\s+annual\s+growth[\s\S]{0,30}?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    # Dividend yield
    ("dividend_yield_pct",
     r'dividend\s+yield[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    # Loan growth
    ("loan_growth_pct",
     r'\bloan\s+(?:book\s+)?growth[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'
     r'|advances?\s+grew?[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    # Deposit growth
    ("deposit_growth_pct",
     r'deposit\s+growth[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    # Repo rate
    ("repo_rate_pct",
     r'repo\s+rate[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    # Inflation / CPI
    ("cpi_inflation_pct",
     r'\bcpi\b[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'
     r'|(?:retail\s+)?inflation[\s\S]{0,20}?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    # Attrition (IT sector)
    ("attrition_pct",
     r'\battrition\s+(?:rate\s+)?(?:stood\s+at\s+|was\s+|of\s+)?(\d{1,2}(?:\.\d{1,2})?)\s*%'),

    # Deal TCV (IT sector)
    ("deal_tcv_bn_usd",
     r'\btcv\b[\s\S]{0,30}?\$?\s*(\d{1,4}(?:\.\d{1,2})?)\s*(?:billion|bn)\b'
     r'|total\s+contract\s+value[\s\S]{0,30}?\$?\s*(\d{1,4}(?:\.\d{1,2})?)\s*(?:bn|billion)'),
]

# Precompile all patterns
_COMPILED_FIGURE_PATTERNS: list[tuple[str, re.Pattern]] = [
    (key, re.compile(pattern, re.IGNORECASE | re.DOTALL))
    for key, pattern in FINANCIAL_FIGURE_PATTERNS
]


# ─────────────────────────────────────────────
# DATE EXTRACTION PATTERNS
# ─────────────────────────────────────────────

DATE_EXTRACTION_PATTERNS: list[str] = [
    r'\b(20[12]\d[-/]\d{1,2}[-/]\d{1,2})\b',           # 2026-01-15
    r'\b(\d{1,2}[-/]\d{1,2}[-/]20[12]\d)\b',            # 15/01/2026
    r'\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,]+20[12]\d)\b', # Jan 2026
    r'\b((?:january|february|march|april|may|june|july|august|september|october|november|december)[\s,]+20[12]\d)\b',
    r'\b(q[1-4]\s*fy\s*20[12]\d)\b',                    # Q3 FY2026
    r'\b(q[1-4]\s*fy\s*\d{2})\b',                       # Q3 FY26
    r'\bfy\s*(20[12]\d)\b',                              # FY2026
    r'\bfy\s*(\d{2})\b',                                 # FY26
    r'\b(20[12]\d[-–](?:20)?[12]\d)\b',                 # 2025-26 / 2025-2026
]

_COMPILED_DATE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in DATE_EXTRACTION_PATTERNS
]


# ─────────────────────────────────────────────
# QUARTER DETECTION
# ─────────────────────────────────────────────

QUARTER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bq1\s*fy\b|\bfirst\s+quarter\b|\bapr(?:il)?[-–]jun(?:e)?\b', re.IGNORECASE), "Q1"),
    (re.compile(r'\bq2\s*fy\b|\bsecond\s+quarter\b|\bjul(?:y)?[-–]sep(?:t)?\b', re.IGNORECASE), "Q2"),
    (re.compile(r'\bq3\s*fy\b|\bthird\s+quarter\b|\boct(?:ober)?[-–]dec(?:ember)?\b', re.IGNORECASE), "Q3"),
    (re.compile(r'\bq4\s*fy\b|\bfourth\s+quarter\b|\bjan(?:uary)?[-–]mar(?:ch)?\b', re.IGNORECASE), "Q4"),
]

FY_PATTERN = re.compile(
    r'\bfy\s*(?:20)?([2-9]\d)\b'       # FY26, FY2026
    r'|\b20([2-9]\d)[-–](?:20)?[0-9]{2}\b',  # 2025-26
    re.IGNORECASE,
)


# ─────────────────────────────────────────────
# TAGGED METADATA DATACLASS
# ─────────────────────────────────────────────

@dataclass
class TaggedMetadata:
    """
    Complete metadata dict attached to each Pinecone vector.
    Every field is filterable in Pinecone metadata queries.
    """
    # Source identification
    source_type:      str   = "unknown"
    source:           str   = ""
    authority_weight: float = 0.20

    # Document info
    company:          Optional[str] = None
    sector:           Optional[str] = None
    date:             str           = "unknown"
    date_year:        Optional[str] = None
    quarter:          Optional[str] = None
    financial_year:   Optional[str] = None
    doc_type:         str           = "unknown"
    document_id:      Optional[str] = None

    # Chunk info (populated by chunker)
    chunk_index:  int = 0
    doc_id:       str = ""
    section:      str = ""
    char_count:   int = 0
    word_count:   int = 0

    # Extracted financial figures
    financial_figures: dict = field(default_factory=dict)

    # Quality signals
    has_financial_figures: bool = False
    has_tables:            bool = False
    detected_terms:        list = field(default_factory=list)

    def to_pinecone_dict(self, text: str = "") -> dict:
        """
        Converts to flat dict for Pinecone metadata.
        Pinecone metadata must be: str, int, float, bool, or list[str].
        Nested dicts (financial_figures) are flattened with prefix.
        """
        result: dict = {
            "source_type":      self.source_type,
            "source":           self.source,
            "authority":        self.authority_weight,
            "company":          self.company or "",
            "sector":           self.sector or "",
            "date":             self.date,
            "date_year":        self.date_year or "",
            "quarter":          self.quarter or "",
            "financial_year":   self.financial_year or "",
            "doc_type":         self.doc_type,
            "document_id":      self.document_id or "",
            "chunk_index":      self.chunk_index,
            "doc_id":           self.doc_id,
            "section":          self.section,
            "char_count":       self.char_count,
            "word_count":       self.word_count,
            "has_figures":      self.has_financial_figures,
            "has_tables":       self.has_tables,
            "detected_terms":   self.detected_terms[:10],  # Pinecone list limit
        }
        if text:
            result["text"] = text[:1000]  # Pinecone metadata value size limit

        # Flatten financial figures: {"nim_pct": 4.2} → {"fig_nim_pct": 4.2}
        for k, v in self.financial_figures.items():
            result[f"fig_{k}"] = v

        # Remove None values — Pinecone rejects None
        return {k: v for k, v in result.items() if v is not None}


# ─────────────────────────────────────────────
# METADATA TAGGER CLASS
# ─────────────────────────────────────────────

class MetadataTagger:
    """
    Auto-tags document chunks with rich, structured metadata.

    Usage:
        from backend.ingestion.metadata_tagger import metadata_tagger

        # Tag a single chunk
        meta = metadata_tagger.tag_chunk(
            chunk_text="HDFC Bank NIM stood at 4.2% in Q3 FY26...",
            file_info={
                "filename": "HDFC_Bank_Annual_Report_2026.pdf",
                "source":   "HDFC Bank Annual Report 2026",
            }
        )

        # Extract financial figures only
        figures = metadata_tagger.extract_financial_figures(chunk_text)
        # → {"nim_pct": 4.2, "gnpa_pct": 1.26, "crar_pct": 18.4}
    """

    def __init__(self):
        self._total_tagged:   int = 0
        self._total_figures:  int = 0
        self._tag_time_ms:  float = 0.0

    # ─────────────────────────────────────────
    # MAIN: tag_chunk
    # ─────────────────────────────────────────

    def tag_chunk(
        self,
        chunk_text: str,
        file_info: Optional[dict] = None,
        base_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Produces a complete metadata dict for a single chunk.
        Merges: file_info + content analysis + financial figure extraction.

        Args:
            chunk_text:     The chunk text content
            file_info:      Dict with at minimum 'filename' or 'source'
                            Optional: source_type, company, date, sector
            base_metadata:  Pre-existing metadata to merge/override with

        Returns:
            Flat dict ready for Pinecone vector metadata

        Examples:
            meta = metadata_tagger.tag_chunk(
                chunk_text="HDFC NIM stood at 4.2% in Q3 FY26.",
                file_info={
                    "filename": "HDFC_Annual_Report_2026.pdf",
                    "source":   "HDFC Bank Annual Report 2026",
                }
            )
            meta["source_type"]    # → "annual_report"
            meta["company"]        # → "HDFC Bank"
            meta["fig_nim_pct"]    # → 4.2
            meta["quarter"]        # → "Q3"
            meta["financial_year"] # → "FY26"
        """
        tag_start = time.perf_counter()
        file_info      = file_info or {}
        base_metadata  = base_metadata or {}
        filename       = file_info.get("filename", "")
        combined_text  = (filename + " " + chunk_text).lower()

        meta = TaggedMetadata()

        # ── Source ────────────────────────────────────────────────────────────
        meta.source = (
            file_info.get("source")
            or base_metadata.get("source")
            or _slugify_filename(filename)
            or "Unknown Source"
        )

        # ── Source type (filename first, then content) ─────────────────────────
        meta.source_type = (
            base_metadata.get("source_type")
            or file_info.get("source_type")
            or self._detect_source_type(filename, chunk_text)
        )

        # ── Authority weight ───────────────────────────────────────────────────
        meta.authority_weight = AUTHORITY_WEIGHTS.get(
            meta.source_type, AUTHORITY_WEIGHTS["unknown"]
        )
        meta.doc_type = meta.source_type

        # ── Company ───────────────────────────────────────────────────────────
        company, sector = self._detect_company_and_sector(combined_text)
        meta.company = (
            base_metadata.get("company")
            or file_info.get("company")
            or company
        )

        # ── Sector ────────────────────────────────────────────────────────────
        meta.sector = (
            base_metadata.get("sector")
            or file_info.get("sector")
            or sector
            or self._detect_sector_from_content(chunk_text)
        )

        # ── Date ──────────────────────────────────────────────────────────────
        raw_date = (
            base_metadata.get("date")
            or file_info.get("date")
            or self.extract_date(chunk_text)
        )
        meta.date      = raw_date or "unknown"
        meta.date_year = _extract_year(meta.date)

        # ── Quarter & Financial Year ───────────────────────────────────────────
        quarter, financial_year = self._detect_quarter_and_fy(
            chunk_text + " " + filename
        )
        meta.quarter        = quarter
        meta.financial_year = financial_year

        # ── Document ID ───────────────────────────────────────────────────────
        meta.document_id = (
            base_metadata.get("document_id")
            or file_info.get("document_id")
            or file_info.get("doc_id")
        )

        # ── Chunk position (from base_metadata if chunker set it) ─────────────
        meta.chunk_index = base_metadata.get("chunk_index", 0)
        meta.doc_id      = base_metadata.get("doc_id", "")
        meta.section     = base_metadata.get("section", "")
        meta.char_count  = len(chunk_text)
        meta.word_count  = len(chunk_text.split())
        meta.has_tables  = "|" in chunk_text and chunk_text.count("|") > 3

        # ── Financial figure extraction ────────────────────────────────────────
        meta.financial_figures = self.extract_financial_figures(chunk_text)
        meta.has_financial_figures = len(meta.financial_figures) > 0

        # ── Detected financial terms ───────────────────────────────────────────
        meta.detected_terms = self._detect_financial_terms(chunk_text)

        # ── Build final pinecone dict ──────────────────────────────────────────
        result = meta.to_pinecone_dict(text=chunk_text)

        # Merge any remaining base_metadata fields not yet set
        for k, v in base_metadata.items():
            if k not in result and v is not None:
                result[k] = v

        duration_ms = (time.perf_counter() - tag_start) * 1000
        self._total_tagged  += 1
        self._total_figures += len(meta.financial_figures)
        self._tag_time_ms   += duration_ms

        logger.debug(
            f"[TAGGER] Tagged chunk | "
            f"source_type={meta.source_type} | "
            f"company={meta.company} | "
            f"sector={meta.sector} | "
            f"quarter={meta.quarter} | "
            f"fy={meta.financial_year} | "
            f"figures={len(meta.financial_figures)} | "
            f"{duration_ms:.1f}ms"
        )
        return result

    # ─────────────────────────────────────────
    # MAIN: extract_financial_figures
    # ─────────────────────────────────────────

    def extract_financial_figures(self, text: str) -> dict[str, float]:
        """
        Extracts structured financial metrics from chunk text via regex.
        Returns only metrics that are clearly present in the text.

        Args:
            text: Chunk text to scan

        Returns:
            Dict of {metric_key: float_value}
            e.g. {"nim_pct": 4.2, "gnpa_pct": 1.26, "crar_pct": 18.4}

        Examples:
            figures = metadata_tagger.extract_financial_figures(
                "NIM stood at 4.2%, GNPA improved to 1.26%, CRAR at 18.4%"
            )
            # → {"nim_pct": 4.2, "gnpa_pct": 1.26, "crar_pct": 18.4}
        """
        if not text:
            return {}

        figures: dict[str, float] = {}

        for metric_key, pattern in _COMPILED_FIGURE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue

            # Find the first non-None captured group
            value_str: Optional[str] = None
            for group in match.groups():
                if group is not None:
                    value_str = group.replace(",", "")
                    break

            if value_str is None:
                continue

            try:
                value = float(value_str)
                # Sanity bounds per metric type
                if _is_plausible_value(metric_key, value):
                    figures[metric_key] = round(value, 4)
            except (ValueError, OverflowError):
                continue

        if figures:
            logger.debug(
                f"[TAGGER] Extracted {len(figures)} figures: "
                f"{list(figures.keys())}"
            )

        return figures

    # ─────────────────────────────────────────
    # MAIN: extract_date
    # ─────────────────────────────────────────

    def extract_date(self, text: str) -> Optional[str]:
        """
        Extracts the most likely publication date from text.
        Tries patterns in priority order (most specific first).

        Args:
            text: Chunk or document text

        Returns:
            Date string (YYYY-MM-DD or YYYY-MM or YYYY) or None

        Examples:
            extract_date("Published January 15, 2026")  # → "January 15, 2026"
            extract_date("Q3 FY26 results")             # → "Q3 FY26"
            extract_date("fiscal year 2026")            # → "2026"
        """
        if not text:
            return None

        for pattern in _COMPILED_DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()

        return None

    # ─────────────────────────────────────────
    # PRIVATE: source type detection
    # ─────────────────────────────────────────

    def _detect_source_type(
        self,
        filename: str,
        content: str,
    ) -> str:
        """
        Detects source type by scanning filename first, then content.
        Returns first match in priority order.
        """
        # Filename-based (highest confidence)
        for pattern, source_type in FILENAME_SOURCE_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                return source_type

        # Content-based (first 600 chars for speed)
        snippet = content[:600]
        for pattern, source_type in CONTENT_SOURCE_PATTERNS:
            if re.search(pattern, snippet, re.IGNORECASE):
                return source_type

        return "unknown"

    # ─────────────────────────────────────────
    # PRIVATE: company + sector detection
    # ─────────────────────────────────────────

    def _detect_company_and_sector(
        self,
        text: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Detects company name and sector from combined filename+content text.
        Returns first matching company pattern.
        """
        for pattern, company_name, sector in _COMPILED_COMPANY:
            if pattern.search(text):
                return company_name, sector
        return None, None

    # ─────────────────────────────────────────
    # PRIVATE: sector from content keywords
    # ─────────────────────────────────────────

    def _detect_sector_from_content(self, text: str) -> Optional[str]:
        """
        Detects sector by counting keyword matches in chunk text.
        Returns sector with highest keyword match count (min 2 matches).
        """
        text_lower = text.lower()
        sector_scores: dict[str, int] = {}

        for sector, keywords in SECTOR_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score >= 2:
                sector_scores[sector] = score

        if not sector_scores:
            return None

        return max(sector_scores, key=lambda s: sector_scores[s])

    # ─────────────────────────────────────────
    # PRIVATE: quarter + FY detection
    # ─────────────────────────────────────────

    def _detect_quarter_and_fy(
        self,
        text: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Detects quarter (Q1-Q4) and financial year (FY26, FY2026)
        from text. Searches combined chunk + filename text.
        """
        quarter: Optional[str] = None
        financial_year: Optional[str] = None

        # Explicit Q+FY pattern: "Q3 FY26", "Q3FY2026"
        explicit = re.search(
            r'\b(q[1-4])\s*fy\s*(20)?([2-9]\d)\b',
            text, re.IGNORECASE,
        )
        if explicit:
            quarter = explicit.group(1).upper()
            fy_suffix = explicit.group(3)
            financial_year = f"FY{fy_suffix}"
            return quarter, financial_year

        # Quarter only
        for pattern, q_label in QUARTER_PATTERNS:
            if pattern.search(text):
                quarter = q_label
                break

        # Financial year
        fy_match = FY_PATTERN.search(text)
        if fy_match:
            fy_raw = fy_match.group(1) or fy_match.group(2)
            if fy_raw and len(fy_raw) == 2:
                financial_year = f"FY{fy_raw}"
            elif fy_raw and len(fy_raw) == 4:
                financial_year = f"FY{fy_raw[2:]}"

        return quarter, financial_year

    # ─────────────────────────────────────────
    # PRIVATE: financial term detection
    # ─────────────────────────────────────────

    def _detect_financial_terms(self, text: str) -> list[str]:
        """
        Detects which known financial terms appear in the chunk.
        Used for term_overlap scoring in retrieval.
        Returns up to 20 most relevant terms.
        """
        text_upper = text.upper()
        found: list[str] = []

        for term in FINANCIAL_TERMS:
            if re.search(rf'\b{re.escape(term)}\b', text_upper):
                found.append(term)
                if len(found) >= 20:
                    break

        return found

    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        avg_time = (
            round(self._tag_time_ms / self._total_tagged, 2)
            if self._total_tagged > 0 else 0.0
        )
        avg_figures = (
            round(self._total_figures / self._total_tagged, 1)
            if self._total_tagged > 0 else 0.0
        )
        return {
            "total_tagged":       self._total_tagged,
            "total_figures":      self._total_figures,
            "avg_tag_ms":         avg_time,
            "avg_figures_per_chunk": avg_figures,
            "total_tag_ms":       round(self._tag_time_ms, 1),
        }


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _is_plausible_value(metric_key: str, value: float) -> bool:
    """
    Guards against regex capturing garbage values.
    Each metric has plausible numeric bounds.
    """
    bounds: dict[str, tuple[float, float]] = {
        "nim_pct":             (0.5,   15.0),   # 0.5% – 15%
        "gnpa_pct":            (0.0,   50.0),   # 0% – 50%
        "nnpa_pct":            (0.0,   30.0),
        "casa_ratio_pct":      (5.0,   80.0),   # 5% – 80%
        "crar_pct":            (8.0,   50.0),   # RBI min is 11.5%
        "tier1_capital_pct":   (6.0,   40.0),
        "pcr_pct":             (10.0, 100.0),
        "roe_pct":             (-20.0, 60.0),
        "roa_pct":             (-5.0,  10.0),
        "pe_ratio":            (0.0,  500.0),
        "eps":                 (-500.0, 10000.0),
        "revenue_crore":       (1.0,  1000000.0),
        "revenue_growth_pct":  (-50.0, 200.0),
        "pat_crore":           (-100000.0, 1000000.0),
        "ebitda_margin_pct":   (-10.0, 80.0),
        "aum_crore":           (1.0,  10000000.0),
        "cagr_pct":            (-20.0, 100.0),
        "dividend_yield_pct":  (0.0,   20.0),
        "loan_growth_pct":     (-50.0, 200.0),
        "deposit_growth_pct":  (-50.0, 200.0),
        "repo_rate_pct":       (0.5,   20.0),
        "cpi_inflation_pct":   (-5.0,  50.0),
        "attrition_pct":       (0.0,  100.0),
        "deal_tcv_bn_usd":     (0.01, 1000.0),
    }
    lo, hi = bounds.get(metric_key, (-1e9, 1e9))
    return lo <= value <= hi


def _slugify_filename(filename: str) -> str:
    """Converts filename to readable source string."""
    if not filename:
        return ""
    name = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE)
    name = re.sub(r'[_\-]+', ' ', name)
    return name.strip().title()


def _extract_year(date_str: str) -> Optional[str]:
    """Extracts 4-digit year from a date string."""
    if not date_str or date_str == "unknown":
        return None
    match = re.search(r'\b(20[12]\d)\b', date_str)
    return match.group(1) if match else None


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

metadata_tagger = MetadataTagger()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "metadata_tagger",
    "MetadataTagger",
    "TaggedMetadata",
    "AUTHORITY_WEIGHTS",
    "SECTOR_KEYWORDS",
    "COMPANY_PATTERNS",
]
