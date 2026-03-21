# backend/utils/financial_terms.py
"""
FinThesisGuard AI — Financial Terms Dictionary
ArkAngel Financial Solutions

500+ Indian financial market acronyms and terminology.
Used by Agent 1 (Acronym Resolver) for query expansion.
Standalone module — no external dependencies.
"""

from typing import Optional


# ─────────────────────────────────────────────
# MASTER FINANCIAL TERMS DICTIONARY
# Format: "ACRONYM": ("Full Form", "Category", "Brief Context")
# ─────────────────────────────────────────────

_TERMS_FULL: dict[str, tuple[str, str, str]] = {

    # ══════════════════════════════════════════
    # BANKING & LENDING
    # ══════════════════════════════════════════

    "NIM":    ("Net Interest Margin", "banking", "Difference between interest income and interest paid, as % of assets"),
    "NII":    ("Net Interest Income", "banking", "Absolute difference between interest earned and interest paid"),
    "CASA":   ("Current Account Savings Account", "banking", "Low-cost deposit ratio — higher CASA = better margins"),
    "GNPA":   ("Gross Non-Performing Assets", "banking", "Total bad loans before provisioning"),
    "NNPA":   ("Net Non-Performing Assets", "banking", "Bad loans after deducting provisions"),
    "NPA":    ("Non-Performing Asset", "banking", "Loan overdue for 90+ days"),
    "PCR":    ("Provision Coverage Ratio", "banking", "% of NPAs covered by provisions"),
    "CRAR":   ("Capital to Risk-weighted Assets Ratio", "banking", "Bank's capital adequacy metric"),
    "CAR":    ("Capital Adequacy Ratio", "banking", "Minimum capital buffer required by RBI"),
    "CET1":   ("Common Equity Tier 1", "banking", "Highest quality regulatory capital"),
    "AT1":    ("Additional Tier 1", "banking", "Perpetual bonds counted as regulatory capital"),
    "TIER1":  ("Tier 1 Capital", "banking", "Core capital including CET1 and AT1"),
    "TIER2":  ("Tier 2 Capital", "banking", "Supplementary capital including subordinated debt"),
    "LCR":    ("Liquidity Coverage Ratio", "banking", "Short-term liquidity buffer requirement"),
    "NSFR":   ("Net Stable Funding Ratio", "banking", "Long-term funding stability requirement"),
    "SLR":    ("Statutory Liquidity Ratio", "banking", "% of deposits banks must hold in govt securities"),
    "CRR":    ("Cash Reserve Ratio", "banking", "% of deposits banks must keep with RBI"),
    "MSF":    ("Marginal Standing Facility", "banking", "RBI's emergency lending window for banks"),
    "LAF":    ("Liquidity Adjustment Facility", "banking", "RBI's repo/reverse repo framework"),
    "OMO":    ("Open Market Operations", "banking", "RBI buying/selling govt securities to manage liquidity"),
    "MCLR":   ("Marginal Cost of Funds-based Lending Rate", "banking", "Benchmark lending rate for banks"),
    "PLR":    ("Prime Lending Rate", "banking", "Older benchmark lending rate"),
    "BPLR":   ("Benchmark Prime Lending Rate", "banking", "Pre-MCLR benchmark rate"),
    "EBLR":   ("External Benchmark-based Lending Rate", "banking", "Repo-linked lending rate"),
    "CD":     ("Certificate of Deposit", "banking", "Short-term instrument issued by banks"),
    "CP":     ("Commercial Paper", "banking", "Short-term unsecured debt instrument"),
    "NBFC":   ("Non-Banking Financial Company", "banking", "Financial company not holding banking license"),
    "NBFC-D": ("Non-Banking Financial Company - Deposit Taking", "banking", "NBFC allowed to accept deposits"),
    "HFC":    ("Housing Finance Company", "banking", "NBFC focused on home loans"),
    "MFI":    ("Microfinance Institution", "banking", "Lender serving low-income borrowers"),
    "SFB":    ("Small Finance Bank", "banking", "Bank licensed to serve underserved segments"),
    "PB":     ("Payments Bank", "banking", "Limited banking license for payments and deposits"),
    "RRB":    ("Regional Rural Bank", "banking", "Bank serving rural areas"),
    "LAP":    ("Loan Against Property", "banking", "Mortgage-backed secured loan"),
    "HL":     ("Home Loan", "banking", "Mortgage for residential property purchase"),
    "AL":     ("Auto Loan", "banking", "Vehicle financing"),
    "PL":     ("Personal Loan", "banking", "Unsecured consumer credit"),
    "CC":     ("Cash Credit", "banking", "Working capital credit facility"),
    "OD":     ("Overdraft", "banking", "Credit facility against deposits"),
    "LC":     ("Letter of Credit", "banking", "Bank guarantee for trade transactions"),
    "BG":     ("Bank Guarantee", "banking", "Bank's commitment to pay on default"),
    "ECB":    ("External Commercial Borrowing", "banking", "Foreign currency borrowing by Indian entities"),
    "FCTL":   ("Foreign Currency Term Loan", "banking", "Term loan in foreign currency"),
    "WACR":   ("Weighted Average Call Rate", "banking", "Overnight interbank lending rate"),
    "WALR":   ("Weighted Average Lending Rate", "banking", "Average interest rate on outstanding loans"),
    "WADR":   ("Weighted Average Deposit Rate", "banking", "Average interest rate on deposits"),
    "SARFAESI": ("Securitisation and Reconstruction of Financial Assets and Enforcement of Security Interest Act", "banking", "Law enabling banks to recover NPAs"),
    "DRT":    ("Debt Recovery Tribunal", "banking", "Tribunal for recovering bank dues"),
    "IBC":    ("Insolvency and Bankruptcy Code", "banking", "Law for corporate insolvency resolution"),
    "NCLT":   ("National Company Law Tribunal", "banking", "Court handling corporate insolvency"),
    "NCLAT":  ("National Company Law Appellate Tribunal", "banking", "Appellate body for NCLT decisions"),
    "ARC":    ("Asset Reconstruction Company", "banking", "Entity buying NPAs from banks"),
    "SMA":    ("Special Mention Account", "banking", "Early warning category before NPA classification"),
    "ICA":    ("Inter-Creditor Agreement", "banking", "Agreement among lenders for stressed asset resolution"),
    "OTS":    ("One Time Settlement", "banking", "Negotiated settlement of dues"),
    "CIBIL":  ("Credit Information Bureau India Limited", "banking", "India's major credit bureau"),
    "CRIF":   ("CRIF High Mark", "banking", "Credit bureau for NBFC and microfinance"),
    "FLDG":   ("First Loss Default Guarantee", "banking", "Co-lending risk sharing mechanism"),
    "BC":     ("Business Correspondent", "banking", "Agent banking model for last-mile banking"),
    "KYC":    ("Know Your Customer", "banking", "Identity verification process"),
    "AML":    ("Anti-Money Laundering", "banking", "Regulations to prevent money laundering"),
    "CFT":    ("Countering Financing of Terrorism", "banking", "Regulations against terrorist financing"),
    "FEMA":   ("Foreign Exchange Management Act", "banking", "Law governing forex transactions in India"),
    "PMLA":   ("Prevention of Money Laundering Act", "banking", "Indian AML legislation"),
    "SWIFT":  ("Society for Worldwide Interbank Financial Telecommunication", "banking", "Global interbank messaging network"),
    "RTGS":   ("Real Time Gross Settlement", "banking", "Instant large-value fund transfer"),
    "NEFT":   ("National Electronic Funds Transfer", "banking", "Batch-based electronic fund transfer"),
    "IMPS":   ("Immediate Payment Service", "banking", "24x7 instant interbank transfer"),
    "UPI":    ("Unified Payments Interface", "banking", "Real-time mobile payment system"),
    "NACH":   ("National Automated Clearing House", "banking", "Electronic mandate for recurring payments"),
    "ECS":    ("Electronic Clearing Service", "banking", "Older bulk payment system"),
    "NPCI":   ("National Payments Corporation of India", "banking", "Umbrella organization for retail payments"),
    "BBPS":   ("Bharat Bill Payment System", "banking", "Integrated bill payment platform"),

    # ══════════════════════════════════════════
    # CAPITAL MARKETS & EQUITY
    # ══════════════════════════════════════════

    "NSE":    ("National Stock Exchange", "capital_market", "India's largest stock exchange by volume"),
    "BSE":    ("Bombay Stock Exchange", "capital_market", "India's oldest stock exchange"),
    "MCX":    ("Multi Commodity Exchange", "capital_market", "India's largest commodity exchange"),
    "NCDEX":  ("National Commodity & Derivatives Exchange", "capital_market", "Agricultural commodity exchange"),
    "FII":    ("Foreign Institutional Investor", "capital_market", "Foreign entity investing in Indian markets (old term)"),
    "FPI":    ("Foreign Portfolio Investor", "capital_market", "Current SEBI category for foreign investors"),
    "DII":    ("Domestic Institutional Investor", "capital_market", "Indian institutions like MFs, insurance companies"),
    "QIB":    ("Qualified Institutional Buyer", "capital_market", "Institutional investor in IPO context"),
    "HNI":    ("High Net-worth Individual", "capital_market", "Investor with >₹2L IPO application"),
    "RII":    ("Retail Individual Investor", "capital_market", "Small investor with <₹2L IPO application"),
    "IPO":    ("Initial Public Offering", "capital_market", "First public sale of company shares"),
    "FPO":    ("Follow-on Public Offer", "capital_market", "Additional shares sold by listed company"),
    "OFS":    ("Offer for Sale", "capital_market", "Existing shareholders selling shares via stock exchange"),
    "QIP":    ("Qualified Institutional Placement", "capital_market", "Fast-track fundraise from institutional investors"),
    "DRHP":   ("Draft Red Herring Prospectus", "capital_market", "Preliminary IPO document filed with SEBI"),
    "RHP":    ("Red Herring Prospectus", "capital_market", "Final IPO document before listing"),
    "FOPO":   ("Follow-on Public Offer", "capital_market", "Secondary equity issuance by listed company"),
    "GDR":    ("Global Depository Receipt", "capital_market", "Share of Indian company traded on foreign exchange"),
    "ADR":    ("American Depository Receipt", "capital_market", "Indian company shares listed on US exchanges"),
    "ESOP":   ("Employee Stock Option Plan", "capital_market", "Stock options granted to employees"),
    "ESOS":   ("Employee Stock Option Scheme", "capital_market", "Scheme under which ESOPs are granted"),
    "ESPS":   ("Employee Stock Purchase Scheme", "capital_market", "Direct stock purchase plan for employees"),
    "NCD":    ("Non-Convertible Debenture", "capital_market", "Fixed income bond not convertible to equity"),
    "FCCB":   ("Foreign Currency Convertible Bond", "capital_market", "Bond convertible to equity, issued in foreign currency"),
    "CB":     ("Convertible Bond", "capital_market", "Bond that can convert to equity"),
    "CCD":    ("Compulsorily Convertible Debenture", "capital_market", "Debenture mandatorily converting to equity"),
    "CCPS":   ("Compulsorily Convertible Preference Share", "capital_market", "Preferred share that converts to equity"),
    "F&O":    ("Futures & Options", "capital_market", "Derivative instruments on exchanges"),
    "OI":     ("Open Interest", "capital_market", "Total outstanding derivative contracts"),
    "PCR":    ("Put-Call Ratio", "capital_market", "Ratio of put to call options, sentiment indicator"),
    "IV":     ("Implied Volatility", "capital_market", "Market's expectation of future volatility"),
    "VIX":    ("Volatility Index", "capital_market", "India VIX measures market fear/uncertainty"),
    "ATM":    ("At The Money", "capital_market", "Option with strike = current market price"),
    "ITM":    ("In The Money", "capital_market", "Option with intrinsic value"),
    "OTM":    ("Out of The Money", "capital_market", "Option with no intrinsic value"),
    "CE":     ("Call Option European", "capital_market", "Right to buy at strike price"),
    "PE":     ("Put Option European", "capital_market", "Right to sell at strike price — also Price-to-Earnings"),
    "LOT":    ("Lot Size", "capital_market", "Minimum quantity in derivative contract"),
    "MTM":    ("Mark to Market", "capital_market", "Daily settlement of derivative positions"),
    "SPAN":   ("Standard Portfolio Analysis of Risk", "capital_market", "Margin calculation methodology"),
    "BTST":   ("Buy Today Sell Tomorrow", "capital_market", "Short-term trading strategy"),
    "STBT":   ("Sell Today Buy Tomorrow", "capital_market", "Short selling strategy"),
    "CNC":    ("Cash and Carry", "capital_market", "Delivery-based equity trade"),
    "MIS":    ("Margin Intraday Square-off", "capital_market", "Intraday leveraged trade"),
    "NIFTY":  ("National Fifty", "capital_market", "NSE's benchmark 50-stock index"),
    "SENSEX": ("Sensitive Index", "capital_market", "BSE's benchmark 30-stock index"),
    "BANKNIFTY": ("Bank Nifty", "capital_market", "NSE's banking sector index"),
    "FINNIFTY": ("Financial Services Index", "capital_market", "NSE's financial services index"),
    "MIDCAP": ("Mid Capitalization", "capital_market", "Companies with medium market cap"),
    "SMALLCAP": ("Small Capitalization", "capital_market", "Companies with smaller market cap"),
    "LARGECAP": ("Large Capitalization", "capital_market", "Top 100 companies by market cap"),
    "MICROCAP": ("Micro Capitalization", "capital_market", "Very small market cap companies"),
    "MCAP":   ("Market Capitalization", "capital_market", "Total market value of outstanding shares"),
    "FLOAT":  ("Free Float", "capital_market", "Shares available for public trading"),
    "SEBI":   ("Securities and Exchange Board of India", "regulatory", "Capital market regulator"),
    "CDSL":   ("Central Depository Services Limited", "capital_market", "Depository for holding securities"),
    "NSDL":   ("National Securities Depository Limited", "capital_market", "India's first and largest depository"),
    "DP":     ("Depository Participant", "capital_market", "Intermediary between investor and depository"),
    "DEMAT":  ("Dematerialized Account", "capital_market", "Electronic account for holding securities"),
    "ISIN":   ("International Securities Identification Number", "capital_market", "Unique identifier for securities"),
    "CUSIP":  ("Committee on Uniform Securities Identification Procedures", "capital_market", "US securities identifier"),
    "T+1":    ("Trade Plus One Day", "capital_market", "Settlement cycle for equity trades in India"),
    "T+2":    ("Trade Plus Two Days", "capital_market", "Older equity settlement cycle"),
    "ALGO":   ("Algorithmic Trading", "capital_market", "Automated trading using programmed strategies"),
    "HFT":    ("High Frequency Trading", "capital_market", "Ultra-fast algorithmic trading"),
    "DMA":    ("Direct Market Access", "capital_market", "Direct order routing to exchange"),
    "SOR":    ("Smart Order Routing", "capital_market", "Automated best-execution order routing"),
    "BLOCK":  ("Block Deal", "capital_market", "Large transaction in separate window"),
    "BULK":   ("Bulk Deal", "capital_market", "Trade >0.5% of equity in single session"),
    "INSIDER": ("Insider Trading", "capital_market", "Trading on non-public material information"),
    "UPSI":   ("Unpublished Price Sensitive Information", "capital_market", "Non-public info that affects share price"),
    "GMP":    ("Grey Market Premium", "capital_market", "Unofficial premium for IPO shares before listing"),
    "ALLOTMENT": ("Share Allotment", "capital_market", "Assignment of shares in IPO process"),
    "ANCHOR": ("Anchor Investor", "capital_market", "Institutional investor allocated IPO shares before open"),
    "OVERSUBSCRIPTION": ("Oversubscription", "capital_market", "Demand exceeding IPO shares available"),

    # ══════════════════════════════════════════
    # FINANCIAL RATIOS & VALUATION
    # ══════════════════════════════════════════

    "PE":     ("Price-to-Earnings Ratio", "ratio", "Market price divided by earnings per share"),
    "PB":     ("Price-to-Book Ratio", "ratio", "Market price divided by book value per share"),
    "PS":     ("Price-to-Sales Ratio", "ratio", "Market cap divided by annual revenue"),
    "EV":     ("Enterprise Value", "ratio", "Total company value including debt minus cash"),
    "EBITDA": ("Earnings Before Interest Tax Depreciation Amortization", "ratio", "Proxy for operating cash flow"),
    "EBIT":   ("Earnings Before Interest and Tax", "ratio", "Operating profit"),
    "EBT":    ("Earnings Before Tax", "ratio", "Profit before tax deduction"),
    "PAT":    ("Profit After Tax", "ratio", "Net profit — bottom line"),
    "PBT":    ("Profit Before Tax", "ratio", "Profit before tax deduction"),
    "EPS":    ("Earnings Per Share", "ratio", "PAT divided by total shares outstanding"),
    "DEPS":   ("Diluted Earnings Per Share", "ratio", "EPS assuming all dilutive instruments converted"),
    "DPS":    ("Dividend Per Share", "ratio", "Dividend paid per equity share"),
    "BV":     ("Book Value", "ratio", "Net assets per share"),
    "BVPS":   ("Book Value Per Share", "ratio", "Shareholders equity divided by shares outstanding"),
    "ROE":    ("Return on Equity", "ratio", "PAT divided by shareholders equity"),
    "ROA":    ("Return on Assets", "ratio", "PAT divided by total assets"),
    "ROCE":   ("Return on Capital Employed", "ratio", "EBIT divided by capital employed"),
    "ROIC":   ("Return on Invested Capital", "ratio", "NOPAT divided by invested capital"),
    "RONW":   ("Return on Net Worth", "ratio", "Same as ROE — PAT divided by networth"),
    "CAGR":   ("Compound Annual Growth Rate", "ratio", "Smoothed annual growth rate over period"),
    "FCF":    ("Free Cash Flow", "ratio", "Operating cash flow minus capex"),
    "OCF":    ("Operating Cash Flow", "ratio", "Cash generated from core business operations"),
    "CAPEX":  ("Capital Expenditure", "ratio", "Spending on fixed assets and infrastructure"),
    "OPEX":   ("Operating Expenditure", "ratio", "Day-to-day business operating costs"),
    "D/E":    ("Debt to Equity Ratio", "ratio", "Total debt divided by shareholders equity"),
    "ICR":    ("Interest Coverage Ratio", "ratio", "EBIT divided by interest expense"),
    "DSCR":   ("Debt Service Coverage Ratio", "ratio", "Cash flow available to service debt"),
    "EV/EBITDA": ("Enterprise Value to EBITDA", "ratio", "Key valuation multiple for businesses"),
    "PEG":    ("Price-to-Earnings-to-Growth", "ratio", "PE ratio divided by earnings growth rate"),
    "PFCF":   ("Price to Free Cash Flow", "ratio", "Market cap divided by FCF"),
    "TOBIN":  ("Tobin's Q", "ratio", "Market value divided by replacement cost of assets"),
    "NAV":    ("Net Asset Value", "ratio", "Total assets minus liabilities — used for MFs and real estate"),
    "INTRINSIC": ("Intrinsic Value", "ratio", "Estimated true value based on fundamentals"),
    "DCF":    ("Discounted Cash Flow", "ratio", "Valuation using present value of future cash flows"),
    "WACC":   ("Weighted Average Cost of Capital", "ratio", "Blended cost of equity and debt"),
    "COE":    ("Cost of Equity", "ratio", "Return required by equity investors"),
    "COD":    ("Cost of Debt", "ratio", "Interest rate paid on borrowings"),
    "BETA":   ("Beta Coefficient", "ratio", "Stock's sensitivity to market movements"),
    "ALPHA":  ("Alpha", "ratio", "Excess return above benchmark"),
    "SHARPE": ("Sharpe Ratio", "ratio", "Return per unit of total risk"),
    "SORTINO": ("Sortino Ratio", "ratio", "Return per unit of downside risk"),
    "TREYNOR": ("Treynor Ratio", "ratio", "Return per unit of systematic risk"),
    "IR":     ("Information Ratio", "ratio", "Active return divided by tracking error"),
    "TE":     ("Tracking Error", "ratio", "Standard deviation of portfolio vs benchmark returns"),
    "DD":     ("Maximum Drawdown", "ratio", "Largest peak-to-trough decline in portfolio"),
    "VaR":    ("Value at Risk", "ratio", "Maximum expected loss at given confidence level"),
    "CVaR":   ("Conditional Value at Risk", "ratio", "Expected loss beyond VaR threshold"),
    "STDDEV": ("Standard Deviation", "ratio", "Statistical measure of volatility"),
    "VARIANCE": ("Variance", "ratio", "Square of standard deviation"),
    "CORR":   ("Correlation", "ratio", "Statistical relationship between two assets"),
    "COVAR":  ("Covariance", "ratio", "Joint variability of two assets"),
    "MARGIN": ("Profit Margin", "ratio", "Profit as percentage of revenue"),
    "GM":     ("Gross Margin", "ratio", "Revenue minus COGS divided by revenue"),
    "OM":     ("Operating Margin", "ratio", "EBIT divided by revenue"),
    "NM":     ("Net Margin", "ratio", "PAT divided by revenue"),
    "EBITDA_MARGIN": ("EBITDA Margin", "ratio", "EBITDA divided by revenue"),
    "ASSET_TURNOVER": ("Asset Turnover Ratio", "ratio", "Revenue divided by total assets"),
    "INV_TURNOVER": ("Inventory Turnover", "ratio", "COGS divided by average inventory"),
    "REC_TURNOVER": ("Receivables Turnover", "ratio", "Revenue divided by average receivables"),
    "DIO":    ("Days Inventory Outstanding", "ratio", "Average days to sell inventory"),
    "DSO":    ("Days Sales Outstanding", "ratio", "Average days to collect payment"),
    "DPO":    ("Days Payable Outstanding", "ratio", "Average days to pay suppliers"),
    "CCC":    ("Cash Conversion Cycle", "ratio", "DIO + DSO - DPO"),
    "WC":     ("Working Capital", "ratio", "Current assets minus current liabilities"),
    "NWC":    ("Net Working Capital", "ratio", "Same as working capital"),
    "CR":     ("Current Ratio", "ratio", "Current assets divided by current liabilities"),
    "QR":     ("Quick Ratio", "ratio", "Liquid assets divided by current liabilities"),
    "CASH_RATIO": ("Cash Ratio", "ratio", "Cash and equivalents divided by current liabilities"),
    "LEVERAGE": ("Financial Leverage", "ratio", "Total assets divided by equity"),
    "GEARING": ("Gearing Ratio", "ratio", "Debt as proportion of equity"),
    "DIVIDEND_YIELD": ("Dividend Yield", "ratio", "Annual dividend divided by share price"),
    "PAYOUT": ("Dividend Payout Ratio", "ratio", "DPS divided by EPS"),
    "RETENTION": ("Retention Ratio", "ratio", "1 minus dividend payout ratio"),

    # ══════════════════════════════════════════
    # MUTUAL FUNDS & ASSET MANAGEMENT
    # ══════════════════════════════════════════

    "AUM":    ("Assets Under Management", "mutual_fund", "Total value of assets managed by fund"),
    "SIP":    ("Systematic Investment Plan", "mutual_fund", "Regular fixed-amount investment in mutual fund"),
    "SWP":    ("Systematic Withdrawal Plan", "mutual_fund", "Regular withdrawal from mutual fund"),
    "STP":    ("Systematic Transfer Plan", "mutual_fund", "Regular transfer between mutual fund schemes"),
    "NFO":    ("New Fund Offer", "mutual_fund", "Launch of new mutual fund scheme"),
    "XIRR":   ("Extended Internal Rate of Return", "mutual_fund", "Returns accounting for irregular cash flows"),
    "IRR":    ("Internal Rate of Return", "mutual_fund", "Discount rate that makes NPV zero"),
    "NPV":    ("Net Present Value", "mutual_fund", "Present value of future cash flows minus initial investment"),
    "ELSS":   ("Equity Linked Savings Scheme", "mutual_fund", "Tax-saving mutual fund with 3-year lock-in"),
    "FOF":    ("Fund of Funds", "mutual_fund", "Mutual fund investing in other mutual funds"),
    "ETF":    ("Exchange Traded Fund", "mutual_fund", "Fund traded on stock exchange like a share"),
    "INDEX":  ("Index Fund", "mutual_fund", "Passive fund tracking a market index"),
    "DEBT":   ("Debt Fund", "mutual_fund", "Mutual fund investing in fixed income securities"),
    "LIQUID": ("Liquid Fund", "mutual_fund", "Overnight to 91-day maturity debt fund"),
    "FMP":    ("Fixed Maturity Plan", "mutual_fund", "Close-ended debt fund with fixed tenure"),
    "GILT":   ("Gilt Fund", "mutual_fund", "Fund investing only in government securities"),
    "HYBRID": ("Hybrid Fund", "mutual_fund", "Fund investing in both equity and debt"),
    "BAF":    ("Balanced Advantage Fund", "mutual_fund", "Dynamically managed hybrid fund"),
    "MOM":    ("Momentum Fund", "mutual_fund", "Fund based on price momentum strategy"),
    "FACTOR": ("Factor Fund", "mutual_fund", "Fund based on specific investment factors"),
    "QUANT":  ("Quantitative Fund", "mutual_fund", "Rules-based algorithmic investment fund"),
    "AMC":    ("Asset Management Company", "mutual_fund", "Company managing mutual fund schemes"),
    "AMFI":   ("Association of Mutual Funds in India", "mutual_fund", "Self-regulatory body for mutual funds"),
    "MFD":    ("Mutual Fund Distributor", "mutual_fund", "Registered intermediary selling mutual funds"),
    "RTA":    ("Registrar and Transfer Agent", "mutual_fund", "Back-office for mutual fund transactions"),
    "CAMS":   ("Computer Age Management Services", "mutual_fund", "India's largest mutual fund RTA"),
    "KFIN":   ("KFin Technologies", "mutual_fund", "Second largest mutual fund RTA"),
    "DIRECT": ("Direct Plan", "mutual_fund", "Mutual fund plan without distributor commission"),
    "REGULAR": ("Regular Plan", "mutual_fund", "Mutual fund plan sold via distributor"),
    "GROWTH": ("Growth Plan", "mutual_fund", "Mutual fund option where gains are reinvested"),
    "IDCW":   ("Income Distribution cum Capital Withdrawal", "mutual_fund", "Previously called dividend plan"),
    "TER":    ("Total Expense Ratio", "mutual_fund", "Annual fee charged by mutual fund scheme"),
    "EXIT_LOAD": ("Exit Load", "mutual_fund", "Fee charged on early redemption"),
    "LOCK_IN": ("Lock-in Period", "mutual_fund", "Mandatory holding period before redemption"),
    "T+3":    ("Trade Plus Three Days", "mutual_fund", "Mutual fund redemption settlement cycle"),
    "KIM":    ("Key Information Memorandum", "mutual_fund", "Summary document for mutual fund scheme"),
    "SID":    ("Scheme Information Document", "mutual_fund", "Detailed prospectus for mutual fund"),
    "SAI":    ("Statement of Additional Information", "mutual_fund", "Legal document for mutual fund trust"),
    "CAS":    ("Consolidated Account Statement", "mutual_fund", "Single statement of all mutual fund holdings"),
    "PMS":    ("Portfolio Management Service", "mutual_fund", "Discretionary investment management for HNIs"),
    "AIF":    ("Alternative Investment Fund", "mutual_fund", "Private pooled fund for sophisticated investors"),
    "VCF":    ("Venture Capital Fund", "mutual_fund", "AIF investing in early-stage companies"),
    "PE_FUND": ("Private Equity Fund", "mutual_fund", "AIF investing in private companies"),
    "HEDGE":  ("Hedge Fund", "mutual_fund", "Aggressively managed fund using complex strategies"),
    "REIT":   ("Real Estate Investment Trust", "mutual_fund", "Listed fund investing in income-generating real estate"),
    "INVIT":  ("Infrastructure Investment Trust", "mutual_fund", "Listed fund investing in infrastructure assets"),
    "SPAC":   ("Special Purpose Acquisition Company", "mutual_fund", "Blank-check company for acquisitions"),
    "NPS":    ("National Pension System", "mutual_fund", "Government-run retirement savings scheme"),
    "APY":    ("Atal Pension Yojana", "mutual_fund", "Pension scheme for unorganized sector workers"),
    "PPF":    ("Public Provident Fund", "mutual_fund", "15-year government savings scheme with tax benefits"),
    "EPF":    ("Employees Provident Fund", "mutual_fund", "Mandatory retirement savings for employees"),
    "EPFO":   ("Employees Provident Fund Organisation", "mutual_fund", "Body managing EPF"),
    "VPF":    ("Voluntary Provident Fund", "mutual_fund", "Additional voluntary contribution to EPF"),
    "NSC":    ("National Savings Certificate", "mutual_fund", "5-year post office savings instrument"),
    "KVP":    ("Kisan Vikas Patra", "mutual_fund", "Post office savings instrument for farmers"),
    "SSY":    ("Sukanya Samriddhi Yojana", "mutual_fund", "Girl child savings scheme"),

    # ══════════════════════════════════════════
    # TAX & ACCOUNTING
    # ══════════════════════════════════════════

    "LTCG":   ("Long-Term Capital Gains", "tax", "Gains on assets held >1 year — 10% equity, 20% others"),
    "STCG":   ("Short-Term Capital Gains", "tax", "Gains on assets held <1 year — 15% equity, slab others"),
    "STT":    ("Securities Transaction Tax", "tax", "Tax on equity trades — 0.1% delivery, 0.025% intraday"),
    "DDT":    ("Dividend Distribution Tax", "tax", "Tax paid by companies on dividends — abolished FY21"),
    "TDS":    ("Tax Deducted at Source", "tax", "Tax deducted by payer before payment"),
    "TCS":    ("Tax Collected at Source", "tax", "Tax collected by seller from buyer"),
    "GST":    ("Goods and Services Tax", "tax", "Unified indirect tax replacing multiple levies"),
    "CGST":   ("Central Goods and Services Tax", "tax", "GST component collected by central government"),
    "SGST":   ("State Goods and Services Tax", "tax", "GST component collected by state government"),
    "IGST":   ("Integrated Goods and Services Tax", "tax", "GST on interstate supply"),
    "ITC":    ("Input Tax Credit", "tax", "Credit for GST paid on business inputs"),
    "MAT":    ("Minimum Alternate Tax", "tax", "Minimum tax for companies with book profits"),
    "AMT":    ("Alternate Minimum Tax", "tax", "Minimum tax for individuals/non-corporate"),
    "DTL":    ("Deferred Tax Liability", "tax", "Tax payable in future due to timing differences"),
    "DTA":    ("Deferred Tax Asset", "tax", "Tax benefit receivable in future"),
    "HUF":    ("Hindu Undivided Family", "tax", "Tax entity for joint Hindu family business"),
    "AOP":    ("Association of Persons", "tax", "Tax entity for group of persons"),
    "BOI":    ("Body of Individuals", "tax", "Tax entity for body of individuals"),
    "ITR":    ("Income Tax Return", "tax", "Annual tax filing document"),
    "PAN":    ("Permanent Account Number", "tax", "Unique tax identification number"),
    "GSTIN":  ("Goods and Services Tax Identification Number", "tax", "Unique GST registration number"),
    "TAN":    ("Tax Deduction Account Number", "tax", "Number for TDS filing entities"),
    "AY":     ("Assessment Year", "tax", "Year in which income is assessed for tax"),
    "PY":     ("Previous Year", "tax", "Year in which income is earned"),
    "FY":     ("Financial Year", "tax", "April to March fiscal year in India"),
    "80C":    ("Section 80C Deduction", "tax", "₹1.5L deduction for ELSS, PPF, LIC, ELSS etc."),
    "80D":    ("Section 80D Deduction", "tax", "Deduction for health insurance premium"),
    "80CCD":  ("Section 80CCD Deduction", "tax", "Deduction for NPS contributions"),
    "HRA":    ("House Rent Allowance", "tax", "Tax-exempt salary component for rent"),
    "LTA":    ("Leave Travel Allowance", "tax", "Tax-exempt travel reimbursement"),
    "NRI":    ("Non-Resident Indian", "tax", "Indian citizen living abroad — different tax treatment"),
    "RNOR":   ("Resident but Not Ordinarily Resident", "tax", "Transitional tax status for returning NRIs"),
    "DTAA":   ("Double Taxation Avoidance Agreement", "tax", "Treaty preventing double taxation across countries"),
    "NRE":    ("Non-Resident External Account", "tax", "Rupee account for NRI — interest tax-free in India"),
    "NRO":    ("Non-Resident Ordinary Account", "tax", "Rupee account for NRI — interest taxable in India"),
    "FCNR":   ("Foreign Currency Non-Resident Account", "tax", "NRI account in foreign currency"),
    "POEM":   ("Place of Effective Management", "tax", "Test for determining company residency for tax"),
    "GAAR":   ("General Anti-Avoidance Rules", "tax", "Rules to prevent abusive tax avoidance"),
    "BEPS":   ("Base Erosion and Profit Shifting", "tax", "OECD framework against multinational tax avoidance"),
    "ICDS":   ("Income Computation and Disclosure Standards", "tax", "Standards for computing business income"),
    "IND_AS": ("Indian Accounting Standards", "tax", "IFRS-converged accounting standards for India"),
    "GAAP":   ("Generally Accepted Accounting Principles", "tax", "Framework of accounting standards"),
    "IFRS":   ("International Financial Reporting Standards", "tax", "Global accounting standards"),
    "AS":     ("Accounting Standards", "tax", "Older ICAI accounting standards"),
    "ICAI":   ("Institute of Chartered Accountants of India", "tax", "Regulatory body for CAs in India"),
    "CA":     ("Chartered Accountant", "tax", "Licensed accounting professional"),
    "CFO":    ("Chief Financial Officer", "tax", "Head of finance in a company"),
    "MD&A":   ("Management Discussion and Analysis", "tax", "Management commentary in annual report"),
    "P&L":    ("Profit and Loss Statement", "tax", "Income statement showing revenues and expenses"),
    "BS":     ("Balance Sheet", "tax", "Financial statement of assets, liabilities, equity"),
    "CF":     ("Cash Flow Statement", "tax", "Statement of cash inflows and outflows"),

    # ══════════════════════════════════════════
    # REGULATORY & GOVERNMENT
    # ══════════════════════════════════════════

    "RBI":    ("Reserve Bank of India", "regulatory", "India's central bank and banking regulator"),
    "SEBI":   ("Securities and Exchange Board of India", "regulatory", "Capital market and investment regulator"),
    "IRDAI":  ("Insurance Regulatory and Development Authority of India", "regulatory", "Insurance sector regulator"),
    "PFRDA":  ("Pension Fund Regulatory and Development Authority", "regulatory", "Pension sector regulator"),
    "MCA":    ("Ministry of Corporate Affairs", "regulatory", "Governs company law and corporate governance"),
    "DPIIT":  ("Department for Promotion of Industry and Internal Trade", "regulatory", "FDI and startup policy body"),
    "FIPB":   ("Foreign Investment Promotion Board", "regulatory", "Now defunct — FDI approval body"),
    "CBDT":   ("Central Board of Direct Taxes", "regulatory", "Administers income tax in India"),
    "CBIC":   ("Central Board of Indirect Taxes and Customs", "regulatory", "Administers GST and customs"),
    "ED":     ("Enforcement Directorate", "regulatory", "Investigates economic offences and FEMA violations"),
    "CBI":    ("Central Bureau of Investigation", "regulatory", "India's premier investigative agency"),
    "SFIO":   ("Serious Fraud Investigation Office", "regulatory", "Investigates corporate fraud"),
    "IBBI":   ("Insolvency and Bankruptcy Board of India", "regulatory", "Regulates insolvency process"),
    "CCI":    ("Competition Commission of India", "regulatory", "Antitrust regulator"),
    "TRAI":   ("Telecom Regulatory Authority of India", "regulatory", "Telecom sector regulator"),
    "CERC":   ("Central Electricity Regulatory Commission", "regulatory", "Electricity market regulator"),
    "MERC":   ("Maharashtra Electricity Regulatory Commission", "regulatory", "State electricity regulator"),
    "IRDA":   ("Insurance Regulatory and Development Authority", "regulatory", "Old name for IRDAI"),
    "IEPFA":  ("Investor Education and Protection Fund Authority", "regulatory", "Manages unclaimed dividends"),
    "SAT":    ("Securities Appellate Tribunal", "regulatory", "Appeals court for SEBI orders"),
    "PMJDY":  ("Pradhan Mantri Jan Dhan Yojana", "regulatory", "Financial inclusion scheme for bank accounts"),
    "MUDRA":  ("Micro Units Development and Refinance Agency", "regulatory", "Loans for small businesses"),
    "SIDBI":  ("Small Industries Development Bank of India", "regulatory", "Development bank for MSMEs"),
    "NABARD": ("National Bank for Agriculture and Rural Development", "regulatory", "Agriculture development bank"),
    "NHB":    ("National Housing Bank", "regulatory", "Regulator and refinancer for HFCs"),
    "EXIM":   ("Export-Import Bank of India", "regulatory", "Bank financing India's international trade"),
    "ECGC":   ("Export Credit Guarantee Corporation", "regulatory", "Export credit insurance"),
    "IRFC":   ("Indian Railway Finance Corporation", "regulatory", "Financing arm of Indian Railways"),
    "NITI":   ("NITI Aayog", "regulatory", "Government policy think tank"),
    "MEA":    ("Ministry of External Affairs", "regulatory", "India's foreign ministry"),
    "MOF":    ("Ministry of Finance", "regulatory", "India's finance ministry"),
    "DEA":    ("Department of Economic Affairs", "regulatory", "Arm of Ministry of Finance"),
    "FSDC":   ("Financial Stability and Development Council", "regulatory", "Inter-regulatory coordination body"),
    "IEX":    ("Indian Energy Exchange", "regulatory", "Power trading exchange"),
    "NCDEX":  ("National Commodity Derivatives Exchange", "regulatory", "Agricultural commodity exchange"),

    # ══════════════════════════════════════════
    # CORPORATE & GOVERNANCE
    # ══════════════════════════════════════════

    "AGM":    ("Annual General Meeting", "corporate", "Yearly shareholder meeting"),
    "EGM":    ("Extraordinary General Meeting", "corporate", "Special shareholder meeting for urgent matters"),
    "MGT":    ("Management", "corporate", "Company's leadership team"),
    "BOD":    ("Board of Directors", "corporate", "Governing body of a company"),
    "MD":     ("Managing Director", "corporate", "Head of company operations"),
    "CMD":    ("Chairman and Managing Director", "corporate", "Combined chairman and MD role"),
    "CEO":    ("Chief Executive Officer", "corporate", "Top executive of company"),
    "CTO":    ("Chief Technology Officer", "corporate", "Head of technology"),
    "COO":    ("Chief Operating Officer", "corporate", "Head of operations"),
    "CRO":    ("Chief Risk Officer", "corporate", "Head of risk management"),
    "CSO":    ("Chief Strategy Officer", "corporate", "Head of strategy"),
    "CHRO":   ("Chief Human Resources Officer", "corporate", "Head of HR"),
    "CLO":    ("Chief Legal Officer", "corporate", "Head of legal affairs"),
    "NED":    ("Non-Executive Director", "corporate", "Director not involved in day-to-day management"),
    "ID":     ("Independent Director", "corporate", "Director without material relationship with company"),
    "AC":     ("Audit Committee", "corporate", "Board committee overseeing financial reporting"),
    "NRC":    ("Nomination and Remuneration Committee", "corporate", "Board committee on director pay"),
    "SRC":    ("Stakeholder Relationship Committee", "corporate", "Board committee for investor grievances"),
    "CSR":    ("Corporate Social Responsibility", "corporate", "Mandatory social spending for qualifying companies"),
    "ESG":    ("Environmental Social Governance", "corporate", "Sustainability investment framework"),
    "GRI":    ("Global Reporting Initiative", "corporate", "Sustainability reporting framework"),
    "BRSR":   ("Business Responsibility and Sustainability Report", "corporate", "SEBI-mandated ESG disclosure"),
    "MoA":    ("Memorandum of Association", "corporate", "Document defining company's objectives"),
    "AoA":    ("Articles of Association", "corporate", "Document governing internal company rules"),
    "ROC":    ("Registrar of Companies", "corporate", "Authority for company registration"),
    "CIN":    ("Corporate Identification Number", "corporate", "Unique identifier for Indian companies"),
    "DIN":    ("Director Identification Number", "corporate", "Unique ID for company directors"),
    "LLPIN":  ("Limited Liability Partnership Identification Number", "corporate", "Unique ID for LLPs"),
    "LLP":    ("Limited Liability Partnership", "corporate", "Hybrid business structure"),
    "JV":     ("Joint Venture", "corporate", "Business partnership for specific project"),
    "SPV":    ("Special Purpose Vehicle", "corporate", "Entity created for specific transaction"),
    "SPC":    ("Special Purpose Company", "corporate", "Company created for specific purpose"),
    "M&A":    ("Mergers and Acquisitions", "corporate", "Corporate consolidation transactions"),
    "LOI":    ("Letter of Intent", "corporate", "Preliminary agreement before formal deal"),
    "MOU":    ("Memorandum of Understanding", "corporate", "Non-binding agreement between parties"),
    "NDA":    ("Non-Disclosure Agreement", "corporate", "Confidentiality agreement"),
    "SHA":    ("Shareholder Agreement", "corporate", "Agreement governing shareholder rights"),
    "SSA":    ("Share Subscription Agreement", "corporate", "Agreement for issuing new shares"),
    "SPA":    ("Share Purchase Agreement", "corporate", "Agreement for buying existing shares"),
    "DMA_CORP": ("Due Diligence", "corporate", "Investigation before a transaction"),
    "SWAP":   ("Stock Swap", "corporate", "Acquisition paid with acquirer's shares"),
    "DEMERGER": ("Demerger", "corporate", "Spinning off a business division"),
    "BUYBACK": ("Share Buyback", "corporate", "Company repurchasing its own shares"),
    "DELISTING": ("Delisting", "corporate", "Removing shares from stock exchange"),
    "OPEN_OFFER": ("Open Offer", "corporate", "Mandatory offer to minority shareholders on acquisition"),
    "PAC":    ("Persons Acting in Concert", "corporate", "Group acting together for acquisition"),
    "PROMOTER": ("Promoter", "corporate", "Founder/controlling shareholder of Indian company"),
    "PLEDGE": ("Share Pledge", "corporate", "Shares pledged as collateral for loan"),
    "ENCUMBRANCE": ("Encumbrance", "corporate", "Lien or charge on shares/assets"),
    "CREEP":  ("Creeping Acquisition", "corporate", "Gradual stake increase — max 5% per year"),
    "CONTROL": ("Control", "corporate", "Ability to direct management of a company"),
    "SUBSIDIARY": ("Subsidiary", "corporate", "Company controlled by parent company"),
    "ASSOCIATE": ("Associate Company", "corporate", "Company with 20-50% stake held"),
    "HOLDING": ("Holding Company", "corporate", "Parent company owning subsidiaries"),
    "CONSOLIDATION": ("Consolidation", "corporate", "Combining subsidiary financials with parent"),
    "STANDALONE": ("Standalone", "corporate", "Financial results of parent company only"),
    "RELATED_PARTY": ("Related Party Transaction", "corporate", "Transaction between connected entities"),
    "CONTINGENT": ("Contingent Liability", "corporate", "Potential liability dependent on future event"),
    "GOODWILL": ("Goodwill", "corporate", "Intangible asset from acquisition premium"),
    "IMPAIRMENT": ("Impairment", "corporate", "Write-down of asset below carrying value"),
    "AMORTIZATION": ("Amortization", "corporate", "Gradual expensing of intangible asset cost"),
    "DEPRECIATION": ("Depreciation", "corporate", "Gradual expensing of tangible asset cost"),

    # ══════════════════════════════════════════
    # MACROECONOMICS & FIXED INCOME
    # ══════════════════════════════════════════

    "GDP":    ("Gross Domestic Product", "macro", "Total economic output of a country"),
    "GVA":    ("Gross Value Added", "macro", "GDP minus taxes plus subsidies"),
    "CPI":    ("Consumer Price Index", "macro", "Measure of retail inflation in India"),
    "WPI":    ("Wholesale Price Index", "macro", "Measure of wholesale inflation"),
    "IIP":    ("Index of Industrial Production", "macro", "Measure of manufacturing output"),
    "PMI":    ("Purchasing Managers Index", "macro", "Survey-based economic activity indicator"),
    "CAD":    ("Current Account Deficit", "macro", "Trade and income deficit with rest of world"),
    "BoP":    ("Balance of Payments", "macro", "Record of all transactions with foreign economies"),
    "FDI":    ("Foreign Direct Investment", "macro", "Foreign investment in productive assets"),
    "FPI":    ("Foreign Portfolio Investment", "macro", "Foreign investment in financial assets"),
    "FOREX":  ("Foreign Exchange", "macro", "Currency trading market"),
    "RES":    ("Foreign Exchange Reserves", "macro", "RBI's holdings of foreign currency assets"),
    "REER":   ("Real Effective Exchange Rate", "macro", "Inflation-adjusted trade-weighted exchange rate"),
    "NEER":   ("Nominal Effective Exchange Rate", "macro", "Trade-weighted nominal exchange rate"),
    "REPO":   ("Repurchase Rate", "macro", "Rate at which RBI lends to banks overnight"),
    "REV_REPO": ("Reverse Repo Rate", "macro", "Rate at which RBI borrows from banks"),
    "SDF":    ("Standing Deposit Facility", "macro", "RBI's floor rate for absorbing liquidity"),
    "MSF_RATE": ("Marginal Standing Facility Rate", "macro", "Emergency borrowing rate for banks"),
    "BANK_RATE": ("Bank Rate", "macro", "RBI's long-term lending rate"),
    "MPC":    ("Monetary Policy Committee", "macro", "RBI committee setting interest rates"),
    "MPS":    ("Monetary Policy Statement", "macro", "RBI's statement on interest rate decisions"),
    "YTM":    ("Yield to Maturity", "macro", "Total return if bond held to maturity"),
    "YTC":    ("Yield to Call", "macro", "Return if bond called before maturity"),
    "YIELD":  ("Yield", "macro", "Annual return on bond as % of price"),
    "GSEC":   ("Government Security", "macro", "Debt instrument issued by central/state government"),
    "SDL":    ("State Development Loan", "macro", "Bonds issued by state governments"),
    "Tbill":  ("Treasury Bill", "macro", "Short-term government security < 1 year"),
    "FRB":    ("Floating Rate Bond", "macro", "Bond with variable interest rate"),
    "IGB":    ("Inflation-indexed Government Bond", "macro", "Bond providing inflation protection"),
    "STRIPS": ("Separate Trading of Registered Interest and Principal Securities", "macro", "Zero-coupon bonds from coupon stripping"),
    "OIS":    ("Overnight Indexed Swap", "macro", "Interest rate swap using overnight rate"),
    "IRS":    ("Interest Rate Swap", "macro", "Derivative exchanging fixed for floating rates"),
    "CDS":    ("Credit Default Swap", "macro", "Insurance against bond default"),
    "CLO":    ("Collateralized Loan Obligation", "macro", "Structured product backed by loan portfolio"),
    "CDO":    ("Collateralized Debt Obligation", "macro", "Structured product backed by debt portfolio"),
    "MBS":    ("Mortgage Backed Security", "macro", "Security backed by pool of mortgages"),
    "ABS":    ("Asset Backed Security", "macro", "Security backed by pool of assets"),
    "RATING": ("Credit Rating", "macro", "Assessment of creditworthiness"),
    "AAA":    ("Triple A Rating", "macro", "Highest credit rating — minimal default risk"),
    "AA":     ("Double A Rating", "macro", "Very high credit quality"),
    "JUNK":   ("Junk Bond", "macro", "High-yield bond with below investment-grade rating"),
    "IG":     ("Investment Grade", "macro", "Bond rated BBB- or above"),
    "HY":     ("High Yield", "macro", "Bond rated below BBB- — also called junk"),
    "SPREAD": ("Credit Spread", "macro", "Yield difference between corporate and government bonds"),
    "OAS":    ("Option Adjusted Spread", "macro", "Spread after removing embedded option value"),
    "DURATION": ("Duration", "macro", "Bond's sensitivity to interest rate changes"),
    "CONVEXITY": ("Convexity", "macro", "Second-order measure of bond price sensitivity"),
    "DV01":   ("Dollar Value of 01", "macro", "Price change for 1 basis point yield move"),
    "BPS":    ("Basis Points", "macro", "1/100th of 1% — 100 bps = 1%"),
    "FISCAL": ("Fiscal Deficit", "macro", "Government spending exceeding revenue"),
    "REVENUE_DEFICIT": ("Revenue Deficit", "macro", "Revenue expenditure exceeding revenue receipts"),
    "PRIMARY_DEFICIT": ("Primary Deficit", "macro", "Fiscal deficit minus interest payments"),
    "FRBM":   ("Fiscal Responsibility and Budget Management Act", "macro", "Law capping India's fiscal deficit"),
    "UNION_BUDGET": ("Union Budget", "macro", "Annual government financial statement"),
    "VOTE_ON_ACCOUNT": ("Vote on Account", "macro", "Interim budget before elections"),
    "DISINVESTMENT": ("Disinvestment", "macro", "Government selling stake in PSUs"),
    "PSU":    ("Public Sector Undertaking", "macro", "Government-owned company"),
    "PSB":    ("Public Sector Bank", "macro", "Government-owned bank"),
    "CPSE":   ("Central Public Sector Enterprise", "macro", "Centrally owned government company"),

    # ══════════════════════════════════════════
    # INSURANCE
    # ══════════════════════════════════════════

    "ULIP":   ("Unit Linked Insurance Plan", "insurance", "Combined insurance and investment product"),
    "LIC":    ("Life Insurance Corporation", "insurance", "India's largest life insurer"),
    "GIC":    ("General Insurance Corporation", "insurance", "India's national reinsurer"),
    "SBI_LIFE": ("SBI Life Insurance", "insurance", "Private life insurer — SBI subsidiary"),
    "PREMIUM": ("Insurance Premium", "insurance", "Payment for insurance coverage"),
    "SUM_ASSURED": ("Sum Assured", "insurance", "Guaranteed amount paid on death/maturity"),
    "CLAIM":  ("Insurance Claim", "insurance", "Request for payment under insurance policy"),
    "IRR_INS": ("Internal Rate of Return", "insurance", "Returns from insurance product"),
    "ANNUITY": ("Annuity", "insurance", "Regular income stream from insurance/pension"),
    "REINSURANCE": ("Reinsurance", "insurance", "Insurance for insurance companies"),
    "SOLVENCY": ("Solvency Ratio", "insurance", "Insurer's ability to meet obligations"),
    "COMBINED_RATIO": ("Combined Ratio", "insurance", "Sum of loss ratio and expense ratio — <100% is profitable"),
    "LOSS_RATIO": ("Loss Ratio", "insurance", "Claims paid as % of premium earned"),
    "EXPENSE_RATIO_INS": ("Expense Ratio", "insurance", "Operating expenses as % of premium"),
    "EMBEDDED_VALUE": ("Embedded Value", "insurance", "Net present value of in-force insurance business"),
    "VNB":    ("Value of New Business", "insurance", "Present value of profits from new policies"),
    "VNB_MARGIN": ("VNB Margin", "insurance", "VNB as % of annualized premium equivalent"),
    "APE":    ("Annualized Premium Equivalent", "insurance", "Standardized measure of insurance premium"),
    "GWP":    ("Gross Written Premium", "insurance", "Total premium before reinsurance"),
    "NEP":    ("Net Earned Premium", "insurance", "Premium earned after reinsurance"),

    # ══════════════════════════════════════════
    # TECHNOLOGY & FINTECH
    # ══════════════════════════════════════════

    "API":    ("Application Programming Interface", "technology", "Interface for software communication"),
    "SaaS":   ("Software as a Service", "technology", "Cloud-based software delivery model"),
    "PaaS":   ("Platform as a Service", "technology", "Cloud platform for developers"),
    "IaaS":   ("Infrastructure as a Service", "technology", "Cloud infrastructure rental"),
    "AI":     ("Artificial Intelligence", "technology", "Machine simulation of human intelligence"),
    "ML":     ("Machine Learning", "technology", "AI subset learning from data"),
    "LLM":    ("Large Language Model", "technology", "AI model trained on text data"),
    "GenAI":  ("Generative AI", "technology", "AI that generates content"),
    "RAG":    ("Retrieval Augmented Generation", "technology", "AI combining retrieval and generation"),
    "NLP":    ("Natural Language Processing", "technology", "AI understanding human language"),
    "OCR":    ("Optical Character Recognition", "technology", "Converting images of text to machine text"),
    "RPA":    ("Robotic Process Automation", "technology", "Software bots automating repetitive tasks"),
    "BLOCKCHAIN": ("Blockchain", "technology", "Distributed ledger technology"),
    "DLT":    ("Distributed Ledger Technology", "technology", "Shared record-keeping system"),
    "CRYPTO": ("Cryptocurrency", "technology", "Digital currency using cryptography"),
    "NFT":    ("Non-Fungible Token", "technology", "Unique digital asset on blockchain"),
    "CBDC":   ("Central Bank Digital Currency", "technology", "Digital form of central bank money"),
    "EKYC":   ("Electronic Know Your Customer", "technology", "Digital identity verification"),
    "VKYC":   ("Video Know Your Customer", "technology", "Video-based identity verification"),
    "AA":     ("Account Aggregator", "technology", "RBI-regulated data sharing framework"),
    "OCEN":   ("Open Credit Enablement Network", "technology", "Credit infrastructure for digital lending"),
    "FidO":   ("Financial Data Operations", "technology", "Financial data management framework"),
    "ONDC":   ("Open Network for Digital Commerce", "technology", "Government's open e-commerce network"),
    "UPI2":   ("UPI 2.0", "technology", "Enhanced version of UPI with credit and overdraft"),
    "ENACH":  ("Electronic National Automated Clearing House", "technology", "Digital mandate system"),
    "CKYC":   ("Central Know Your Customer", "technology", "Central KYC registry"),
    "VRRR":   ("Variable Rate Reverse Repo", "technology", "RBI's sterilization tool"),
    "GSTN":   ("Goods and Services Tax Network", "technology", "IT infrastructure for GST"),
    "FINTECH": ("Financial Technology", "technology", "Technology-driven financial services"),
    "WEALTHTECH": ("Wealth Technology", "technology", "Technology for wealth management"),
    "INSURTECH": ("Insurance Technology", "technology", "Technology for insurance services"),
    "REGTECH": ("Regulatory Technology", "technology", "Technology for regulatory compliance"),
    "LENDTECH": ("Lending Technology", "technology", "Technology for lending services"),
    "PAYTECH": ("Payment Technology", "technology", "Technology for payment services"),
}


# ─────────────────────────────────────────────
# SIMPLIFIED DICTIONARY (for quick lookups)
# Format: "ACRONYM": "Full Form"
# ─────────────────────────────────────────────

FINANCIAL_TERMS: dict[str, str] = {
    acronym: details[0]
    for acronym, details in _TERMS_FULL.items()
}


# ─────────────────────────────────────────────
# CATEGORY-GROUPED TERMS
# ─────────────────────────────────────────────

TERMS_BY_CATEGORY: dict[str, dict[str, str]] = {}
for _acronym, (_full, _category, _context) in _TERMS_FULL.items():
    TERMS_BY_CATEGORY.setdefault(_category, {})[_acronym] = _full


# ─────────────────────────────────────────────
# REVERSE LOOKUP: Full Form → Acronym
# ─────────────────────────────────────────────

REVERSE_TERMS: dict[str, str] = {
    details[0].lower(): acronym
    for acronym, details in _TERMS_FULL.items()
}


# ─────────────────────────────────────────────
# AMBIGUOUS TERMS (same acronym, multiple meanings)
# ─────────────────────────────────────────────

AMBIGUOUS_TERMS: dict[str, list[str]] = {
    "PE": [
        "Price-to-Earnings Ratio (valuation context)",
        "Put Option European (derivatives context)",
        "Private Equity (investment context)",
    ],
    "PCR": [
        "Provision Coverage Ratio (banking NPA context)",
        "Put-Call Ratio (derivatives context)",
    ],
    "FPI": [
        "Foreign Portfolio Investor (regulatory context)",
        "Foreign Portfolio Investment (macro context)",
    ],
    "NAV": [
        "Net Asset Value (mutual fund / real estate context)",
        "Net Asset Value (company valuation context)",
    ],
    "CD": [
        "Certificate of Deposit (banking context)",
        "Convertible Debenture (corporate finance context)",
    ],
    "CP": [
        "Commercial Paper (debt market context)",
        "Circular/Compliance (regulatory context)",
    ],
    "IR": [
        "Information Ratio (investment context)",
        "Investor Relations (corporate context)",
        "Interest Rate (macro context)",
    ],
    "AA": [
        "Account Aggregator (fintech context)",
        "Double A Rating (credit rating context)",
        "Additional Tier 1 Approvals (banking context)",
    ],
}


# ─────────────────────────────────────────────
# PUBLIC API FUNCTIONS
# ─────────────────────────────────────────────

def expand_acronym(term: str) -> str:
    """
    Expands a financial acronym to its full form.
    Returns the original term if not found.

    Args:
        term: Acronym string (case-insensitive)

    Returns:
        Full form string, or original term if not found

    Examples:
        >>> expand_acronym("NIM")
        'Net Interest Margin'
        >>> expand_acronym("nim")
        'Net Interest Margin'
        >>> expand_acronym("UNKNOWN")
        'UNKNOWN'
    """
    normalized = term.strip().upper()
    return FINANCIAL_TERMS.get(normalized, term)


def expand_acronym_with_context(term: str) -> dict:
    """
    Returns full expansion with category and context.

    Args:
        term: Acronym string (case-insensitive)

    Returns:
        Dict with keys: acronym, full_form, category, context, is_ambiguous

    Examples:
        >>> expand_acronym_with_context("NIM")
        {
            'acronym': 'NIM',
            'full_form': 'Net Interest Margin',
            'category': 'banking',
            'context': 'Difference between interest income and interest paid...',
            'is_ambiguous': False
        }
    """
    normalized = term.strip().upper()
    if normalized in _TERMS_FULL:
        full, category, context = _TERMS_FULL[normalized]
        return {
            "acronym": normalized,
            "full_form": full,
            "category": category,
            "context": context,
            "is_ambiguous": normalized in AMBIGUOUS_TERMS,
            "alternate_meanings": AMBIGUOUS_TERMS.get(normalized, []),
        }
    return {
        "acronym": normalized,
        "full_form": term,
        "category": "unknown",
        "context": "",
        "is_ambiguous": False,
        "alternate_meanings": [],
    }


def get_all_acronyms() -> list[str]:
    """
    Returns sorted list of all known financial acronyms.

    Returns:
        Sorted list of acronym strings

    Examples:
        >>> acronyms = get_all_acronyms()
        >>> "NIM" in acronyms
        True
        >>> len(acronyms) > 500
        True
    """
    return sorted(FINANCIAL_TERMS.keys())


def is_financial_term(term: str) -> bool:
    """
    Checks if a term is a known financial acronym.

    Args:
        term: String to check (case-insensitive)

    Returns:
        True if term is a known financial acronym

    Examples:
        >>> is_financial_term("NIM")
        True
        >>> is_financial_term("nim")
        True
        >>> is_financial_term("HELLO")
        False
    """
    return term.strip().upper() in FINANCIAL_TERMS


def get_terms_by_category(category: str) -> dict[str, str]:
    """
    Returns all terms for a specific category.

    Args:
        category: Category name (banking, capital_market, tax, etc.)

    Returns:
        Dict of {acronym: full_form} for the category

    Examples:
        >>> banking_terms = get_terms_by_category("banking")
        >>> "NIM" in banking_terms
        True
    """
    return TERMS_BY_CATEGORY.get(category.lower(), {})


def get_all_categories() -> list[str]:
    """
    Returns list of all term categories.

    Returns:
        List of category name strings
    """
    return sorted(TERMS_BY_CATEGORY.keys())


def find_acronyms_in_text(text: str) -> list[dict]:
    """
    Scans text and returns all financial acronyms found with their expansions.

    Args:
        text: Input text to scan

    Returns:
        List of dicts with acronym details for each match found

    Examples:
        >>> results = find_acronyms_in_text("HDFC NIM improved while GNPA declined")
        >>> [r['acronym'] for r in results]
        ['NIM', 'GNPA']
    """
    import re
    # Match 2-10 char uppercase words (potential acronyms)
    pattern = r'\b[A-Z][A-Z0-9/_&]{1,9}\b'
    candidates = re.findall(pattern, text)
    found = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen and is_financial_term(candidate):
            seen.add(candidate)
            found.append(expand_acronym_with_context(candidate))
    return found


def expand_text_acronyms(text: str, inline: bool = True) -> str:
    """
    Replaces acronyms in text with expanded forms.

    Args:
        text: Input text containing acronyms
        inline: If True, replaces inline: "NIM" → "Net Interest Margin (NIM)"
                If False, appends glossary at end

    Returns:
        Text with expanded acronyms

    Examples:
        >>> expand_text_acronyms("HDFC NIM improved while GNPA declined")
        'HDFC Net Interest Margin (NIM) improved while Gross Non-Performing Assets (GNPA) declined'
    """
    import re
    pattern = r'\b([A-Z][A-Z0-9/_&]{1,9})\b'

    if inline:
        def replace_match(match):
            acronym = match.group(1)
            if is_financial_term(acronym):
                full_form = expand_acronym(acronym)
                return f"{full_form} ({acronym})"
            return acronym

        return re.sub(pattern, replace_match, text)
    else:
        found = find_acronyms_in_text(text)
        if not found:
            return text
        glossary = "\n\nGLOSSARY:\n" + "\n".join(
            f"  {item['acronym']}: {item['full_form']}" for item in found
        )
        return text + glossary


def get_term_count() -> int:
    """Returns total number of financial terms in dictionary."""
    return len(FINANCIAL_TERMS)


def search_terms(query: str) -> list[dict]:
    """
    Fuzzy search across acronyms and full forms.

    Args:
        query: Search string

    Returns:
        List of matching term dicts

    Examples:
        >>> results = search_terms("interest margin")
        >>> any("NIM" in r['acronym'] for r in results)
        True
    """
    query_lower = query.lower()
    results = []
    for acronym, (full_form, category, context) in _TERMS_FULL.items():
        if (
            query_lower in acronym.lower()
            or query_lower in full_form.lower()
            or query_lower in context.lower()
        ):
            results.append({
                "acronym": acronym,
                "full_form": full_form,
                "category": category,
                "context": context,
            })
    return sorted(results, key=lambda x: x["acronym"])
