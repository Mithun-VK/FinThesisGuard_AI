# test_orchestrator.py
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from backend.pipeline.orchestrator import (
    orchestrator,
    CircuitBreaker,
    CircuitState,
    MetricsCollector,
    _make_cache_key,
    _new_request_id,
    _record_error,
    _fallback_answer,
)
from backend.models.query_models import QueryType


# ─────────────────────────────────────────────────────────────────────
# SAMPLE QUERIES
# ─────────────────────────────────────────────────────────────────────

FACTUAL_QUERY     = "What is HDFC Bank's Net Interest Margin for Q3 FY26?"
COMPARATIVE_QUERY = "Compare HDFC Bank vs ICICI Bank NIM and GNPA for Q3 FY26"
REGULATORY_QUERY  = "Latest RBI circular on NBFC liquidity coverage ratio norms"
VALID_THESIS      = """
HDFC Bank will outperform the sector in FY27 because NIM expansion of 20bps
is expected as RBI cuts repo rate by 75bps over 12 months, driving PAT growth
of 18-20% and enabling a re-rating from 2.5x to 3.2x book value as GNPA
remains below 1.5% and CASA ratio holds above 42%.
"""
SHORT_THESIS = "HDFC is good"


# ─────────────────────────────────────────────────────────────────────
# MOCK DATA
# ─────────────────────────────────────────────────────────────────────

MOCK_CHUNKS = [
    {
        "id"              : "c1",
        "text"            : (
            "HDFC Bank reported NIM of 4.2% for Q3 FY26. "
            "GNPA ratio stands at 1.26%, lowest in 5 years. "
            "CASA ratio was 42.3% as of December 2025."
        ),
        "source"          : "HDFC Bank Q3 FY26 Earnings Report",
        "source_type"     : "annual_report",
        "final_score"     : 0.92,
        "authority_weight": 1.3,
        "date"            : "2026-01-15",
        "url"             : "https://hdfcbank.com/investor-relations/q3fy26",
        "title"           : "HDFC Bank Q3 FY26 Earnings",
        "metadata"        : {"source_type": "annual_report", "company": "HDFC Bank"},
    },
    {
        "id"              : "c2",
        "text"            : (
            "RBI MPC voted 4-2 to cut repo rate by 25bps to 6.25% in February 2026. "
            "Further cuts of 50bps expected in H1 FY27 given CPI within 4% target. "
            "MCLR-linked loans constitute 58% of HDFC Bank book."
        ),
        "source"          : "RBI MPC Minutes February 2026",
        "source_type"     : "rbi",
        "final_score"     : 0.88,
        "authority_weight": 1.5,
        "date"            : "2026-02-10",
        "url"             : "https://rbi.org.in/scripts/BS_PressReleaseDisplay.aspx",
        "title"           : "RBI MPC February 2026 Statement",
        "metadata"        : {"source_type": "rbi"},
    },
    {
        "id"              : "c3",
        "text"            : (
            "ICICI Bank NIM came in at 3.9% for Q3 FY26, down 5bps QoQ. "
            "GNPA ratio at 1.96%, improved from 2.1% in Q2. "
            "Credit growth at 14.2% YoY for ICICI vs 16.1% for HDFC."
        ),
        "source"          : "Motilal Oswal Banking Sector Report Jan 2026",
        "source_type"     : "broker_research",
        "final_score"     : 0.78,
        "authority_weight": 1.0,
        "date"            : "2026-01-20",
        "url"             : "https://motilaloswalsecurities.com/reports",
        "title"           : "Banking Sector Q3 FY26 Review",
        "metadata"        : {"source_type": "broker_research"},
    },
]

MOCK_RAG_ANSWER = (
    "HDFC Bank's NIM for Q3 FY26 is **4.2%** [1], compared to ICICI Bank's "
    "3.9% [3]. HDFC maintains a 30bps NIM premium driven by its superior CASA "
    "ratio of 42.3% [1] and a higher share of retail loans."
)

# FIX: historical_analogs must be a list of dicts matching the HistoricalAnalog
# Pydantic schema — NOT plain strings. Pydantic v2 cannot coerce str → model.
# Fields are derived from the ThesisPipeline / Agent 6 output contract.
MOCK_THESIS_DATA = {
    "assumptions": [
        {
            "id": 1, "text": "RBI cuts repo rate by 75bps over 12 months",
            "category": "macro", "confidence": 0.65, "dependency_on": [],
            "is_critical": True, "historical_support": True,
        },
        {
            "id": 2, "text": "NIM expands 20bps on MCLR repricing",
            "category": "margin", "confidence": 0.60, "dependency_on": [1],
            "is_critical": True, "historical_support": False,
        },
        {
            "id": 3, "text": "PAT grows 18-20% on NIM and credit growth",
            "category": "demand", "confidence": 0.58, "dependency_on": [2],
            "is_critical": True, "historical_support": False,
        },
        {
            "id": 4, "text": "Stock re-rates from 2.5x to 3.2x book",
            "category": "valuation", "confidence": 0.50, "dependency_on": [3],
            "is_critical": False, "historical_support": False,
        },
    ],
    "dependency_chain"       : ["RBI rate cut -> NIM expansion -> PAT growth 18-20% -> re-rating 3.2x book"],
    # FIX: each entry is now a dict matching HistoricalAnalog fields,
    # not a bare string. Required fields: title, period, outcome, relevance, similarity.
    "historical_analogs": [
        {
            "title"           : "HDFC NIM expansion cycle 2016-2018",
            "period"          : "2016-2018",
            "outcome"         : "NIM expanded 30bps over 18 months following RBI rate-cut cycle",
            "lesson"          : "Rate-cut-driven margin expansion materialises with 2–3 quarter lag",
            "similarity_score": 0.82,
       },
        {
            "title"           : "Banking re-rating post RBI easing 2019",
            "period"          : "2019",
            "outcome"         : "Banking index re-rated 22% over 9 months on policy normalisation",
            "lesson"          : "Sector re-ratings require sustained earnings delivery, not just rate cuts",
            "similarity_score": 0.74,
        },
    ],
    "structural_robustness"  : "Medium",
    "has_circular_dependency": False,
}

# FIX: MOCK_STRESS_DATA is merged into thesis_data via state["thesis_data"].update(synthesis)
# inside _node_stress_synthesize. It does NOT need to carry historical_analogs because
# those survive from Agent 6's output (MOCK_THESIS_DATA) — update() only overwrites
# keys that are present in the new dict. Keeping stress data focused on what Agent 7
# actually produces (risks, break_conditions, strength, robustness, confidence).
MOCK_STRESS_DATA = {
    "risks": [
        {"dimension": "Valuation",  "score": 8, "rationale": "3.2x book is 3-year high; stretched vs peers"},
        {"dimension": "Macro",      "score": 6, "rationale": "75bps cut in 12 months more than consensus 50bps"},
        {"dimension": "Margin",     "score": 5, "rationale": "20bps expansion assumes no deposit repricing lag"},
        {"dimension": "Demand",     "score": 4, "rationale": "16% credit growth in line with recent trajectory"},
        {"dimension": "Regulatory", "score": 3, "rationale": "LCR norms stable; no near-term tightening signals"},
        {"dimension": "Liquidity",  "score": 3, "rationale": "CASA at 42.3%, adequate liquidity buffer"},
    ],
    "break_conditions": [
        {
            "condition"     : "RBI delivers fewer than 50bps of cuts by Q2 FY27",
            "trigger_metric": "repo_rate",
            "threshold"     : "< 50bps cumulative cuts",
            "probability"   : "Medium",
        },
        {
            "condition"     : "GNPA rises above 1.8% on unsecured retail stress",
            "trigger_metric": "gnpa_ratio",
            "threshold"     : "> 1.8%",
            "probability"   : "Low",
        },
    ],
    "thesis_strength"      : "Medium",
    "structural_robustness": "Medium",
    "confidence"           : "Medium",
}


# ─────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────

def _blank_state(query=None, is_thesis=False):
    query = query or FACTUAL_QUERY
    return {
        "request_id"       : _new_request_id(),
        "query"            : query,
        "query_type"       : "",
        "is_thesis"        : is_thesis,
        "expanded_query"   : query,
        "sub_queries"      : [],
        "retrieval_filters": {},
        "chunks"           : [],
        "reranked_chunks"  : [],
        "conflicts"        : [],
        "quant_issues"     : [],
        "answer"           : "",
        "thesis_data"      : {},
        "response"         : {},
        "confidence"       : "Low",
        "node_timings"     : {},
        "total_latency_ms" : 0.0,
        "error"            : None,
        "cache_hit"        : False,
    }


# ─────────────────────────────────────────────────────────────────────
# TEST 1: Query type classifier
# ─────────────────────────────────────────────────────────────────────
print("=== Test 1: _classify_query_type ===")


def test_classifier():
    from backend.pipeline.orchestrator import Orchestrator
    tests = [
        (FACTUAL_QUERY,     QueryType.FACTUAL,     "factual"),
        (COMPARATIVE_QUERY, QueryType.COMPARATIVE, "comparative"),
        (REGULATORY_QUERY,  QueryType.REGULATORY,  "regulatory"),
        (VALID_THESIS,      QueryType.THESIS,      "thesis"),
        ("HDFC is good",    QueryType.FACTUAL,     "short thesis -> factual"),
    ]
    for query, expected, label in tests:
        result = Orchestrator._classify_query_type(query)
        assert result == expected, f"FAIL [{label}]: expected {expected}, got {result}"
        print(f"  ok [{label}] -> {result}")

    f = Orchestrator._build_retrieval_filters(QueryType.REGULATORY)
    assert "source_type" in f
    assert "rbi"  in f["source_type"]["$in"]
    assert "sebi" in f["source_type"]["$in"]
    print(f"  ok REGULATORY filter -> {f['source_type']['$in']}")

    assert Orchestrator._build_retrieval_filters(QueryType.FACTUAL) == {}
    print("  ok FACTUAL filter -> empty dict")


test_classifier()


# ─────────────────────────────────────────────────────────────────────
# TEST 2: Cache key generation
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 2: _make_cache_key ===")


def test_cache_keys():
    k1 = _make_cache_key("rag",    "What is HDFC NIM?")
    k2 = _make_cache_key("rag",    "What is HDFC NIM?")
    k3 = _make_cache_key("rag",    "WHAT IS HDFC NIM?")
    k4 = _make_cache_key("thesis", "What is HDFC NIM?")
    assert k1 == k2,                 "Same input -> same key"
    assert k1 == k3,                 "Case-insensitive normalisation"
    assert k1 != k4,                 "Different namespace -> different key"
    assert k1.startswith("rag:"),    "rag: prefix"
    assert k4.startswith("thesis:"), "thesis: prefix"
    print(f"  ok Deterministic:    {k1}")
    print(f"  ok Case-insensitive: {k1} == {k3}")
    print("  ok Namespace-scoped: rag != thesis")
    ids = {_new_request_id() for _ in range(100)}
    assert len(ids) == 100, "request IDs must be unique"
    print("  ok 100 unique request IDs generated")


test_cache_keys()


# ─────────────────────────────────────────────────────────────────────
# TEST 3: _record_error — first error wins, no overwrite
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 3: _record_error ===")


def test_record_error():
    state = {"error": None, "node_timings": {}}
    state = _record_error(state, "retrieve",  ValueError("Pinecone timeout"))
    assert "retrieve" in state["error"]
    print(f"  ok First error recorded: {state['error']}")

    state = _record_error(state, "synthesize", Exception("Groq 503"))
    assert "retrieve"   in state["error"]
    assert "synthesize" not in state["error"]
    print("  ok First error NOT overwritten by second")

    msg = _fallback_answer("Groq 503")
    assert "Groq 503" in msg
    print("  ok Fallback answer contains error hint")

    msg2 = _fallback_answer(None)
    assert "unable" in msg2.lower()
    print("  ok Fallback answer without error is safe")


test_record_error()


# ─────────────────────────────────────────────────────────────────────
# TEST 4: CircuitBreaker — full lifecycle
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 4: CircuitBreaker lifecycle ===")


async def test_circuit_breaker():
    # FIX: recovery_timeout=0 causes a monotonic clock race on fast machines.
    # 0.05s gives the clock enough headroom; we sleep 0.06s before checking.
    cb = CircuitBreaker("test-cb", failure_threshold=3, recovery_timeout=0.05)
    assert cb.state == CircuitState.CLOSED
    print("  ok Initial state: CLOSED")

    async def succeed():
        return "ok"

    async def fail():
        raise RuntimeError("service down")

    result = await cb.call(succeed())
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED
    print("  ok Successful call -> stays CLOSED")

    for _ in range(3):
        try:
            await cb.call(fail())
        except RuntimeError:
            pass

    assert cb._state == CircuitState.OPEN
    print(f"  ok 3 failures -> OPEN (failure_count={cb._failure_count})")

    # FIX: calling cb.call(succeed()) when OPEN raises before the coroutine is
    # awaited, leaving it unawaited. Close it explicitly to suppress the warning.
    _open_coro = succeed()
    try:
        await cb.call(_open_coro)
        assert False, "Should have raised"
    except RuntimeError as e:
        _open_coro.close()          # suppress "coroutine never awaited" warning
        assert "OPEN" in str(e)
        print(f"  ok OPEN blocks new calls: {e}")

    # FIX: sleep past recovery_timeout before reading state
    await asyncio.sleep(0.06)
    assert cb.state == CircuitState.HALF_OPEN, (
        f"Expected HALF_OPEN, got {cb.state} "
        f"(age={(asyncio.get_event_loop().time() - cb._last_failure_time)*1000:.1f}ms)"
    )
    print("  ok After timeout -> HALF_OPEN")

    result = await cb.call(succeed())
    assert result == "ok"
    assert cb._state == CircuitState.CLOSED
    assert cb._failure_count == 0
    print("  ok Probe success -> CLOSED (failure_count reset to 0)")


asyncio.run(test_circuit_breaker())


# ─────────────────────────────────────────────────────────────────────
# TEST 5: MetricsCollector
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 5: MetricsCollector ===")


async def test_metrics():
    m = MetricsCollector()
    s = m.snapshot()
    assert s["total_queries"] == 0
    print("  ok Initial state: all zeros")

    await m.record("FACTUAL",     1200, cache_hit=False, error=False)
    await m.record("FACTUAL",      800, cache_hit=True,  error=False)
    await m.record("COMPARATIVE", 1800, cache_hit=False, error=False)
    await m.record("THESIS",      2500, cache_hit=False, error=True)
    await m.record("REGULATORY",   950, cache_hit=True,  error=False)

    s = m.snapshot()
    assert s["total_queries"] == 5
    assert s["cache_hits"]    == 2
    assert s["errors"]        == 1
    assert abs(s["cache_hit_rate"] - 0.40) < 0.01
    assert abs(s["error_rate"]     - 0.20) < 0.01
    assert s["by_query_type"]["FACTUAL"]     == 2
    assert s["by_query_type"]["THESIS"]      == 1
    assert s["by_query_type"]["COMPARATIVE"] == 1
    assert s["by_query_type"]["REGULATORY"]  == 1
    assert s["p50_latency_ms"] <= s["p95_latency_ms"] <= s["p99_latency_ms"]
    print(f"  ok total_queries:  {s['total_queries']}")
    print(f"  ok cache_hit_rate: {s['cache_hit_rate']:.2f}")
    print(f"  ok error_rate:     {s['error_rate']:.2f}")
    print(f"  ok p50={s['p50_latency_ms']}ms  p95={s['p95_latency_ms']}ms  p99={s['p99_latency_ms']}ms")
    print(f"  ok by_query_type:  {s['by_query_type']}")


asyncio.run(test_metrics())


# ─────────────────────────────────────────────────────────────────────
# TEST 6: Node — _node_classify
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 6: Node — _node_classify ===")


async def test_node_classify():
    state = _blank_state(FACTUAL_QUERY)
    result = await orchestrator._node_classify(state)
    assert result["query_type"] == QueryType.FACTUAL
    assert result["is_thesis"]  is False
    assert "classify" in result["node_timings"]
    print("  ok FACTUAL query classified correctly")

    state = _blank_state(REGULATORY_QUERY)
    result = await orchestrator._node_classify(state)
    assert result["query_type"] == QueryType.REGULATORY
    assert "rbi" in result["retrieval_filters"]["source_type"]["$in"]
    print("  ok REGULATORY query -> rbi/sebi filter applied")

    state = _blank_state(VALID_THESIS)
    result = await orchestrator._node_classify(state)
    assert result["query_type"] == QueryType.THESIS
    assert result["is_thesis"]  is True
    print("  ok THESIS query classified correctly")


asyncio.run(test_node_classify())


# ─────────────────────────────────────────────────────────────────────
# TEST 7: Node — _node_acronym_resolve
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 7: Node — _node_acronym_resolve ===")


async def test_node_acronym_resolve():
    state    = _blank_state("What is HDFC NIM and GNPA?")
    expanded = "What is HDFC Net Interest Margin (NIM) and Gross Non-Performing Assets (GNPA)?"

    with patch.object(orchestrator.agent1, "resolve",
                      new=AsyncMock(return_value=expanded)):
        result = await orchestrator._node_acronym_resolve(state)
    assert "Net Interest Margin" in result["expanded_query"]
    assert "acronym_resolve" in result["node_timings"]
    print(f"  ok NIM expanded: {result['expanded_query'][:60]}...")

    with patch.object(orchestrator.agent1, "resolve",
                      new=AsyncMock(side_effect=Exception("LLM OOM"))):
        result2 = await orchestrator._node_acronym_resolve(state)
    assert result2["expanded_query"] == state["query"]
    assert result2["error"] is None
    print("  ok LLM failure -> graceful fallback to raw query (no crash)")


asyncio.run(test_node_acronym_resolve())


# ─────────────────────────────────────────────────────────────────────
# TEST 8: Node — _node_decompose
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 8: Node — _node_decompose ===")


async def test_node_decompose():
    state = _blank_state(FACTUAL_QUERY)
    state["query_type"] = QueryType.FACTUAL

    with patch.object(orchestrator.agent2, "is_complex_query",
                      new=AsyncMock(return_value=False)):
        result = await orchestrator._node_decompose(state)
    assert result["sub_queries"] == [FACTUAL_QUERY]
    print("  ok Simple query -> single sub-query (decompose skipped)")

    state["query_type"]     = QueryType.COMPARATIVE
    state["expanded_query"] = COMPARATIVE_QUERY
    sub_q = ["HDFC NIM Q3 FY26", "ICICI NIM Q3 FY26",
             "HDFC GNPA Q3 FY26", "ICICI GNPA Q3 FY26"]

    with patch.object(orchestrator.agent2, "is_complex_query",
                      new=AsyncMock(return_value=True)), \
         patch.object(orchestrator.agent2, "decompose",
                      new=AsyncMock(return_value=sub_q)):
        result2 = await orchestrator._node_decompose(state)
    assert len(result2["sub_queries"]) == 4
    print(f"  ok Comparative query -> {len(result2['sub_queries'])} sub-queries:")
    for q in result2["sub_queries"]:
        print(f"    -> {q}")


asyncio.run(test_node_decompose())


# ─────────────────────────────────────────────────────────────────────
# TEST 9: Node — _node_retrieve + _node_rerank
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 9: Node — _node_retrieve + _node_rerank ===")


async def test_node_retrieve_and_rerank():
    state = _blank_state(FACTUAL_QUERY)
    state["sub_queries"] = [FACTUAL_QUERY]

    with patch.object(orchestrator.retriever, "retrieve_multi",
                      new=AsyncMock(return_value=MOCK_CHUNKS)):
        result = await orchestrator._node_retrieve(state)
    assert len(result["chunks"]) == 3
    assert "retrieve" in result["node_timings"]
    print(f"  ok Retrieved {len(result['chunks'])} chunks:")
    for c in result["chunks"]:
        print(f"    [{c['final_score']:.2f}] {c['title']}")

    shuffled = [MOCK_CHUNKS[2], MOCK_CHUNKS[0], MOCK_CHUNKS[1]]
    with patch.object(orchestrator.agent3, "rerank",
                      new=AsyncMock(return_value=sorted(
                          shuffled, key=lambda c: -c["final_score"]))):
        result["chunks"] = shuffled
        result2 = await orchestrator._node_rerank(result)
    scores = [c["final_score"] for c in result2["reranked_chunks"]]
    assert scores == sorted(scores, reverse=True)
    print(f"  ok Rerank order: {scores} (descending confirmed)")

    with patch.object(orchestrator.retriever, "retrieve_multi",
                      new=AsyncMock(side_effect=Exception("Pinecone 503"))):
        state["chunks"] = []
        fail_result = await orchestrator._node_retrieve(state)
    assert fail_result["chunks"] == []
    assert fail_result["error"] is not None
    print("  ok Pinecone failure -> empty chunks + error recorded (no crash)")


asyncio.run(test_node_retrieve_and_rerank())


# ─────────────────────────────────────────────────────────────────────
# TEST 10: Node — conflict_detect + quant_validate (parallel)
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 10: Node — conflict_detect + quant_validate (parallel) ===")


async def test_parallel_validation():
    state = _blank_state(COMPARATIVE_QUERY)
    state["query_type"]      = QueryType.COMPARATIVE
    state["chunks"]          = MOCK_CHUNKS
    state["reranked_chunks"] = MOCK_CHUNKS

    mock_conflict = MagicMock()
    mock_conflict.model_dump = MagicMock(return_value={
        "field"     : "NIM",
        "source_a"  : "HDFC Q3 FY26 Annual Report",
        "source_b"  : "Motilal Oswal Report",
        "value_a"   : "4.2%",
        "value_b"   : "4.1%",
        "severity"  : "Minor",
        "resolution": "Annual report (authority=1.3) preferred over broker note (authority=1.0)",
    })

    with patch.object(orchestrator.agent4, "detect_conflicts",
                      new=AsyncMock(return_value=[mock_conflict])), \
         patch.object(orchestrator.agent5, "validate_quantities",
                      new=AsyncMock(return_value=["NIM formula check: 4.2% = NII/AvgAssets ok"])):
        cd_result, qv_result = await asyncio.gather(
            orchestrator._node_conflict_detect(dict(state)),
            orchestrator._node_quant_validate(dict(state)),
        )
    assert len(cd_result["conflicts"])    == 1
    assert len(qv_result["quant_issues"]) == 1
    assert cd_result["conflicts"][0]["field"] == "NIM"
    print(f"  ok Conflict: {cd_result['conflicts'][0]['field']} | {cd_result['conflicts'][0]['severity']}")
    print(f"    {cd_result['conflicts'][0]['resolution']}")
    print(f"  ok Quant issue: {qv_result['quant_issues'][0]}")

    with patch.object(orchestrator.agent4, "detect_conflicts",
                      new=AsyncMock(side_effect=Exception("Agent4 OOM"))), \
         patch.object(orchestrator.agent5, "validate_quantities",
                      new=AsyncMock(return_value=["check passed"])):
        cd_fail, qv_ok = await asyncio.gather(
            orchestrator._node_conflict_detect(dict(state)),
            orchestrator._node_quant_validate(dict(state)),
        )
    assert cd_fail["conflicts"]       == []
    assert len(qv_ok["quant_issues"]) == 1
    print("  ok Agent4 crash -> empty conflicts (graceful), Agent5 still ran")


asyncio.run(test_parallel_validation())


# ─────────────────────────────────────────────────────────────────────
# TEST 11: Node — _node_synthesize
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 11: Node — _node_synthesize ===")


async def test_node_synthesize():
    state = _blank_state(FACTUAL_QUERY)
    state["reranked_chunks"] = MOCK_CHUNKS
    state["conflicts"]       = []

    with patch.object(orchestrator.rag_pipeline, "synthesize_answer",
                      new=AsyncMock(return_value=(MOCK_RAG_ANSWER, "High"))):
        result = await orchestrator._node_synthesize(state)
    assert "4.2%" in result["answer"]
    assert result["confidence"] == "High"
    assert "synthesize" in result["node_timings"]
    print(f"  ok Answer: {result['answer'][:80]}...")
    print(f"  ok Confidence: {result['confidence']}")

    with patch.object(orchestrator.rag_pipeline, "synthesize_answer",
                      new=AsyncMock(side_effect=Exception("Groq 503"))):
        fail_result = await orchestrator._node_synthesize(state)
    assert isinstance(fail_result["answer"], str)
    assert fail_result["confidence"] == "Low"
    assert "unable" in fail_result["answer"].lower()
    print("  ok Groq 503 -> fallback answer, confidence=Low (no crash)")


asyncio.run(test_node_synthesize())


# ─────────────────────────────────────────────────────────────────────
# TEST 12: run_query() — cache hit / miss / full pipeline
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 12: run_query() — cache hit / miss / full pipeline ===")


async def test_run_query():
    # orchestrator._initialized = True bypasses the guard completely.
    # DO NOT call orchestrator.initialize() — it is async and would attempt
    # live Pinecone/Redis/Groq connections. Calling it without await
    # creates an unawaited coroutine (RuntimeWarning) and does nothing.
    orchestrator._initialized = True

    try:
        cached_response = {
            "answer"     : "Cached HDFC NIM is 4.2%",
            "citations"  : [], "confidence": "High",
            "conflicts"  : [], "quant_issues": [],
            "data_gaps"  : [], "reasoning": "", "agents_used": [], "latency_ms": 50,
        }

        # ── Cache HIT ─────────────────────────────────────────────────────────
        with patch("backend.utils.cache.cache.get_cached",
                   new=AsyncMock(return_value=cached_response)), \
             patch("backend.utils.cache.cache.set_cache",
                   new=AsyncMock()):
            r = await orchestrator.run_query(FACTUAL_QUERY, use_cache=True)
        assert r.answer == "Cached HDFC NIM is 4.2%"
        print(f"  ok Cache HIT -> returned instantly: '{r.answer[:50]}...'")

        # ── Cache MISS → full pipeline ────────────────────────────────────────
        with patch("backend.utils.cache.cache.get_cached",
                   new=AsyncMock(return_value=None)), \
             patch("backend.utils.cache.cache.set_cache",
                   new=AsyncMock()), \
             patch.object(orchestrator.agent1, "resolve",
                          new=AsyncMock(return_value=FACTUAL_QUERY)), \
             patch.object(orchestrator.agent2, "is_complex_query",
                          new=AsyncMock(return_value=False)), \
             patch.object(orchestrator.retriever, "retrieve_multi",
                          new=AsyncMock(return_value=MOCK_CHUNKS)), \
             patch.object(orchestrator.agent3, "rerank",
                          new=AsyncMock(return_value=MOCK_CHUNKS)), \
             patch.object(orchestrator.agent4, "detect_conflicts",
                          new=AsyncMock(return_value=[])), \
             patch.object(orchestrator.agent5, "validate_quantities",
                          new=AsyncMock(return_value=[])), \
             patch.object(orchestrator.rag_pipeline, "synthesize_answer",
                          new=AsyncMock(return_value=(MOCK_RAG_ANSWER, "High"))):
            r2 = await orchestrator.run_query(FACTUAL_QUERY, use_cache=True)
        assert "4.2%" in r2.answer
        assert r2.confidence == "High"
        assert r2.latency_ms >= 0
        print(f"  ok Cache MISS -> full pipeline ran: '{r2.answer[:60]}...'")
        print(f"  ok Confidence: {r2.confidence} | latency: {r2.latency_ms}ms")
        print(f"  ok Agents used: {r2.agents_used}")

    finally:
        orchestrator._initialized = False

asyncio.run(test_run_query())


# ─────────────────────────────────────────────────────────────────────
# TEST 13: run_thesis() — validation, full pipeline
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 13: run_thesis() — validation + full pipeline ===")


async def test_run_thesis():
    orchestrator._initialized = True

    try:
        # ── Invalid thesis rejected before LLM call ───────────────────────────
        with patch.object(orchestrator.thesis_pipeline, "validate_thesis_input",
                          return_value=(False, "No reasoning provided")):
            try:
                await orchestrator.run_thesis(SHORT_THESIS)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Thesis validation failed" in str(e)
                print(f"  ok Short thesis rejected before LLM call: {e}")

        # ── Valid thesis → full pipeline ──────────────────────────────────────
        # NOTE on mock layering:
        #   agent6.analyze  returns MOCK_THESIS_DATA  (Agent 6 output — includes
        #                   historical_analogs as proper dicts)
        #   agent7.synthesize returns MOCK_STRESS_DATA (Agent 7 output — risks,
        #                   break_conditions, strength, robustness, confidence)
        #
        #   _node_stress_synthesize does: state["thesis_data"].update(synthesis)
        #   So the final thesis_data is a merge of both dicts. Keys in
        #   MOCK_STRESS_DATA overwrite the same keys from MOCK_THESIS_DATA,
        #   but historical_analogs (only in MOCK_THESIS_DATA) survives intact.
        with patch("backend.utils.cache.cache.get_cached",
                   new=AsyncMock(return_value=None)), \
             patch("backend.utils.cache.cache.set_cache",
                   new=AsyncMock()), \
             patch.object(orchestrator.thesis_pipeline, "validate_thesis_input",
                          return_value=(True, "ok")), \
             patch.object(orchestrator.agent1, "resolve",
                          new=AsyncMock(return_value=VALID_THESIS)), \
             patch.object(orchestrator.retriever, "retrieve_thesis_context",
                          new=AsyncMock(return_value=MOCK_CHUNKS)), \
             patch.object(orchestrator.agent6, "analyze",
                          new=AsyncMock(return_value=MOCK_THESIS_DATA)), \
             patch.object(orchestrator.agent5, "validate_quantities",
                          new=AsyncMock(return_value=[])), \
             patch.object(orchestrator.agent7, "synthesize",
                          new=AsyncMock(return_value=MOCK_STRESS_DATA)):
            r = await orchestrator.run_thesis(VALID_THESIS)

        assert r.thesis_strength       == "Medium"
        assert r.structural_robustness == "Medium"
        assert r.confidence            == "Medium"
        assert len(r.assumptions)      == 4
        assert len(r.risks)            == 6
        assert len(r.break_conditions) == 2
        assert len(r.dependency_chain) == 1
        assert len(r.historical_analogs) == 2
        assert r.latency_ms            >= 0
        print(f"  ok Thesis strength: {r.thesis_strength}")
        print(f"  ok Assumptions: {len(r.assumptions)}")
        for a in r.assumptions:
            print(f"    [{a.id}] [{a.category}] conf={a.confidence} "
                  f"critical={a.is_critical} deps={a.dependency_on}")

        for risk in r.risks:
            filled = risk.score
            bar    = (chr(9608) * filled) + (chr(9617) * (10 - filled))
            print(f"    {risk.dimension:12} [{bar}] {risk.score}/10  {risk.rationale}")

        for bc in r.break_conditions:
            print(f"    [{bc.probability}] {bc.condition}")

        for ha in r.historical_analogs:
            print(f"    [{ha.period}] {ha.title} (similarity={ha.similarity_score})")

    finally:
        orchestrator._initialized = False


asyncio.run(test_run_thesis())


# ─────────────────────────────────────────────────────────────────────
# TEST 14: Graceful degradation — all failure modes
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 14: Graceful degradation ===")


async def test_graceful_degradation():
    orchestrator._initialized = True

    try:
        # ── Pinecone down → fallback answer, no crash ─────────────────────────
        with patch("backend.utils.cache.cache.get_cached",
                   new=AsyncMock(return_value=None)), \
             patch("backend.utils.cache.cache.set_cache",
                   new=AsyncMock()), \
             patch.object(orchestrator.agent1, "resolve",
                          new=AsyncMock(return_value=FACTUAL_QUERY)), \
             patch.object(orchestrator.agent2, "is_complex_query",
                          new=AsyncMock(return_value=False)), \
             patch.object(orchestrator.retriever, "retrieve_multi",
                          new=AsyncMock(side_effect=Exception("Pinecone 503"))), \
             patch.object(orchestrator.agent3, "rerank",
                          new=AsyncMock(return_value=[])), \
             patch.object(orchestrator.agent4, "detect_conflicts",
                          new=AsyncMock(return_value=[])), \
             patch.object(orchestrator.agent5, "validate_quantities",
                          new=AsyncMock(return_value=[])), \
             patch.object(orchestrator.rag_pipeline, "synthesize_answer",
                          new=AsyncMock(return_value=("Fallback answer.", "Low"))):
            r = await orchestrator.run_query(FACTUAL_QUERY)
        assert r is not None
        print("  ok Pinecone down -> degraded response returned (no crash)")

        # ── Groq down → Low confidence fallback ──────────────────────────────
        with patch("backend.utils.cache.cache.get_cached",
                   new=AsyncMock(return_value=None)), \
             patch("backend.utils.cache.cache.set_cache",
                   new=AsyncMock()), \
             patch.object(orchestrator.agent1, "resolve",
                          new=AsyncMock(return_value=FACTUAL_QUERY)), \
             patch.object(orchestrator.agent2, "is_complex_query",
                          new=AsyncMock(return_value=False)), \
             patch.object(orchestrator.retriever, "retrieve_multi",
                          new=AsyncMock(return_value=MOCK_CHUNKS)), \
             patch.object(orchestrator.agent3, "rerank",
                          new=AsyncMock(return_value=MOCK_CHUNKS)), \
             patch.object(orchestrator.agent4, "detect_conflicts",
                          new=AsyncMock(return_value=[])), \
             patch.object(orchestrator.agent5, "validate_quantities",
                          new=AsyncMock(return_value=[])), \
             patch.object(orchestrator.rag_pipeline, "synthesize_answer",
                          new=AsyncMock(side_effect=Exception("Groq 503"))):
            r2 = await orchestrator.run_query(FACTUAL_QUERY)
        assert r2 is not None
        assert r2.confidence == "Low"
        print("  ok Groq 503 -> confidence=Low, fallback answer returned")

        # ── Redis down → query still completes ───────────────────────────────
        # _safe_cache_get/_safe_cache_set wrap the cache calls in a circuit
        # breaker and swallow the exception — pipeline must still succeed.
        with patch("backend.utils.cache.cache.get_cached",
                   new=AsyncMock(side_effect=Exception("Redis ECONNREFUSED"))), \
             patch("backend.utils.cache.cache.set_cache",
                   new=AsyncMock(side_effect=Exception("Redis ECONNREFUSED"))), \
             patch.object(orchestrator.agent1, "resolve",
                          new=AsyncMock(return_value=FACTUAL_QUERY)), \
             patch.object(orchestrator.agent2, "is_complex_query",
                          new=AsyncMock(return_value=False)), \
             patch.object(orchestrator.retriever, "retrieve_multi",
                          new=AsyncMock(return_value=MOCK_CHUNKS)), \
             patch.object(orchestrator.agent3, "rerank",
                          new=AsyncMock(return_value=MOCK_CHUNKS)), \
             patch.object(orchestrator.agent4, "detect_conflicts",
                          new=AsyncMock(return_value=[])), \
             patch.object(orchestrator.agent5, "validate_quantities",
                          new=AsyncMock(return_value=[])), \
             patch.object(orchestrator.rag_pipeline, "synthesize_answer",
                          new=AsyncMock(return_value=(MOCK_RAG_ANSWER, "High"))):
            r3 = await orchestrator.run_query(FACTUAL_QUERY, use_cache=True)
        assert r3 is not None
        print("  ok Redis ECONNREFUSED -> pipeline still ran, response returned")

    finally:
        orchestrator._initialized = False


asyncio.run(test_graceful_degradation())


# ─────────────────────────────────────────────────────────────────────
# TEST 15: compare_theses() — parallel execution
# ─────────────────────────────────────────────────────────────────────
print("")
print("=== Test 15: compare_theses() — parallel execution ===")


async def test_compare_theses():
    orchestrator._initialized = True

    THESIS_B = (
        "ICICI Bank will underperform in FY27 because rising credit costs from "
        "unsecured retail lending will compress NIM by 15bps and reduce ROE by "
        "200bps, driven by RBI macroprudential tightening on personal loans."
    )

    try:
        with patch("backend.utils.cache.cache.get_cached",
                   new=AsyncMock(return_value=None)), \
             patch("backend.utils.cache.cache.set_cache",
                   new=AsyncMock()), \
             patch.object(orchestrator.thesis_pipeline, "validate_thesis_input",
                          return_value=(True, "ok")), \
             patch.object(orchestrator.agent1, "resolve",
                          new=AsyncMock(side_effect=lambda q: q)), \
             patch.object(orchestrator.retriever, "retrieve_thesis_context",
                          new=AsyncMock(return_value=MOCK_CHUNKS)), \
             patch.object(orchestrator.agent6, "analyze",
                          new=AsyncMock(return_value=MOCK_THESIS_DATA)), \
             patch.object(orchestrator.agent5, "validate_quantities",
                          new=AsyncMock(return_value=[])), \
             patch.object(orchestrator.agent7, "synthesize",
                          new=AsyncMock(return_value=MOCK_STRESS_DATA)):
            result = await orchestrator.compare_theses(VALID_THESIS, THESIS_B)

        assert "thesis_a" in result
        assert "thesis_b" in result
        assert result["thesis_a"]["thesis_strength"] == "Medium"
        assert result["thesis_b"]["thesis_strength"] == "Medium"
        print("  ok Both theses evaluated in parallel")
        print(f"  ok thesis_a strength: {result['thesis_a']['thesis_strength']}")
        print(f"  ok thesis_b strength: {result['thesis_b']['thesis_strength']}")

    finally:
        orchestrator._initialized = False


asyncio.run(test_compare_theses())


print("")
print("=" * 60)
print("All orchestrator tests passed")
print("=" * 60)