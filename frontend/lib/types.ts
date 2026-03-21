// lib/types.ts

export type ConfidenceLevel = 'High' | 'Medium' | 'Low';

export type AuthorityType =
  | 'rbi'
  | 'sebi'
  | 'mca'
  | 'taxlaw'
  | 'annualreport'
  | 'earningstranscript'
  | 'brokerresearch'
  | 'news'
  | 'blog'
  | 'unknown';

// ─────────────────────────────────────────────────────────────────────────────
// SHARED
// ─────────────────────────────────────────────────────────────────────────────

export interface Citation {
  id:               number;
  title:            string;
  source_type:      string;
  date:             string;
  authority_weight: number;
  relevance_score:  number;
  url?:             string;
  excerpt:          string;
  company?:         string;
  sector?:          string;
}

export interface Conflict {
  field:               string;
  source_a:            string;
  value_a:             string;
  source_b:            string;
  value_b:             string;
  recommended_source:  string;
  reason:              string;
}

// ─────────────────────────────────────────────────────────────────────────────
// RAG
// ─────────────────────────────────────────────────────────────────────────────

export interface QueryRequest {
  query:     string;
  top_k?:    number;
  use_cache?: boolean;
  stream?:   boolean;
  filters?:  Record<string, string>;
}

export interface QueryResponse {
  answer:        string;
  citations:     Citation[];
  confidence:    ConfidenceLevel;
  reasoning:     string;
  conflicts:     Conflict[];
  data_gaps:     string[];
  latency_ms:    number;
  agents_used:   string[];
  cache_hit:     boolean;
  request_id?:   string;
  query_type:    string;
  sub_queries:   string[];
  timestamp:     string;
}

// ─────────────────────────────────────────────────────────────────────────────
// THESIS — building blocks
// ─────────────────────────────────────────────────────────────────────────────

export interface ThesisRequest {
  thesis:        string;
  context?:      string;
  time_horizon?: string;
  asset_class?:  'equity' | 'debt' | 'commodity' | 'currency' | 'other';
  use_cache?:    boolean;
  quick_mode?:   boolean;
}

export interface Assumption {
  id:                  number;
  text:                string;
  category:            string;
  confidence:          number;
  confidence_label:    ConfidenceLevel;
  historical_support:  boolean;
  supporting_evidence: string[];
  dependency_on:       number[];
  is_critical:         boolean;
  risk_flag:           'critical' | 'warning' | 'caution' | 'supported';
}

export interface RiskScore {
  dimension:              string;
  dimension_label:        string;
  score:                  number;
  severity_label:         'Critical' | 'High' | 'Medium' | 'Low';
  severity_color:         'red' | 'orange' | 'yellow' | 'green';
  rationale:              string;
  supporting_data:        string[];
  related_assumption_ids: number[];
}

export interface BreakCondition {
  condition:              string;
  trigger_metric:         string;
  threshold:              string;
  probability:            'High' | 'Medium' | 'Low';
  data_source:            string;
  monitoring_frequency?:  string;
  triggered:              boolean;
  urgency_label:          string;
  urgency_color:          'red' | 'orange' | 'yellow' | 'green';
  related_assumption_ids: number[];
}

export interface DependencyNode {
  id:          number;
  label:       string;
  description: string;
  depends_on:  number[];
  risk_score?: number;
  is_terminal: boolean;
  node_color:  'red' | 'orange' | 'green' | 'gray';
}

export interface HistoricalAnalog {
  title:            string;
  period:           string;
  similarity_score: number;
  similarity_label: string;
  outcome:          string;
  lesson:           string;
  source?:          string;
}

export interface QuantitativeFlag {
  field:        string;
  thesis_claim: string;
  corpus_data:  string;
  discrepancy:  string;
  severity:     'High' | 'Medium' | 'Low';
}

// ─────────────────────────────────────────────────────────────────────────────
// THESIS — main response
// ─────────────────────────────────────────────────────────────────────────────

export interface ThesisResponse {
  thesis_text:             string;
  thesis_strength:         'Strong' | 'Medium' | 'Weak';
  structural_robustness:   'High' | 'Medium' | 'Low';
  confidence:              ConfidenceLevel;
  verdict_summary:         string;
  avg_risk_score:          number;
  assumption_support_rate: number;
  assumptions:             Assumption[];
  dependency_chain:        string[];
  dependency_nodes:        DependencyNode[];
  quantitative_flags:      QuantitativeFlag[];
  risks:                   RiskScore[];
  break_conditions:        BreakCondition[];
  historical_analogs:      HistoricalAnalog[];
  synthesis:               string;
  citations:               Citation[];
  latency_ms:              number;
  cache_hit:               boolean;
  request_id?:             string;
  timestamp:               string;
}

// ─────────────────────────────────────────────────────────────────────────────
// THESIS — comparison (POST /api/thesis/compare)
// ─────────────────────────────────────────────────────────────────────────────

export interface ThesisCompareResponse {
  thesis_a:           ThesisResponse;
  thesis_b:           ThesisResponse;
  /** "thesis_a" | "thesis_b" | "tie" */
  winner:             string;
  comparison_summary: string;
  latency_ms:         number;
  /** Absolute difference in avg_risk_score between the two theses. */
  risk_delta:         number;
}

// ─────────────────────────────────────────────────────────────────────────────
// HEALTH
// ─────────────────────────────────────────────────────────────────────────────

export interface ServiceHealth {
  name:        string;
  status:      'ok' | 'degraded' | 'unavailable' | 'unknown';
  latency_ms?: number;
  version?:    string;
  error?:      string;
  checked_at:  string;
}

export interface HealthResponse {
  status:          'ok' | 'degraded' | 'unavailable' | 'unknown';
  app_name:        string;
  version:         string;
  environment:     string;
  uptime_seconds:  number;
  services:        Record<string, ServiceHealth>;
  corpus_stats?:   Record<string, unknown>;
  cache_stats?:    Record<string, unknown>;
  timestamp:       string;
}

// ─────────────────────────────────────────────────────────────────────────────
// ERRORS
// ─────────────────────────────────────────────────────────────────────────────

export interface ApiError {
  error:       string;
  message:     string;
  status_code: number;
  request_id?: string;
}