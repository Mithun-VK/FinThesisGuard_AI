/* eslint-disable @typescript-eslint/no-explicit-any */
// lib/api.ts
// ─────────────────────────────────────────────────────────────────────────────
// FinThesisGuard AI — Production API Client
// ArkAngel Financial Solutions
// ─────────────────────────────────────────────────────────────────────────────

'use client';

import type {
  QueryRequest,
  QueryResponse,
  ThesisRequest,
  ThesisResponse,
  ThesisCompareResponse,
  HealthResponse,
  ApiError,
} from './types';

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────────────────

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') ?? 'http://127.0.0.1:8000';

const TIMEOUT_DEFAULT = 35_000;
const TIMEOUT_THESIS  = 65_000;
const TIMEOUT_QUICK   = 10_000;
const TIMEOUT_HEALTH  =  5_000;

const MAX_RETRIES   = 3;
const RETRY_BASE_MS = 500;
const RETRY_MAX_MS  = 8_000;

const CB_FAILURE_THRESHOLD = 5;
const CB_RESET_AFTER_MS    = 30_000;

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Full HTTP-enriched wrapper — carries response metadata alongside body.
 * Used when callers need requestId / latencyMs / cacheHit from headers.
 * Hooks use the .data-unwrapped aliases instead.
 */
export interface ApiResponse<T> {
  data:      T;
  requestId: string | null;
  latencyMs: number | null;
  cacheHit:  boolean;
  headers:   Headers;
}

export interface ThesisHistoryEntry {
  request_id:            string;
  mode:                  string;
  thesis_preview:        string;
  thesis_strength:       string;
  structural_robustness: string;
  confidence:            string;
  avg_risk_score:        number;
  highest_risk:          { dimension: string | null; score: number | null };
  assumption_count:      number;
  critical_assumptions:  number;
  break_condition_count: number;
  high_prob_breaks:      number;
  citation_count:        number;
  cache_hit:             boolean;
  latency_ms:            number;
  agents_used:           string[];
  timestamp_utc:         string;
}

export interface ThesisHistoryResponse {
  total:      number;
  max:        number;
  entries:    ThesisHistoryEntry[];
  fetched_at: string;
}

const NON_RETRYABLE = new Set([400, 401, 403, 404, 409, 422]);

// ─────────────────────────────────────────────────────────────────────────────
// CIRCUIT BREAKER
// ─────────────────────────────────────────────────────────────────────────────

type CbState = 'CLOSED' | 'OPEN' | 'HALF_OPEN';

const _cb = {
  state:    'CLOSED' as CbState,
  failures: 0,
  openedAt: 0,

  record(success: boolean): void {
    if (success) { this.failures = 0; this.state = 'CLOSED'; return; }
    this.failures++;
    if (this.failures >= CB_FAILURE_THRESHOLD) {
      this.state = 'OPEN'; this.openedAt = Date.now();
    }
  },

  allow(): boolean {
    if (this.state === 'CLOSED' || this.state === 'HALF_OPEN') return true;
    if (Date.now() - this.openedAt >= CB_RESET_AFTER_MS) {
      this.state = 'HALF_OPEN'; return true;
    }
    return false;
  },

  reset(): void { this.state = 'CLOSED'; this.failures = 0; this.openedAt = 0; },
};

export function resetCircuitBreaker(): void { _cb.reset(); }
export function circuitBreakerState(): CbState { return _cb.state; }

// ─────────────────────────────────────────────────────────────────────────────
// IN-FLIGHT DEDUPLICATION
// ─────────────────────────────────────────────────────────────────────────────

const _pending     = new Map<string, Promise<any>>();
const _controllers = new Map<string, AbortController>();

function _makeKey(path: string, body: unknown): string {
  try {
    const raw = JSON.stringify({ path, body: body ?? {} });
    let h = 5381;
    for (let i = 0; i < raw.length; i++) h = ((h << 5) + h) ^ raw.charCodeAt(i);
    return `${path}:${(h >>> 0).toString(16)}`;
  } catch {
    return `${path}:${Math.random().toString(36).slice(2)}`;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// FETCH PRIMITIVE
// ─────────────────────────────────────────────────────────────────────────────

async function _fetchWithTimeout(
  url: string, options: RequestInit, timeoutMs: number, key: string,
): Promise<Response> {
  const controller = new AbortController();
  _controllers.set(key, controller);

  const callerSignal = options.signal as AbortSignal | undefined;
  if (callerSignal) callerSignal.addEventListener('abort', () => controller.abort());

  const tid = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(tid);
    _controllers.delete(key);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CORE REQUEST  — returns ApiResponse<T> (HTTP metadata + body)
// ─────────────────────────────────────────────────────────────────────────────

async function _request<T>(
  path:      string,
  options:   RequestInit = {},
  timeoutMs: number      = TIMEOUT_DEFAULT,
): Promise<ApiResponse<T>> {
  if (!_cb.allow()) {
    throw _makeErr(
      'circuit_open',
      `API circuit breaker is OPEN after ${CB_FAILURE_THRESHOLD} failures. ` +
      `Auto-resets in ${CB_RESET_AFTER_MS / 1000}s.`,
      503,
    );
  }

  const url = `${BASE_URL}${path}`;
  const key = _makeKey(path, options.body);
  if (_pending.has(key)) return _pending.get(key)!;

  const execute = async (): Promise<ApiResponse<T>> => {
    let lastError: unknown;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        const res = await _fetchWithTimeout(
          url,
          {
            ...options,
            headers: {
              'Content-Type': 'application/json',
              Accept:         'application/json',
              ...(options.headers as Record<string, string>),
            },
          },
          timeoutMs,
          key,
        );

        if (!res.ok) {
          let errBody: ApiError;
          try {
            const raw = await res.json();
            errBody = raw?.detail ?? raw;
          } catch {
            errBody = _makeErr('http_error', `HTTP ${res.status} ${res.statusText}`, res.status);
          }
          errBody.status_code ??= res.status;

          if (NON_RETRYABLE.has(res.status)) { _cb.record(false); throw errBody; }

          lastError = errBody;
          if (attempt < MAX_RETRIES) {
            await _sleep(Math.min(RETRY_BASE_MS * 2 ** attempt, RETRY_MAX_MS));
            continue;
          }
          _cb.record(false);
          throw lastError;
        }

        const data = (await res.json()) as T;
        _cb.record(true);

        return {
          data,
          requestId: res.headers.get('X-Request-Id'),
          latencyMs: _parseMs(res.headers.get('X-Response-Time')),
          cacheHit:  res.headers.get('X-Cache') === 'HIT',
          headers:   res.headers,
        };

      } catch (e: any) {
        if (e?.name === 'AbortError') {
          _cb.record(false);
          throw _makeErr('timeout', `Request to ${path} timed out after ${timeoutMs}ms`, 408);
        }
        if (_isApiErrorShape(e)) throw e;

        lastError = e;
        if (attempt < MAX_RETRIES) await _sleep(Math.min(RETRY_BASE_MS * 2 ** attempt, RETRY_MAX_MS));
      }
    }

    _cb.record(false);
    throw lastError;
  };

  const promise = execute().finally(() => _pending.delete(key));
  _pending.set(key, promise);
  return promise;
}

// ─────────────────────────────────────────────────────────────────────────────
// PUBLIC api OBJECT  — returns ApiResponse<T> (for callers that need headers)
// ─────────────────────────────────────────────────────────────────────────────

export const api = {

  query(body: QueryRequest): Promise<ApiResponse<QueryResponse>> {
    return _request<QueryResponse>('/api/query', { method: 'POST', body: JSON.stringify(body) });
  },

  validateThesis(body: ThesisRequest): Promise<ApiResponse<ThesisResponse>> {
    return _request<ThesisResponse>('/api/validate-thesis', {
      method: 'POST', body: JSON.stringify(body),
    }, TIMEOUT_THESIS);
  },

  validateThesisQuick(body: ThesisRequest): Promise<ApiResponse<ThesisResponse>> {
    return _request<ThesisResponse>('/api/validate-thesis/quick', {
      method: 'POST', body: JSON.stringify({ ...body, quick_mode: true }),
    }, TIMEOUT_QUICK);
  },

  compareTheses(thesis_a: string, thesis_b: string): Promise<ApiResponse<ThesisCompareResponse>> {
    return _request<ThesisCompareResponse>('/api/thesis/compare', {
      method: 'POST', body: JSON.stringify({ thesis_a, thesis_b }),
    }, TIMEOUT_THESIS);
  },

  thesisHistory(): Promise<ApiResponse<ThesisHistoryResponse>> {
    return _request<ThesisHistoryResponse>('/api/thesis/history', { method: 'GET' }, TIMEOUT_HEALTH);
  },

  thesisById(requestId: string): Promise<ApiResponse<{ request_id: string; summary: ThesisHistoryEntry }>> {
    return _request(`/api/thesis/${encodeURIComponent(requestId)}`, { method: 'GET' }, TIMEOUT_HEALTH);
  },

  health(): Promise<ApiResponse<HealthResponse>> {
    return _request<HealthResponse>('/health', { method: 'GET' }, TIMEOUT_HEALTH);
  },

  metrics(): Promise<ApiResponse<Record<string, unknown>>> {
    return _request<Record<string, unknown>>('/api/metrics', { method: 'GET' }, TIMEOUT_HEALTH);
  },

  clearRagCache(): Promise<ApiResponse<Record<string, unknown>>> {
    return _request('/api/cache/clear', { method: 'POST' }, TIMEOUT_HEALTH);
  },

  clearThesisCache(): Promise<ApiResponse<Record<string, unknown>>> {
    return _request('/api/thesis/cache/flush', { method: 'POST' }, TIMEOUT_HEALTH);
  },

  clearThesisHistory(): Promise<ApiResponse<Record<string, unknown>>> {
    return _request('/api/thesis/history', { method: 'DELETE' }, TIMEOUT_HEALTH);
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// STREAMING QUERY  (SSE)
// ─────────────────────────────────────────────────────────────────────────────

export async function* streamQuery(
  body: QueryRequest, signal?: AbortSignal,
): AsyncGenerator<string, void, undefined> {
  const key = _makeKey('/api/query/stream', body);
  if (_pending.has(key)) {
    throw _makeErr('stream_conflict', 'A streaming request for this query is already in progress.', 409);
  }

  const controller = new AbortController();
  _controllers.set(key, controller);
  if (signal) signal.addEventListener('abort', () => controller.abort());
  const tid = setTimeout(() => controller.abort(), TIMEOUT_DEFAULT);

  try {
    const res = await fetch(`${BASE_URL}/api/query`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body:    JSON.stringify({ ...body, stream: true }),
      signal:  controller.signal,
    });

    if (!res.ok || !res.body) throw _makeErr('stream_failed', `Stream failed: HTTP ${res.status}`, res.status);

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();
          if (payload && payload !== '[DONE]') yield payload;
        }
      }
      if (buffer.startsWith('data: ')) {
        const payload = buffer.slice(6).trim();
        if (payload && payload !== '[DONE]') yield payload;
      }
    } finally {
      reader.releaseLock();
    }

  } catch (e: any) {
    if (e?.name === 'AbortError')
      throw _makeErr('timeout', `Streaming query timed out after ${TIMEOUT_DEFAULT}ms`, 408);
    throw e;
  } finally {
    clearTimeout(tid);
    _controllers.delete(key);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// BATCH QUERY
// ─────────────────────────────────────────────────────────────────────────────

export function batchQuery(queries: QueryRequest[]): Promise<ApiResponse<QueryResponse[]>> {
  if (queries.length === 0) return Promise.resolve(_emptyResponse<QueryResponse[]>([]));
  return _request<QueryResponse[]>('/api/query/batch', {
    method: 'POST', body: JSON.stringify(queries.slice(0, 10)),
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// POLLING HELPERS
// ─────────────────────────────────────────────────────────────────────────────

export function pollHealth(onResult: (data: HealthResponse) => void, intervalMs = 15_000): () => void {
  let alive = true;
  const tick = async () => {
    try { const { data } = await api.health(); if (alive) onResult(data); } catch { /* ignore */ }
    if (alive) setTimeout(tick, intervalMs);
  };
  tick();
  return () => { alive = false; };
}

export function pollMetrics(onResult: (data: Record<string, unknown>) => void, intervalMs = 30_000): () => void {
  let alive = true;
  const tick = async () => {
    try { const { data } = await api.metrics(); if (alive) onResult(data); } catch { /* ignore */ }
    if (alive) setTimeout(tick, intervalMs);
  };
  tick();
  return () => { alive = false; };
}

// ─────────────────────────────────────────────────────────────────────────────
// TYPE GUARDS
// ─────────────────────────────────────────────────────────────────────────────

export function isApiError(e: unknown): e is ApiError {
  return typeof e === 'object' && e !== null &&
    'error' in e && typeof (e as any).error === 'string';
}
export function isTimeoutError(e: unknown): boolean {
  return isApiError(e) && (e.error === 'timeout' || e.status_code === 408);
}
export function isRateLimitError(e: unknown): boolean {
  return isApiError(e) && e.status_code === 429;
}
export function isValidationError(e: unknown): boolean {
  return isApiError(e) && (e.status_code === 400 || e.error === 'thesis_validation_failed');
}
export function isServiceUnavailable(e: unknown): boolean {
  return isApiError(e) && e.status_code === 503;
}
export function getErrorMessage(e: unknown): string {
  if (isApiError(e))         return e.message ?? e.error;
  if (e instanceof Error)    return e.message;
  if (typeof e === 'string') return e;
  return 'An unexpected error occurred.';
}

// ─────────────────────────────────────────────────────────────────────────────
// CANCELLATION
// ─────────────────────────────────────────────────────────────────────────────

export function cancelAllRequests(): void {
  _controllers.forEach(c => c.abort());
  _controllers.clear();
  _pending.clear();
}

export function cancelRequest(path: string, body?: unknown): void {
  const key = _makeKey(path, body);
  _controllers.get(key)?.abort();
  _controllers.delete(key);
  _pending.delete(key);
}

// ─────────────────────────────────────────────────────────────────────────────
// CONVENIENCE ALIASES  ← THE FIX IS HERE
// All hooks (useQuery, useThesis, useHealth) call these.
// They unwrap .data so the hook receives QueryResponse / ThesisResponse /
// HealthResponse directly — matching types.ts exactly.
// ─────────────────────────────────────────────────────────────────────────────

/** Returns QueryResponse directly — used by useQuery hook */
export const queryRAG = (body: QueryRequest): Promise<QueryResponse> =>
  api.query(body).then(r => r.data);

/** Returns ThesisResponse directly — used by useThesis hook */
export const validateThesis = (body: ThesisRequest): Promise<ThesisResponse> =>
  api.validateThesis(body).then(r => r.data);

/** Returns ThesisResponse directly — quick mode */
export const quickThesis = (body: ThesisRequest): Promise<ThesisResponse> =>
  api.validateThesisQuick(body).then(r => r.data);

/** Returns ThesisCompareResponse directly */
export const compareTheses = (a: string, b: string): Promise<ThesisCompareResponse> =>
  api.compareTheses(a, b).then(r => r.data);

/** Returns HealthResponse directly — used by useHealth hook */
export const getHealth = (): Promise<HealthResponse> =>
  api.health().then(r => r.data);

// ─────────────────────────────────────────────────────────────────────────────
// PRIVATE UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

function _sleep(ms: number): Promise<void> { return new Promise(r => setTimeout(r, ms)); }

function _parseMs(header: string | null): number | null {
  if (!header) return null;
  const n = parseInt(header.replace('ms', '').trim(), 10);
  return isNaN(n) ? null : n;
}

function _makeErr(error: string, message: string, status_code: number): ApiError {
  return { error, message, status_code };
}

function _isApiErrorShape(e: unknown): e is ApiError {
  return typeof e === 'object' && e !== null &&
    'error' in e && 'message' in e && 'status_code' in e;
}

function _emptyResponse<T>(data: T): ApiResponse<T> {
  return { data, requestId: null, latencyMs: null, cacheHit: false, headers: new Headers() };
}
