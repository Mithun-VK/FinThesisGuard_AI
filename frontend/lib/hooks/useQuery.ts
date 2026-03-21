'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { queryRAG } from '@/lib/api';
import type { QueryRequest, QueryResponse, ConfidenceLevel } from '@/lib/types';

// ── L1 in-memory cache ───────────────────────────────────────────────────────
const L1: Map<string, { data: QueryResponse; expires: number }> = new Map();
const L1_TTL = 5 * 60 * 1000; // 5 min

function cacheKey(p: QueryRequest): string {
  return [
    p.query.trim().toLowerCase(),
    p.top_k ?? 10,
    JSON.stringify(p.filters ?? {}),
  ].join('::');
}

// ── Retry with exponential backoff ───────────────────────────────────────────
async function withRetry<T>(
  fn: () => Promise<T>,
  attempts = 2,
  baseMs = 500,
): Promise<T> {
  let last: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (e) {
      last = e;
      // Do NOT retry on 4xx client errors
      if (
        e &&
        typeof e === 'object' &&
        'status_code' in e &&
        typeof (e as { status_code: number }).status_code === 'number' &&
        (e as { status_code: number }).status_code < 500
      ) {
        throw e;
      }
      if (i < attempts - 1) await new Promise(r => setTimeout(r, baseMs * 2 ** i));
    }
  }
  throw last;
}

// ── Status ───────────────────────────────────────────────────────────────────
export type QueryStatus = 'idle' | 'loading' | 'success' | 'error';

// ── Options ──────────────────────────────────────────────────────────────────
export interface UseQueryOptions {
  useL1Cache?:      boolean;
  useBackendCache?: boolean;
  maxRetries?:      number;
  onSuccess?:       (data: QueryResponse) => void;
  onError?:         (msg: string) => void;
}

// ── Return shape ─────────────────────────────────────────────────────────────
export interface UseQueryReturn {
  // State
  data:          QueryResponse | null;
  status:        QueryStatus;
  error:         string | null;
  latency_ms:    number | null;
  fromCache:     boolean;
  confidence:    ConfidenceLevel | null;
  citationCount: number;
  conflictCount: number;
  // Actions
  submit: (params: QueryRequest) => Promise<void>;
  retry:  () => Promise<void>;
  clear:  () => void;
}

// ── Hook ─────────────────────────────────────────────────────────────────────
export function useQuery(opts: UseQueryOptions = {}): UseQueryReturn {
  const {
    useL1Cache      = true,
    useBackendCache = true,
    maxRetries      = 2,
    onSuccess,
    onError,
  } = opts;

  const [data,       setData]       = useState<QueryResponse | null>(null);
  const [status,     setStatus]     = useState<QueryStatus>('idle');
  const [error,      setError]      = useState<string | null>(null);
  const [latency,    setLatency]    = useState<number | null>(null);
  const [fromCache,  setFromCache]  = useState(false);

  const lastParams = useRef<QueryRequest | null>(null);
  const abort      = useRef<AbortController | null>(null);
  const mounted    = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const submit = useCallback(async (params: QueryRequest) => {
    // Cancel any in-flight fetch
    abort.current?.abort();
    abort.current = new AbortController();
    lastParams.current = params;

    // L1 cache hit
    if (useL1Cache) {
      const hit = L1.get(cacheKey(params));
      if (hit && hit.expires > Date.now()) {
        if (!mounted.current) return;
        setData(hit.data);
        setStatus('success');
        setError(null);
        setLatency(hit.data.latency_ms);
        setFromCache(true);
        onSuccess?.(hit.data);
        return;
      }
    }

    if (!mounted.current) return;
    setStatus('loading');
    setError(null);
    setFromCache(false);
    const t0 = performance.now();

    try {
      const result = await withRetry(
        () => queryRAG({ ...params, use_cache: useBackendCache }),
        maxRetries,
      );
      if (!mounted.current) return;

      if (useL1Cache) {
        L1.set(cacheKey(params), { data: result, expires: Date.now() + L1_TTL });
      }

      setData(result);
      setStatus('success');
      setError(null);
      setLatency(result.latency_ms ?? Math.round(performance.now() - t0));
      setFromCache(result.cache_hit);
      onSuccess?.(result);
    } catch (e: unknown) {
      if (!mounted.current) return;
      const msg =
        e && typeof e === 'object' && 'message' in e
          ? String((e as { message: string }).message)
          : 'Query failed. Please try again.';
      setStatus('error');
      setError(msg);
      setLatency(Math.round(performance.now() - t0));
      onError?.(msg);
    }
  }, [useL1Cache, useBackendCache, maxRetries, onSuccess, onError]);

  const retry = useCallback(async () => {
    if (lastParams.current) await submit(lastParams.current);
  }, [submit]);

  const clear = useCallback(() => {
    abort.current?.abort();
    setData(null);
    setStatus('idle');
    setError(null);
    setLatency(null);
    setFromCache(false);
    lastParams.current = null;
  }, []);

  return {
    data,
    status,
    error,
    latency_ms:    latency,
    fromCache,
    confidence:    data?.confidence    ?? null,
    citationCount: data?.citations?.length ?? 0,
    conflictCount: data?.conflicts?.length ?? 0,
    submit,
    retry,
    clear,
  };
}
