'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { validateThesis } from '@/lib/api';
import type {
  ThesisRequest,
  ThesisResponse,
  ConfidenceLevel,
  RiskScore,
} from '@/lib/types';

// ── L1 in-memory cache ───────────────────────────────────────────────────────
const L1: Map<string, { data: ThesisResponse; expires: number }> = new Map();
const L1_TTL = 10 * 60 * 1000; // 10 min (thesis is expensive)

function cacheKey(p: ThesisRequest): string {
  return [
    p.thesis.trim().toLowerCase(),
    p.asset_class  ?? 'any',
    p.time_horizon ?? 'any',
    p.quick_mode   ?? false,
  ].join('::');
}

// ── Stage animation ──────────────────────────────────────────────────────────
export const THESIS_STAGES = [
  { id: 'resolving',    label: 'Resolving financial terms...',              durationMs: 600  },
  { id: 'retrieving',   label: 'Retrieving historical context...',          durationMs: 1200 },
  { id: 'extracting',   label: 'Extracting assumptions & dependencies...',  durationMs: 1800 },
  { id: 'validating',   label: 'Validating quantitative claims...',         durationMs: 1200 },
  { id: 'scoring',      label: 'Scoring risk dimensions (1–10)...',         durationMs: 1200 },
  { id: 'synthesizing', label: 'Synthesizing final verdict...',             durationMs: 800  },
] as const;

export type ThesisStageId = (typeof THESIS_STAGES)[number]['id'];

// ── Status ───────────────────────────────────────────────────────────────────
export type ThesisStatus = 'idle' | 'loading' | 'success' | 'error';

// ── Options ──────────────────────────────────────────────────────────────────
export interface UseThesisOptions {
  useL1Cache?:      boolean;
  useBackendCache?: boolean;
  onStageChange?:   (stageId: ThesisStageId, label: string, progress: number) => void;
  onSuccess?:       (data: ThesisResponse) => void;
  onError?:         (msg: string) => void;
}

// ── Return shape ─────────────────────────────────────────────────────────────
export interface UseThesisReturn {
  // State
  data:                ThesisResponse | null;
  status:              ThesisStatus;
  error:               string | null;
  latency_ms:          number | null;
  fromCache:           boolean;
  currentStage:        string | null;
  currentStageId:      ThesisStageId | null;
  progress:            number; // 0–100
  // Derived from ThesisResponse — null until success
  confidence:          ConfidenceLevel | null;
  thesisStrength:      ThesisResponse['thesis_strength'] | null;
  avgRiskScore:        number | null;
  highestRisk:         RiskScore | null;
  assumptionCount:     number;
  criticalCount:       number;
  triggeredBreaks:     number;
  // Actions
  submit: (params: ThesisRequest) => Promise<void>;
  retry:  () => Promise<void>;
  clear:  () => void;
}

// ── Hook ─────────────────────────────────────────────────────────────────────
export function useThesis(opts: UseThesisOptions = {}): UseThesisReturn {
  const {
    useL1Cache      = true,
    useBackendCache = true,
    onStageChange,
    onSuccess,
    onError,
  } = opts;

  const [data,           setData]           = useState<ThesisResponse | null>(null);
  const [status,         setStatus]         = useState<ThesisStatus>('idle');
  const [error,          setError]          = useState<string | null>(null);
  const [latency,        setLatency]        = useState<number | null>(null);
  const [fromCache,      setFromCache]      = useState(false);
  const [currentStage,   setCurrentStage]   = useState<string | null>(null);
  const [currentStageId, setCurrentStageId] = useState<ThesisStageId | null>(null);
  const [progress,       setProgress]       = useState(0);

  const lastParams = useRef<ThesisRequest | null>(null);
  const timers     = useRef<ReturnType<typeof setTimeout>[]>([]);
  const mounted    = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      timers.current.forEach(clearTimeout);
    };
  }, []);

  // Simulate stage progression while real request is in-flight
  const startStages = useCallback((quick: boolean) => {
    timers.current.forEach(clearTimeout);
    timers.current = [];

    // Scale durations if quick_mode
    const scale = quick ? 0.4 : 1;
    let elapsed = 0;

    THESIS_STAGES.forEach((stage, i) => {
      const delay = elapsed;
      const t = setTimeout(() => {
        if (!mounted.current) return;
        const pct = Math.round(((i + 1) / THESIS_STAGES.length) * 90); // 90% max until done
        setCurrentStage(stage.label);
        setCurrentStageId(stage.id);
        setProgress(pct);
        onStageChange?.(stage.id, stage.label, pct);
      }, delay);

      timers.current.push(t);
      elapsed += Math.round(stage.durationMs * scale);
    });
  }, [onStageChange]);

  const stopStages = useCallback(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }, []);

  const submit = useCallback(async (params: ThesisRequest) => {
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
        setProgress(100);
        setCurrentStage(null);
        setCurrentStageId(null);
        onSuccess?.(hit.data);
        return;
      }
    }

    if (!mounted.current) return;
    setStatus('loading');
    setError(null);
    setFromCache(false);
    setData(null);
    setProgress(0);
    setCurrentStage(THESIS_STAGES[0].label);
    setCurrentStageId(THESIS_STAGES[0].id);

    startStages(params.quick_mode ?? false);
    const t0 = performance.now();

    try {
      const result = await validateThesis({
        ...params,
        use_cache: useBackendCache,
      });
      if (!mounted.current) return;
      stopStages();

      if (useL1Cache) {
        L1.set(cacheKey(params), { data: result, expires: Date.now() + L1_TTL });
      }

      setData(result);
      setStatus('success');
      setError(null);
      setLatency(result.latency_ms ?? Math.round(performance.now() - t0));
      setFromCache(result.cache_hit);
      setProgress(100);
      setCurrentStage(null);
      setCurrentStageId(null);
      onSuccess?.(result);
    } catch (e: unknown) {
      if (!mounted.current) return;
      stopStages();

      const msg =
        e && typeof e === 'object' && 'message' in e
          ? String((e as { message: string }).message)
          : 'Thesis validation failed. Please try again.';

      setStatus('error');
      setError(msg);
      setLatency(Math.round(performance.now() - t0));
      setProgress(0);
      setCurrentStage(null);
      setCurrentStageId(null);
      onError?.(msg);
    }
  }, [useL1Cache, useBackendCache, startStages, stopStages, onSuccess, onError]);

  const retry = useCallback(async () => {
    if (lastParams.current) await submit(lastParams.current);
  }, [submit]);

  const clear = useCallback(() => {
    stopStages();
    setData(null);
    setStatus('idle');
    setError(null);
    setLatency(null);
    setFromCache(false);
    setProgress(0);
    setCurrentStage(null);
    setCurrentStageId(null);
    lastParams.current = null;
  }, [stopStages]);

  // Derived fields — computed fresh each render from data
  const highestRisk = data?.risks?.length
    ? [...data.risks].sort((a, b) => b.score - a.score)[0]
    : null;

  return {
    data,
    status,
    error,
    latency_ms:      latency,
    fromCache,
    currentStage,
    currentStageId,
    progress,
    confidence:      data?.confidence          ?? null,
    thesisStrength:  data?.thesis_strength     ?? null,
    avgRiskScore:    data?.avg_risk_score       ?? null,
    highestRisk,
    assumptionCount: data?.assumptions?.length  ?? 0,
    criticalCount:   data?.assumptions?.filter(a => a.is_critical).length ?? 0,
    triggeredBreaks: data?.break_conditions?.filter(b => b.triggered).length ?? 0,
    submit,
    retry,
    clear,
  };
}
