'use client';

import { useState, useCallback } from 'react';
import ThesisInput from '@/components/thesis/ThesisInput';
import ThesisSummaryCard from '@/components/thesis/ThesisSummaryCard';
import AssumptionTree from '@/components/thesis/AssumptionTree';
import DependencyChain from '@/components/thesis/DependencyChain';
import RiskScoreCard from '@/components/thesis/RiskScoreCard';
import BreakConditionList from '@/components/thesis/BreakConditionList';
import CitationPanel from '@/components/rag/CitationPanel';
import LoadingSpinner from '@/components/shared/LoadingSpinner';
import { validateThesis, quickThesis, getErrorMessage, isValidationError, isRateLimitError, isServiceUnavailable } from '@/lib/api';
import type { ThesisResponse, ThesisRequest } from '@/lib/types';

// ── Example theses shown in empty state ──────────────────────────────────────
const EXAMPLE_THESES = [
  'HDFC Bank will outperform the sector in FY27 because NIM expansion of 20bps is expected as RBI cuts repo rate by 75bps, driving PAT growth of 18-20% and re-rating to 3.2x book value.',
  'Reliance Industries is overvalued at current levels because Jio\'s ARPU growth is plateauing while capital intensity remains high, compressing FCF yield below 3%.',
  'TCS will underperform over 12 months because enterprise IT discretionary spend is declining due to macro uncertainty, driven by client budget freezes across BFSI and retail verticals.',
];

// ── Map API errors → user-friendly messages ───────────────────────────────────
function toUserMessage(e: unknown): string {
  if (isValidationError(e)) {
    const msg = getErrorMessage(e);
    // Surface the specific validation reason from the backend
    if (msg.includes('Thesis validation failed —')) {
      return msg.replace('Thesis validation failed — ', '');
    }
    return 'Your thesis needs a subject, a directional claim, and a reason (because / driven by / due to).';
  }
  if (isRateLimitError(e))     return 'Too many requests. Please wait a moment and try again.';
  if (isServiceUnavailable(e)) return 'The analysis pipeline is initializing. Please retry in a few seconds.';
  const msg = getErrorMessage(e);
  if (msg.toLowerCase().includes('timeout'))
    return 'Thesis validation timed out. Try the Quick Mode toggle for a faster result.';
  return msg || 'An unexpected error occurred. Please try again.';
}

// ── Loading step labels — shown sequentially while pipeline runs ───────────────
const FULL_STEPS = [
  'Expanding acronyms and enriching thesis text…',
  'Retrieving historical analogs from corpus…',
  'Extracting assumptions and building dependency chain…',
  'Validating quantitative claims…',
  'Scoring 6 risk dimensions…',
  'Identifying break conditions…',
  'Synthesizing verdict…',
];

const QUICK_STEPS = [
  'Extracting assumptions and building dependency chain…',
  'Scoring risk dimensions…',
  'Synthesizing verdict…',
];

export default function ThesisPage() {
  const [response, setResponse]       = useState<ThesisResponse | null>(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [currentThesis, setCurrentThesis] = useState('');
  const [quickMode, setQuickMode]     = useState(false);
  const [stepIndex, setStepIndex]     = useState(0);

  // ── Animated loading step cycle ────────────────────────────────────────────
  const runSteps = useCallback((steps: string[]) => {
    setStepIndex(0);
    let i = 0;
    const interval = setInterval(() => {
      i = Math.min(i + 1, steps.length - 1);
      setStepIndex(i);
      if (i >= steps.length - 1) clearInterval(interval);
    }, quickMode ? 900 : 1400);
    return interval;
  }, [quickMode]);

  // ── Core submit handler ────────────────────────────────────────────────────
  const handleSubmit = useCallback(async (thesis: string) => {
    if (!thesis.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResponse(null);
    setCurrentThesis(thesis.trim());

    const steps   = quickMode ? QUICK_STEPS : FULL_STEPS;
    const interval = runSteps(steps);

    const payload: ThesisRequest = {
      thesis:     thesis.trim(),
      use_cache:  true,
      quick_mode: quickMode,
    };

    try {
      // validateThesis and quickThesis both return ThesisResponse directly
      const result = quickMode
        ? await quickThesis(payload)
        : await validateThesis(payload);
      setResponse(result);
    } catch (e) {
      setError(toUserMessage(e));
    } finally {
      clearInterval(interval);
      setLoading(false);
    }
  }, [loading, quickMode, runSteps]);

  // ── Derived state ──────────────────────────────────────────────────────────
  const hasCitations       = (response?.citations?.length ?? 0) > 0;
  const hasAnalogs         = (response?.historical_analogs?.length ?? 0) > 0;
  const hasBreakConditions = (response?.break_conditions?.length ?? 0) > 0;
  const hasRisks           = (response?.risks?.length ?? 0) > 0;
  const hasAssumptions     = (response?.assumptions?.length ?? 0) > 0;
  const hasDependencyChain = (response?.dependency_chain?.length ?? 0) > 0;
  const showResults        = response !== null && !loading;
  const steps              = quickMode ? QUICK_STEPS : FULL_STEPS;

  // ── Export & Share ─────────────────────────────────────────────────────────
  const handleExport = useCallback(() => {
    if (!response) return;
    // Placeholder — wire jsPDF here
    alert(`PDF Export — Thesis Strength: ${response.thesis_strength} | Confidence: ${response.confidence}`);
  }, [response]);

  const handleShare = useCallback(() => {
    if (typeof window === 'undefined' || !currentThesis) return;
    const url = `${window.location.origin}/thesis?q=${encodeURIComponent(currentThesis.slice(0, 200))}`;
    navigator.clipboard.writeText(url).then(() => {
      // Optionally show a toast here
    });
  }, [currentThesis]);

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '48px 40px' }}>

      {/* ── Page header ──────────────────────────────────────────────────────── */}
      <div style={{ marginBottom: '36px' }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: '8px',
          marginBottom: '12px', background: 'rgba(230, 57, 70, 0.1)',
          border: '1px solid rgba(230, 57, 70, 0.3)', borderRadius: '9999px',
          padding: '6px 16px',
        }}>
          <span style={{ fontSize: '12px', color: '#E63946', fontWeight: 600 }}>
            ⚡ Thesis Validator
          </span>
        </div>

        <h1 style={{
          fontSize: 'clamp(28px, 4vw, 48px)', fontWeight: 800, color: '#F1F1F1',
          margin: '0 0 12px', fontFamily: "'Space Grotesk', sans-serif", lineHeight: '1.15',
        }}>
          Stress-Test Your{' '}
          <span className="gradient-text">Investment Thesis</span>
        </h1>

        <p style={{ fontSize: '16px', color: '#A0A0A0', maxWidth: '560px', lineHeight: '1.6', margin: 0 }}>
          AI-powered decomposition with assumption analysis, dependency mapping,
          risk scoring, and break condition identification.
        </p>
      </div>

      {/* ── Input + Quick Mode toggle ─────────────────────────────────────────── */}
      <div style={{ marginBottom: '32px' }}>
        <ThesisInput onSubmit={handleSubmit} loading={loading} />

        {/* Quick Mode toggle row */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: '12px',
          marginTop: '16px', padding: '12px 16px',
          background: quickMode ? 'rgba(230,57,70,0.07)' : 'rgba(255,255,255,0.03)',
          border: `1px solid ${quickMode ? 'rgba(230,57,70,0.3)' : 'rgba(255,255,255,0.08)'}`,
          borderRadius: '10px', transition: 'all 0.25s ease',
        }}>
          {/* Toggle switch */}
          <button
            onClick={() => setQuickMode(v => !v)}
            disabled={loading}
            aria-pressed={quickMode}
            aria-label="Toggle quick mode"
            style={{
              position: 'relative', width: '40px', height: '22px',
              borderRadius: '11px', border: 'none', cursor: loading ? 'not-allowed' : 'pointer',
              background: quickMode ? '#E63946' : 'rgba(255,255,255,0.15)',
              transition: 'background 0.2s ease', flexShrink: 0, padding: 0,
            }}
          >
            <span style={{
              position: 'absolute', top: '3px',
              left: quickMode ? '21px' : '3px',
              width: '16px', height: '16px', borderRadius: '50%',
              background: '#fff', transition: 'left 0.2s ease', display: 'block',
            }} />
          </button>

          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '14px', fontWeight: 600, color: quickMode ? '#F1F1F1' : '#A0A0A0' }}>
                Quick Mode
              </span>
              {quickMode && (
                <span style={{
                  fontSize: '11px', fontWeight: 600, color: '#E63946',
                  background: 'rgba(230,57,70,0.15)', padding: '2px 8px',
                  borderRadius: '9999px', border: '1px solid rgba(230,57,70,0.3)',
                }}>
                  ~40% faster
                </span>
              )}
            </div>
            <p style={{ fontSize: '12px', color: '#6B6B6B', margin: '2px 0 0' }}>
              {quickMode
                ? 'Skips corpus retrieval and quantitative validation (Agent 5). Confidence capped at Medium.'
                : 'Full 7-agent pipeline: corpus retrieval, quant validation, historical analogs. Higher accuracy.'}
            </p>
          </div>

          {/* Mode pill */}
          <div style={{
            fontSize: '11px', padding: '4px 10px', borderRadius: '6px',
            background: quickMode ? 'rgba(234,179,8,0.15)' : 'rgba(34,197,94,0.12)',
            color: quickMode ? '#EAB308' : '#22C55E',
            border: `1px solid ${quickMode ? 'rgba(234,179,8,0.3)' : 'rgba(34,197,94,0.25)'}`,
            fontWeight: 600, whiteSpace: 'nowrap',
          }}>
            {quickMode ? '⚡ 2 agents' : '🔬 7 agents'}
          </div>
        </div>
      </div>

      {/* ── Error ────────────────────────────────────────────────────────────── */}
      {error && !loading && (
        <div style={{
          marginBottom: '24px', padding: '16px 20px',
          background: 'rgba(230, 57, 70, 0.08)',
          border: '1px solid rgba(230, 57, 70, 0.35)',
          borderRadius: '12px', display: 'flex', alignItems: 'flex-start', gap: '12px',
        }}>
          <span style={{ fontSize: '18px', marginTop: '1px' }}>⚠️</span>
          <div>
            <p style={{ color: '#E63946', fontSize: '14px', fontWeight: 600, margin: '0 0 4px' }}>
              Validation failed
            </p>
            <p style={{ color: '#C08080', fontSize: '13px', margin: 0, lineHeight: '1.5' }}>
              {error}
            </p>
            {error.includes('subject') || error.includes('claim') || error.includes('reason') ? (
              <p style={{ color: '#6B6B6B', fontSize: '12px', margin: '8px 0 0', fontStyle: 'italic' }}>
                Example: &quot;HDFC Bank will outperform because NIM expansion of 20bps is expected
                as RBI cuts rates by 75bps over FY26.&quot;
              </p>
            ) : null}
          </div>
        </div>
      )}

      {/* ── Loading ───────────────────────────────────────────────────────────── */}
      {loading && (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          padding: '60px 20px', gap: '24px',
        }}>
          <LoadingSpinner variant="dots" size="lg" text={steps[stepIndex]} />

          {/* Step progress pills */}
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', justifyContent: 'center', maxWidth: '500px' }}>
            {steps.map((step, i) => (
              <div key={i} style={{
                fontSize: '11px', padding: '3px 10px', borderRadius: '9999px',
                background: i <= stepIndex ? 'rgba(230,57,70,0.15)' : 'rgba(255,255,255,0.05)',
                border: `1px solid ${i <= stepIndex ? 'rgba(230,57,70,0.4)' : 'rgba(255,255,255,0.08)'}`,
                color: i <= stepIndex ? '#E63946' : '#4B4B4B',
                transition: 'all 0.3s ease', fontWeight: i === stepIndex ? 600 : 400,
              }}>
                {i < stepIndex ? '✓' : i === stepIndex ? '◆' : '○'} Step {i + 1}
              </div>
            ))}
          </div>

          {quickMode && (
            <p style={{ color: '#6B6B6B', fontSize: '12px', margin: 0 }}>
              Quick Mode active — skipping corpus retrieval and Agent 5
            </p>
          )}
        </div>
      )}

      {/* ── Results — ordered per Phase 5 spec ────────────────────────────────── */}
      {showResults && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

          {/* 1 — Summary Card (verdict header — most important, always first) */}
          <div style={{ animation: 'revealSection 0.4s ease forwards' }}>
            <ThesisSummaryCard
              response={response}
              thesisText={currentThesis}
              onExport={handleExport}
              onShare={handleShare}
            />
            {quickMode && (
              <div style={{
                marginTop: '8px', padding: '8px 14px', borderRadius: '8px',
                background: 'rgba(234,179,8,0.08)', border: '1px solid rgba(234,179,8,0.25)',
                fontSize: '12px', color: '#EAB308', display: 'flex', alignItems: 'center', gap: '6px',
              }}>
                <span>⚡</span>
                <span>
                  Quick Mode result — confidence capped at Medium. Run in Full Mode for corpus-backed analysis.
                </span>
              </div>
            )}
          </div>

          {/* 2 — Dependency Chain (structural logic map) */}
          {hasDependencyChain && (
            <div style={{ animation: 'revealSection 0.4s 0.1s ease forwards', opacity: 0 }}>
              <DependencyChain
                chain={response.dependency_chain}
                nodes={response.dependency_nodes}
                risks={response.risks}
              />
            </div>
          )}

          {/* 3 — Assumption Tree */}
          {hasAssumptions && (
            <div style={{ animation: 'revealSection 0.4s 0.2s ease forwards', opacity: 0 }}>
              <AssumptionTree assumptions={response.assumptions} />
            </div>
          )}

          {/* 4 — Risk Score Cards grid */}
          {hasRisks && (
            <div style={{ animation: 'revealSection 0.4s 0.3s ease forwards', opacity: 0 }}>
              <div style={{ marginBottom: '12px' }}>
                <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#F1F1F1', margin: '0 0 4px' }}>
                  Risk Dimensions
                </h3>
                <p style={{ fontSize: '13px', color: '#6B6B6B', margin: 0 }}>
                  Scored 1–10 across {response.risks.length} dimensions. Higher = greater risk.
                </p>
              </div>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '14px',
              }}>
                {response.risks.map(risk => (
                  <RiskScoreCard key={risk.dimension} risk={risk} />
                ))}
              </div>
            </div>
          )}

          {/* 5 — Break Conditions */}
          {hasBreakConditions && (
            <div style={{ animation: 'revealSection 0.4s 0.4s ease forwards', opacity: 0 }}>
              <BreakConditionList conditions={response.break_conditions} />
            </div>
          )}

          {/* 6 — Historical Analogs */}
          {hasAnalogs && (
            <div style={{ animation: 'revealSection 0.4s 0.5s ease forwards', opacity: 0 }}>
              <div style={{
                background: '#111', border: '1px solid #222',
                borderRadius: '12px', padding: '20px',
              }}>
                <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#F1F1F1', margin: '0 0 16px' }}>
                  📚 Historical Analogs
                </h3>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                  gap: '12px',
                }}>
                  {response.historical_analogs.map((analog, i) => (
                    <div key={i} style={{
                      padding: '14px 16px', background: '#0A0A0A',
                      border: '1px solid #1E1E1E', borderRadius: '10px',
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '6px' }}>
                        <span style={{ fontSize: '13px', fontWeight: 600, color: '#F1F1F1' }}>
                          {analog.title}
                        </span>
                        <span style={{
                          fontSize: '11px', color: '#6B6B6B', padding: '2px 8px',
                          background: 'rgba(255,255,255,0.05)', borderRadius: '4px',
                          whiteSpace: 'nowrap', marginLeft: '8px',
                        }}>
                          {analog.period}
                        </span>
                      </div>
                      <div style={{
                        fontSize: '11px', fontWeight: 600, marginBottom: '6px',
                        color: analog.similarity_score >= 0.75 ? '#E63946'
                             : analog.similarity_score >= 0.55 ? '#EAB308' : '#6B6B6B',
                      }}>
                        {analog.similarity_label} ({Math.round(analog.similarity_score * 100)}% match)
                      </div>
                      <p style={{ fontSize: '12px', color: '#A0A0A0', margin: '0 0 6px', lineHeight: '1.5' }}>
                        {analog.outcome}
                      </p>
                      <p style={{ fontSize: '11px', color: '#EAB308', margin: 0, fontStyle: 'italic', lineHeight: '1.5' }}>
                        Lesson: {analog.lesson}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* 7 — Citation Panel (only when corpus was queried, i.e. not quick mode) */}
          {hasCitations && !quickMode && (
            <div style={{ animation: 'revealSection 0.4s 0.6s ease forwards', opacity: 0 }}>
              <CitationPanel citations={response.citations} />
            </div>
          )}

          {/* Quick Mode upsell when no citations */}
          {!hasCitations && quickMode && showResults && (
            <div style={{
              padding: '16px 20px', borderRadius: '10px',
              background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px',
            }}>
              <div>
                <p style={{ fontSize: '13px', fontWeight: 600, color: '#A0A0A0', margin: '0 0 2px' }}>
                  No corpus citations in Quick Mode
                </p>
                <p style={{ fontSize: '12px', color: '#5B5B5B', margin: 0 }}>
                  Run in Full Mode to retrieve historical evidence, RBI circulars, and broker research supporting your thesis.
                </p>
              </div>
              <button
                onClick={() => { setQuickMode(false); handleSubmit(currentThesis); }}
                disabled={loading}
                style={{
                  padding: '8px 16px', borderRadius: '8px', cursor: 'pointer',
                  background: 'rgba(230,57,70,0.12)', border: '1px solid rgba(230,57,70,0.3)',
                  color: '#E63946', fontSize: '13px', fontWeight: 600, whiteSpace: 'nowrap',
                  transition: 'all 0.2s ease',
                }}
              >
                Run Full Analysis →
              </button>
            </div>
          )}
        </div>
      )}

      {/* ── Empty state ───────────────────────────────────────────────────────── */}
      {!response && !loading && !error && (
        <div style={{ textAlign: 'center', padding: '60px 20px' }}>
          <div style={{ fontSize: '56px', marginBottom: '16px', opacity: 0.4 }}>⚡</div>
          <p style={{ color: '#6B6B6B', fontSize: '15px', marginBottom: '32px', opacity: 0.7 }}>
            Enter an investment thesis above to begin stress-testing
          </p>

          {/* Example theses */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxWidth: '680px', margin: '0 auto' }}>
            <p style={{ fontSize: '12px', color: '#4B4B4B', textTransform: 'uppercase', letterSpacing: '0.08em', margin: '0 0 4px' }}>
              Example theses
            </p>
            {EXAMPLE_THESES.map(thesis => (
              <button
                key={thesis}
                onClick={() => handleSubmit(thesis)}
                style={{
                  padding: '12px 18px', borderRadius: '10px', cursor: 'pointer', textAlign: 'left',
                  background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
                  color: '#A0A0A0', fontSize: '13px', lineHeight: '1.5',
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={e => {
                  const el = e.currentTarget;
                  el.style.background    = 'rgba(230,57,70,0.07)';
                  el.style.borderColor   = 'rgba(230,57,70,0.3)';
                  el.style.color         = '#F1F1F1';
                }}
                onMouseLeave={e => {
                  const el = e.currentTarget;
                  el.style.background    = 'rgba(255,255,255,0.03)';
                  el.style.borderColor   = 'rgba(255,255,255,0.08)';
                  el.style.color         = '#A0A0A0';
                }}
              >
                {thesis}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}