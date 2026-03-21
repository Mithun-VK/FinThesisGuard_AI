'use client';

import { useState, useCallback } from 'react';
import QueryBox from '@/components/rag/QueryBox';
import AnswerCard from '@/components/rag/AnswerCard';
import CitationPanel from '@/components/rag/CitationPanel';
import ConflictAlert from '@/components/rag/ConflictAlert';
import LoadingSpinner from '@/components/shared/LoadingSpinner';
import { queryRAG } from '@/lib/api';
import { getErrorMessage, isValidationError, isRateLimitError, isServiceUnavailable } from '@/lib/api';
import type { QueryResponse, QueryRequest } from '@/lib/types';

// ── Suggested queries shown in empty state ────────────────────────────────────
const SUGGESTED_QUERIES = [
  "What is HDFC Bank's NIM trend over the last 4 quarters?",
  "Summarize SEBI's latest guidelines on algo trading for retail investors",
  "What are the NRI tax implications for dividend income from Indian equities?",
  "Compare RBI's repo rate decisions in 2024 vs 2023",
];

// ── Map API error shapes to user-friendly messages ────────────────────────────
function toUserMessage(e: unknown): string {
  if (isValidationError(e))    return 'Your query could not be validated. Please try rephrasing it.';
  if (isRateLimitError(e))     return 'Too many requests. Please wait a moment and try again.';
  if (isServiceUnavailable(e)) return 'The analysis service is temporarily unavailable. Please retry in a few seconds.';
  const msg = getErrorMessage(e);
  if (msg.toLowerCase().includes('timeout')) return 'The request timed out. Try a shorter or simpler query.';
  return msg || 'An unexpected error occurred. Please try again.';
}

export default function HomePage() {
  const [response, setResponse]     = useState<QueryResponse | null>(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState<string | null>(null);
  const [activeQuery, setActiveQuery] = useState('');

  // ── Core submit handler ────────────────────────────────────────────────────
  const handleSubmit = useCallback(async (query: string) => {
    if (!query.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResponse(null);
    setActiveQuery(query.trim());

    const payload: QueryRequest = {
      query:     query.trim(),
      top_k:     10,
      use_cache: true,
      stream:    false,
    };

    try {
      // queryRAG already unwraps .data — result is QueryResponse directly
      const result = await queryRAG(payload);
      setResponse(result);
    } catch (e) {
      setError(toUserMessage(e));
    } finally {
      setLoading(false);
    }
  }, [loading]);

  const handleSuggestedQuery = (q: string) => handleSubmit(q);

  // ── Derived state ──────────────────────────────────────────────────────────
  const hasConflicts  = (response?.conflicts?.length ?? 0) > 0;
  const hasCitations  = (response?.citations?.length ?? 0) > 0;
  const showResults   = response !== null && !loading;

  return (
    <div style={{ minHeight: 'calc(100vh - 64px - 65px)', padding: '48px 40px' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>

        {/* ── Hero ──────────────────────────────────────────────────────────── */}
        <div style={{ textAlign: 'center', marginBottom: '40px' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: '8px',
            marginBottom: '16px', background: 'rgba(230, 57, 70, 0.1)',
            border: '1px solid rgba(230, 57, 70, 0.3)', borderRadius: '9999px',
            padding: '6px 16px',
          }}>
            <span style={{
              width: '6px', height: '6px', borderRadius: '50%',
              background: '#22C55E', display: 'inline-block',
            }} />
            <span style={{ fontSize: '12px', color: '#A0A0A0' }}>
              50M+ documents indexed · Live
            </span>
          </div>

          <h1 style={{
            fontSize: 'clamp(36px, 5vw, 64px)', fontWeight: '800',
            lineHeight: '1.1', margin: '0 0 16px',
            fontFamily: "'Space Grotesk', sans-serif",
          }}>
            <span className="gradient-text">Ask Anything.</span>
            <br />
            <span style={{ color: '#F1F1F1' }}>Get Verified Answers.</span>
          </h1>

          <p style={{ fontSize: '18px', color: '#A0A0A0', maxWidth: '560px', margin: '0 auto', lineHeight: '1.6' }}>
            Financial intelligence powered by RBI circulars, SEBI filings, annual reports, and broker research.
          </p>
        </div>

        {/* ── Query box ─────────────────────────────────────────────────────── */}
        <div style={{ maxWidth: '760px', margin: '0 auto 40px' }}>
          <QueryBox
            onSubmit={handleSubmit}
            loading={loading}
            initialValue={activeQuery}
          />
        </div>

        {/* ── Error — shown when error !== null ─────────────────────────────── */}
        {error && !loading && (
          <div style={{
            maxWidth: '760px', margin: '0 auto 24px', padding: '14px 18px',
            background: 'rgba(230, 57, 70, 0.08)',
            border: '1px solid rgba(230, 57, 70, 0.35)',
            borderRadius: '10px', display: 'flex', alignItems: 'flex-start', gap: '10px',
          }}>
            <span style={{ color: '#E63946', fontSize: '16px', marginTop: '1px' }}>⚠</span>
            <div>
              <p style={{ color: '#E63946', fontSize: '14px', margin: '0 0 4px', fontWeight: 600 }}>
                Query failed
              </p>
              <p style={{ color: '#C0A0A0', fontSize: '13px', margin: 0 }}>
                {error}
              </p>
            </div>
          </div>
        )}

        {/* ── Loading — shown while loading === true ────────────────────────── */}
        {loading && (
          <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
            <LoadingSpinner variant="skeleton" text="Analyzing across 50M+ documents..." />
          </div>
        )}

        {/* ── Results — gated behind result !== null && !loading ─────────────── */}
        {showResults && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: hasCitations ? '1fr minmax(280px, 340px)' : '1fr',
            gap: '24px', maxWidth: '1100px', margin: '0 auto',
            animation: 'revealSection 0.5s ease forwards',
          }}>
            {/* Left column: answer + conflict alert */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <AnswerCard response={response} />
              {hasConflicts && (
                <ConflictAlert conflicts={response.conflicts} />
              )}
            </div>

            {/* Right column: citations — only rendered when citations exist */}
            {hasCitations && (
              <div>
                <CitationPanel citations={response.citations} />
              </div>
            )}
          </div>
        )}

        {/* ── Empty state — shown when no query has been run ────────────────── */}
        {!response && !loading && !error && (
          <div style={{ textAlign: 'center', marginTop: '60px' }}>
            <div style={{ fontSize: '48px', marginBottom: '12px' }}>🔍</div>
            <p style={{ color: '#A0A0A0', fontSize: '15px', marginBottom: '32px' }}>
              Ask a financial question to get started
            </p>

            {/* Suggested queries */}
            <div style={{
              display: 'flex', flexWrap: 'wrap', gap: '10px',
              justifyContent: 'center', maxWidth: '760px', margin: '0 auto',
            }}>
              {SUGGESTED_QUERIES.map(q => (
                <button
                  key={q}
                  onClick={() => handleSuggestedQuery(q)}
                  style={{
                    padding: '8px 16px', borderRadius: '9999px', cursor: 'pointer',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.12)',
                    color: '#C0C0C0', fontSize: '13px',
                    transition: 'all 0.2s ease', textAlign: 'left',
                  }}
                  onMouseEnter={e => {
                    (e.currentTarget as HTMLButtonElement).style.background = 'rgba(230,57,70,0.12)';
                    (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(230,57,70,0.4)';
                    (e.currentTarget as HTMLButtonElement).style.color = '#F1F1F1';
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.05)';
                    (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.12)';
                    (e.currentTarget as HTMLButtonElement).style.color = '#C0C0C0';
                  }}
                >
                  {q.length > 60 ? q.slice(0, 60) + '…' : q}
                </button>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}