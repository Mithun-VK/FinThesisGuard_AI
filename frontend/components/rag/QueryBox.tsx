'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, ArrowRight } from 'lucide-react';
import { EXAMPLE_QUERIES } from '@/lib/constants';

interface QueryBoxProps {
  onSubmit: (query: string) => void;
  loading?: boolean;
  initialValue?: string;
}

const MAX_CHARS = 1000;

const THESIS_KEYWORDS = ['will outperform', 'investment thesis', 'valuation', 'capex', 'margin expansion', 'stress-test', 'due to', 'driven by', 'CAGR'];

function detectThesisInput(text: string): boolean {
  const lower = text.toLowerCase();
  const matchCount = THESIS_KEYWORDS.filter(k => lower.includes(k.toLowerCase())).length;
  return matchCount >= 2 || text.length > 300;
}

export default function QueryBox({ onSubmit, loading = false, initialValue = '' }: QueryBoxProps) {
  const [query, setQuery] = useState(initialValue);
  const [focused, setFocused] = useState(false);
  const [showThesisBanner, setShowThesisBanner] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { if (initialValue) setQuery(initialValue); }, [initialValue]);

  useEffect(() => {
    setShowThesisBanner(query.length > 50 && detectThesisInput(query));
  }, [query]);

  const handleSubmit = useCallback(() => {
    if (query.trim() && !loading) {
      onSubmit(query.trim());
    }
  }, [query, loading, onSubmit]);

  const charCount = query.length;
  const isOverLimit = charCount > MAX_CHARS;

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === 'Enter' && query.trim()) {
        e.preventDefault();
        handleSubmit();
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [query, handleSubmit]);

  return (
    <div style={{ width: '100%' }}>
      {/* Main input container */}
      <div style={{
        position: 'relative',
        background: '#1A1A1A',
        border: `1px solid ${focused ? '#E63946' : '#2A2A2A'}`,
        borderRadius: '16px',
        transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
        boxShadow: focused ? '0 0 0 2px rgba(230, 57, 70, 0.15)' : 'none',
      }}>
        <textarea
          ref={textareaRef}
          value={query}
          onChange={e => setQuery(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="Ask about HDFC Bank NIM, SEBI circulars, NRI tax rules..."
          rows={3}
          style={{
            width: '100%',
            background: 'transparent',
            border: 'none',
            outline: 'none',
            resize: 'none',
            padding: '18px 20px 48px',
            color: '#F1F1F1',
            fontSize: '16px',
            lineHeight: '1.6',
            fontFamily: 'Inter, sans-serif',
          }}
        />

        {/* Bottom bar */}
        <div style={{
          position: 'absolute',
          bottom: '14px',
          left: '20px',
          right: '14px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <span style={{
            fontSize: '11px',
            color: isOverLimit ? '#E63946' : '#A0A0A0',
          }}>
            {charCount}/{MAX_CHARS} · Ctrl+Enter to submit
          </span>
          <button
            onClick={handleSubmit}
            disabled={!query.trim() || loading || isOverLimit}
            style={{
              background: query.trim() && !loading && !isOverLimit ? '#E63946' : '#2A2A2A',
              border: 'none',
              borderRadius: '10px',
              padding: '8px 16px',
              cursor: query.trim() && !loading && !isOverLimit ? 'pointer' : 'not-allowed',
              color: 'white',
              fontSize: '13px',
              fontWeight: '600',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              transition: 'all 0.2s ease',
              boxShadow: query.trim() && !loading && !isOverLimit ? '0 0 15px rgba(230, 57, 70, 0.35)' : 'none',
            }}
            onMouseEnter={e => {
              if (query.trim() && !loading) {
                (e.currentTarget as HTMLElement).style.boxShadow = '0 0 20px rgba(230, 57, 70, 0.6)';
              }
            }}
            onMouseLeave={e => {
              if (query.trim() && !loading) {
                (e.currentTarget as HTMLElement).style.boxShadow = '0 0 15px rgba(230, 57, 70, 0.35)';
              }
            }}
          >
            {loading ? (
              <>
                <div style={{ width: '12px', height: '12px', borderRadius: '50%', border: '2px solid rgba(255,255,255,0.3)', borderTop: '2px solid white', animation: 'spin 0.8s linear infinite' }} />
                Analyzing...
              </>
            ) : (
              <>
                <Send size={13} />
                Send
              </>
            )}
          </button>
        </div>
      </div>

      {/* Thesis detection banner */}
      {showThesisBanner && (
        <div style={{
          marginTop: '10px',
          padding: '10px 16px',
          background: 'rgba(230, 57, 70, 0.08)',
          border: '1px solid rgba(230, 57, 70, 0.3)',
          borderRadius: '10px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
        }}>
          <p style={{ fontSize: '13px', color: '#A0A0A0', margin: 0 }}>
            💡 This looks like an investment thesis. Try our{' '}
            <strong style={{ color: '#E63946' }}>Thesis Validator</strong> for a deeper analysis.
          </p>
          <a
            href="/thesis"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              color: '#E63946',
              fontSize: '12px',
              fontWeight: '600',
              textDecoration: 'none',
              whiteSpace: 'nowrap',
            }}
          >
            Switch <ArrowRight size={12} />
          </a>
        </div>
      )}

      {/* Example query chips */}
      <div style={{ marginTop: '12px', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        {EXAMPLE_QUERIES.slice(0, 4).map((q, i) => (
          <button
            key={i}
            onClick={() => setQuery(q)}
            style={{
              background: '#1A1A1A',
              border: '1px solid #2A2A2A',
              borderRadius: '9999px',
              padding: '6px 14px',
              fontSize: '12px',
              color: '#A0A0A0',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
              whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => {
              (e.currentTarget as HTMLElement).style.borderColor = '#E63946';
              (e.currentTarget as HTMLElement).style.color = '#E63946';
            }}
            onMouseLeave={e => {
              (e.currentTarget as HTMLElement).style.borderColor = '#2A2A2A';
              (e.currentTarget as HTMLElement).style.color = '#A0A0A0';
            }}
          >
            {q.length > 50 ? q.substring(0, 50) + '...' : q}
          </button>
        ))}
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
