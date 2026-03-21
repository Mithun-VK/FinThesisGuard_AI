'use client';

import { useState, useEffect } from 'react';
import { Lightbulb, Loader2 } from 'lucide-react';
import { EXAMPLE_THESES } from '@/lib/constants';

interface ThesisInputProps {
  onSubmit: (thesis: string) => void;
  loading?: boolean;
  initialValue?: string;
}

export default function ThesisInput({ onSubmit, loading = false, initialValue = '' }: ThesisInputProps) {
  const [thesis, setThesis] = useState(initialValue);
  const [focused, setFocused] = useState(false);

  useEffect(() => { if (initialValue) setThesis(initialValue); }, [initialValue]);

  const wordCount = thesis.trim() ? thesis.trim().split(/\s+/).length : 0;
  const estimatedTime = Math.max(15, Math.round(wordCount * 0.3));

  return (
    <div style={{ width: '100%' }}>
      {/* Example buttons */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '13px', color: '#A0A0A0', display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Lightbulb size={13} /> Load example:
        </span>
        {Object.entries(EXAMPLE_THESES).map(([key, value]) => (
          <button
            key={key}
            onClick={() => setThesis(value)}
            style={{
              background: 'rgba(230, 57, 70, 0.08)',
              border: '1px solid rgba(230, 57, 70, 0.3)',
              borderRadius: '6px',
              padding: '4px 12px',
              cursor: 'pointer',
              color: '#E63946',
              fontSize: '12px',
              fontWeight: '500',
              transition: 'all 0.15s',
            }}
            onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = 'rgba(230, 57, 70, 0.15)'}
            onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = 'rgba(230, 57, 70, 0.08)'}
          >
            {key === 'nvidia' ? 'NVIDIA AI thesis' : 'HDFC Bank thesis'}
          </button>
        ))}
      </div>

      {/* Textarea */}
      <div style={{
        background: '#1A1A1A',
        border: `1px solid ${focused ? '#E63946' : '#2A2A2A'}`,
        borderRadius: '12px',
        transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
        boxShadow: focused ? '0 0 0 2px rgba(230, 57, 70, 0.15)' : loading ? '0 0 20px rgba(230, 57, 70, 0.2)' : 'none',
        animation: loading ? 'borderGlow 2s ease-in-out infinite' : 'none',
      }}>
        <textarea
          value={thesis}
          onChange={e => setThesis(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="e.g. NVIDIA will outperform due to AI demand growth and hyperscaler capex expansion..."
          rows={8}
          style={{
            width: '100%',
            background: 'transparent',
            border: 'none',
            outline: 'none',
            resize: 'vertical',
            padding: '18px 20px',
            color: '#F1F1F1',
            fontSize: '15px',
            lineHeight: '1.7',
            fontFamily: 'Inter, sans-serif',
          }}
        />
      </div>

      {/* Meta row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '8px', padding: '0 4px' }}>
        <div style={{ display: 'flex', gap: '16px' }}>
          <span style={{ fontSize: '12px', color: '#A0A0A0' }}>
            {wordCount} words
          </span>
          {wordCount > 0 && (
            <span style={{ fontSize: '12px', color: '#A0A0A0' }}>
              ~{estimatedTime}s analysis time
            </span>
          )}
        </div>
      </div>

      {/* Submit button */}
      <button
        onClick={() => thesis.trim() && !loading && onSubmit(thesis.trim())}
        disabled={!thesis.trim() || loading}
        style={{
          marginTop: '16px',
          width: '100%',
          background: thesis.trim() && !loading ? 'linear-gradient(135deg, #E63946, #B52A35)' : '#2A2A2A',
          border: 'none',
          borderRadius: '12px',
          padding: '16px',
          cursor: thesis.trim() && !loading ? 'pointer' : 'not-allowed',
          color: 'white',
          fontSize: '16px',
          fontWeight: '700',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '10px',
          transition: 'all 0.2s ease',
          boxShadow: thesis.trim() && !loading ? '0 4px 20px rgba(230, 57, 70, 0.3)' : 'none',
          fontFamily: "'Space Grotesk', sans-serif",
        }}
        onMouseEnter={e => {
          if (thesis.trim() && !loading) {
            (e.currentTarget as HTMLElement).style.boxShadow = '0 4px 30px rgba(230, 57, 70, 0.5)';
            (e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)';
          }
        }}
        onMouseLeave={e => {
          (e.currentTarget as HTMLElement).style.boxShadow = '0 4px 20px rgba(230, 57, 70, 0.3)';
          (e.currentTarget as HTMLElement).style.transform = 'translateY(0)';
        }}
      >
        {loading ? (
          <>
            <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
            Stress-testing your thesis...
          </>
        ) : (
          'Validate Thesis →'
        )}
      </button>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
