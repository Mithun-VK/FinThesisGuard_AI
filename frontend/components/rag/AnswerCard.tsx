'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Copy, ChevronDown, ChevronUp, Check } from 'lucide-react';
import ConfidenceBadge from './ConfidenceBadge';
import type { QueryResponse } from '@/lib/types';

interface AnswerCardProps {
  response: QueryResponse;
}

export default function AnswerCard({ response }: AnswerCardProps) {
  const [reasoningOpen, setReasoningOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(response.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Add citation markers to answer text
  const answerWithCitations = response.answer.replace(
    /\[(\d+)\]/g,
    (_, num) => `<sup class="citation-marker">[${num}]</sup>`
  );

  return (
    <div style={{
      background: '#1A1A1A',
      border: '1px solid #2A2A2A',
      borderLeft: '4px solid #E63946',
      borderRadius: '12px',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 20px',
        borderBottom: '1px solid #2A2A2A',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <h3 style={{ fontSize: '15px', fontWeight: '600', color: '#F1F1F1', margin: 0 }}>Answer</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <ConfidenceBadge confidence={response.confidence} />
          <button
            onClick={handleCopy}
            style={{
              background: 'none',
              border: '1px solid #2A2A2A',
              borderRadius: '6px',
              padding: '5px 10px',
              cursor: 'pointer',
              color: '#A0A0A0',
              fontSize: '12px',
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              transition: 'all 0.15s',
            }}
            onMouseEnter={e => (e.currentTarget.style.borderColor = '#E63946')}
            onMouseLeave={e => (e.currentTarget.style.borderColor = '#2A2A2A')}
          >
            {copied ? <Check size={12} color="#22C55E" /> : <Copy size={12} />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>

      {/* Answer body */}
      <div style={{ padding: '20px' }}>
        <div className="answer-markdown" style={{
          color: '#F1F1F1',
          fontSize: '15px',
          lineHeight: '1.8',
        }}>
          <ReactMarkdown
            components={{
              p: ({ children }) => (
                <p>
                  {React.Children.map(children, (child) => {
                    if (typeof child === 'string') {
                      const parts = child.split(/(\[\d+\])/g);
                      return parts.map((part, i) => {
                        const match = part.match(/\[(\d+)\]/);
                        if (match) {
                          return <sup key={i} className="citation-marker">{part}</sup>;
                        }
                        return part;
                      });
                    }
                    return child;
                  })}
                </p>
              ),
            }}
          >
            {response.answer}
          </ReactMarkdown>
        </div>

        {/* Data gaps */}
        {response.data_gaps && response.data_gaps.length > 0 && (
          <div style={{
            marginTop: '16px',
            padding: '12px 16px',
            background: 'rgba(160, 160, 160, 0.08)',
            border: '1px solid #2A2A2A',
            borderRadius: '8px',
          }}>
            <p style={{ fontSize: '12px', color: '#A0A0A0', margin: '0 0 6px', fontWeight: '600' }}>Data Gaps:</p>
            <ul style={{ margin: 0, padding: '0 0 0 16px' }}>
              {response.data_gaps.map((gap, i) => (
                <li key={i} style={{ fontSize: '12px', color: '#A0A0A0', marginBottom: '4px' }}>{gap}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Reasoning section */}
      {response.reasoning && (
        <div style={{ borderTop: '1px solid #2A2A2A' }}>
          <button
            onClick={() => setReasoningOpen(!reasoningOpen)}
            style={{
              width: '100%',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '12px 20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              color: '#A0A0A0',
              fontSize: '13px',
              fontWeight: '500',
            }}
          >
            <span>Reasoning</span>
            {reasoningOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {reasoningOpen && (
            <div style={{
              padding: '0 20px 16px',
              color: '#A0A0A0',
              fontSize: '13px',
              lineHeight: '1.6',
              fontStyle: 'italic',
            }}>
              {response.reasoning}
            </div>
          )}
        </div>
      )}

      <style>{`
        .answer-markdown h1, .answer-markdown h2, .answer-markdown h3 {
          color: #F1F1F1;
          font-family: 'Space Grotesk', sans-serif;
          margin: 16px 0 8px;
        }
        .answer-markdown p { margin: 0 0 12px; }
        .answer-markdown ul, .answer-markdown ol { padding-left: 20px; margin: 0 0 12px; }
        .answer-markdown li { margin-bottom: 4px; }
        .answer-markdown strong { color: #F1F1F1; font-weight: 600; }
        .answer-markdown code { background: #2A2A2A; padding: 2px 6px; border-radius: 4px; font-size: 13px; }
        .citation-marker { color: #E63946; font-weight: 700; font-size: 11px; }
      `}</style>
    </div>
  );
}
