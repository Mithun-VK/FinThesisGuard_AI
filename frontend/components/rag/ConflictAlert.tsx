'use client';

import { useState } from 'react';
import { AlertTriangle, ChevronDown, ChevronUp, CheckCircle } from 'lucide-react';
import type { Conflict } from '@/lib/types';

interface ConflictAlertProps {
  conflicts: Conflict[];
}

export default function ConflictAlert({ conflicts }: ConflictAlertProps) {
  const [expanded, setExpanded] = useState(false);

  if (!conflicts || conflicts.length === 0) return null;

  return (
    <div style={{
      background: '#2A2000',
      border: '1px solid #EAB308',
      borderRadius: '12px',
      padding: '16px 20px',
      marginTop: '16px',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <AlertTriangle size={18} color="#EAB308" />
          <span style={{ fontWeight: '600', color: '#EAB308', fontSize: '14px' }}>
            Conflicting Information Detected
          </span>
          <span style={{
            background: 'rgba(234, 179, 8, 0.2)',
            color: '#EAB308',
            fontSize: '11px',
            padding: '2px 8px',
            borderRadius: '9999px',
            fontWeight: '600',
          }}>
            {conflicts.length} conflict{conflicts.length > 1 ? 's' : ''}
          </span>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#EAB308', padding: '4px' }}
        >
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {/* Summary of first conflict */}
      <div style={{ marginTop: '10px' }}>
        {conflicts.slice(0, expanded ? conflicts.length : 1).map((c, i) => (
          <div key={i} style={{ marginBottom: expanded && i < conflicts.length - 1 ? '12px' : 0 }}>
            <p style={{ color: '#F1F1F1', fontSize: '13px', margin: '0 0 6px' }}>
              <span style={{ color: '#EAB308' }}>{c.source_a}</span> says{' '}
              <span style={{ color: '#FF6B6B' }}>{c.field ? `${c.field} = ` : ''}{c.value_a}</span>
              {' '}vs{' '}
              <span style={{ color: '#EAB308' }}>{c.source_b}</span> says{' '}
              <span style={{ color: '#FF6B6B' }}>{c.field ? `${c.field} = ` : ''}{c.value_b}</span>
            </p>
            <p style={{ color: '#22C55E', fontSize: '12px', margin: 0, display: 'flex', alignItems: 'center', gap: '5px' }}>
              <CheckCircle size={12} />
              Recommended: Use{' '}
              <strong style={{ color: '#22C55E' }}>{c.recommended_source}</strong>
              {' '}— more recent + higher authority
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
