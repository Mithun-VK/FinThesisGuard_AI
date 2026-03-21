'use client';

import { useState } from 'react';
import { Clock, Star, Trash2, ChevronLeft, ChevronRight } from 'lucide-react';

interface SidebarItem {
  text: string;
  type: 'query' | 'thesis';
  timestamp: string;
}

interface SidebarProps {
  recentQueries?: string[];
  recentTheses?: string[];
  onSelectQuery?: (query: string) => void;
  onSelectThesis?: (thesis: string) => void;
  onClearHistory?: () => void;
}

export default function Sidebar({
  recentQueries = [],
  recentTheses = [],
  onSelectQuery,
  onSelectThesis,
  onClearHistory,
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  const items: SidebarItem[] = [
    ...recentQueries.map(q => ({ text: q, type: 'query' as const, timestamp: 'Recent' })),
    ...recentTheses.map(t => ({ text: t, type: 'thesis' as const, timestamp: 'Recent' })),
  ];

  if (collapsed) {
    return (
      <div style={{
        width: '48px',
        background: '#1A1A1A',
        borderRight: '1px solid #2A2A2A',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: '16px 0',
        transition: 'width 0.3s ease',
      }}>
        <button
          onClick={() => setCollapsed(false)}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#A0A0A0', padding: '8px' }}
        >
          <ChevronRight size={18} />
        </button>
      </div>
    );
  }

  return (
    <div style={{
      width: '260px',
      background: '#1A1A1A',
      borderRight: '1px solid #2A2A2A',
      display: 'flex',
      flexDirection: 'column',
      transition: 'width 0.3s ease',
      flexShrink: 0,
    }}>
      {/* Header */}
      <div style={{ padding: '16px', borderBottom: '1px solid #2A2A2A', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '13px', fontWeight: '600', color: '#A0A0A0', letterSpacing: '0.05em', textTransform: 'uppercase' }}>History</span>
        <button
          onClick={() => setCollapsed(true)}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#A0A0A0', padding: '4px' }}
        >
          <ChevronLeft size={16} />
        </button>
      </div>

      {/* Items */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
        {items.length === 0 ? (
          <div style={{ padding: '24px 16px', textAlign: 'center' }}>
            <Star size={24} color="#2A2A2A" style={{ margin: '0 auto 8px' }} />
            <p style={{ color: '#A0A0A0', fontSize: '12px', margin: 0 }}>No history yet</p>
          </div>
        ) : (
          <>
            {recentQueries.length > 0 && (
              <div style={{ marginBottom: '8px' }}>
                <p style={{ fontSize: '11px', color: '#A0A0A0', padding: '4px 8px', margin: '0 0 4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Recent Queries
                </p>
                {recentQueries.slice(0, 5).map((q, i) => (
                  <button
                    key={i}
                    onClick={() => onSelectQuery?.(q)}
                    style={{
                      width: '100%',
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      textAlign: 'left',
                      padding: '8px',
                      borderRadius: '6px',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '8px',
                      transition: 'background 0.15s',
                    }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'rgba(230, 57, 70, 0.1)')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                  >
                    <Clock size={12} color="#E63946" style={{ flexShrink: 0, marginTop: '2px' }} />
                    <span style={{ fontSize: '12px', color: '#F1F1F1', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {q}
                    </span>
                  </button>
                ))}
              </div>
            )}

            {recentTheses.length > 0 && (
              <div>
                <p style={{ fontSize: '11px', color: '#A0A0A0', padding: '4px 8px', margin: '0 0 4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Recent Theses
                </p>
                {recentTheses.slice(0, 3).map((t, i) => (
                  <button
                    key={i}
                    onClick={() => onSelectThesis?.(t)}
                    style={{
                      width: '100%',
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      textAlign: 'left',
                      padding: '8px',
                      borderRadius: '6px',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '8px',
                    }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'rgba(230, 57, 70, 0.1)')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                  >
                    <Star size={12} color="#FFA500" style={{ flexShrink: 0, marginTop: '2px' }} />
                    <span style={{ fontSize: '12px', color: '#F1F1F1', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {t.substring(0, 60)}...
                    </span>
                  </button>
                ))}
              </div>
            )}
          </>
        )}
      </div>

      {/* Clear button */}
      {items.length > 0 && (
        <div style={{ padding: '12px', borderTop: '1px solid #2A2A2A' }}>
          <button
            onClick={onClearHistory}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              background: 'none',
              border: '1px solid #2A2A2A',
              borderRadius: '6px',
              padding: '8px',
              cursor: 'pointer',
              color: '#A0A0A0',
              fontSize: '12px',
              transition: 'all 0.15s',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = '#E63946';
              e.currentTarget.style.color = '#E63946';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = '#2A2A2A';
              e.currentTarget.style.color = '#A0A0A0';
            }}
          >
            <Trash2 size={12} />
            Clear History
          </button>
        </div>
      )}
    </div>
  );
}
