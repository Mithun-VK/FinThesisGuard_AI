import type { Citation } from '@/lib/types';
import { AUTHORITY_STYLES } from '@/lib/constants';
import type { AuthorityType } from '@/lib/types';
import { ExternalLink, FileText } from 'lucide-react';

interface CitationPanelProps {
  citations: Citation[];
}

export default function CitationPanel({ citations }: CitationPanelProps) {
  if (!citations || citations.length === 0) return null;

  return (
    <div style={{
      background: '#1A1A1A',
      border: '1px solid #2A2A2A',
      borderRadius: '12px',
      overflow: 'hidden',
      height: 'fit-content',
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 20px',
        borderBottom: '1px solid #2A2A2A',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
      }}>
        <FileText size={16} color="#E63946" />
        <h3 style={{ fontSize: '15px', fontWeight: '600', color: '#F1F1F1', margin: 0 }}>Sources</h3>
        <span style={{
          background: '#E63946',
          color: 'white',
          fontSize: '11px',
          fontWeight: '700',
          padding: '2px 8px',
          borderRadius: '9999px',
          marginLeft: '2px',
        }}>
          {citations.length}
        </span>
      </div>

      {/* Citation cards */}
      <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {citations.map((citation) => {
          const authority = citation.source_type as AuthorityType;
          const authorityStyle = AUTHORITY_STYLES[authority] || { bg: '#2A2A2A', text: '#F1F1F1', border: '#2A2A2A' };
          const scorePercent = Math.round(citation.relevance_score * 100);

          return (
            <div
              key={citation.id}
              className="glass-card"
              style={{
                padding: '14px',
                cursor: citation.url ? 'pointer' : 'default',
                transition: 'all 0.2s ease',
              }}
              onClick={() => citation.url && window.open(citation.url, '_blank')}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.boxShadow = '0 0 15px rgba(230, 57, 70, 0.15)';
                (e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)';
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.boxShadow = 'none';
                (e.currentTarget as HTMLElement).style.transform = 'translateY(0)';
              }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                {/* Number circle */}
                <div style={{
                  width: '24px',
                  height: '24px',
                  borderRadius: '50%',
                  background: '#E63946',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '11px',
                  fontWeight: '700',
                  color: 'white',
                  flexShrink: 0,
                }}>
                  {citation.id}
                </div>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '8px', marginBottom: '6px' }}>
                    <p style={{ fontSize: '13px', fontWeight: '600', color: '#F1F1F1', margin: 0, lineHeight: '1.4' }}>
                      {citation.title}
                    </p>
                    {citation.url && <ExternalLink size={12} color="#A0A0A0" style={{ flexShrink: 0, marginTop: '2px' }} />}
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '8px' }}>
                    <span style={{
                      background: authorityStyle.bg,
                      color: authorityStyle.text,
                      border: `1px solid ${authorityStyle.border}`,
                      fontSize: '10px',
                      fontWeight: '600',
                      padding: '2px 8px',
                      borderRadius: '9999px',
                    }}>
                      {citation.source_type}
                    </span>
                    {citation.date && (
                      <span style={{ fontSize: '11px', color: '#A0A0A0' }}>{citation.date}</span>
                    )}
                  </div>

                  {/* Relevance score bar */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
                      <span style={{ fontSize: '10px', color: '#A0A0A0' }}>Relevance</span>
                      <span style={{ fontSize: '10px', color: '#E63946', fontWeight: '600' }}>{scorePercent}%</span>
                    </div>
                    <div style={{ height: '3px', background: '#2A2A2A', borderRadius: '2px', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%',
                        width: `${scorePercent}%`,
                        background: 'linear-gradient(90deg, #E63946, #FF6B6B)',
                        borderRadius: '2px',
                      }} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
