import type { Assumption } from '@/lib/types';
import { CheckCircle, XCircle, ChevronDown } from 'lucide-react';
import { useState } from 'react';

interface AssumptionTreeProps {
  assumptions: Assumption[];
}

export default function AssumptionTree({ assumptions }: AssumptionTreeProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  if (!assumptions || assumptions.length === 0) return null;

  return (
    <div style={{
      background: '#1A1A1A',
      border: '1px solid #2A2A2A',
      borderRadius: '12px',
      overflow: 'hidden',
    }}>
      <div style={{ padding: '16px 20px', borderBottom: '1px solid #2A2A2A' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '600', color: '#F1F1F1', margin: 0 }}>
          Key Assumptions
        </h3>
      </div>
      <div style={{ padding: '12px' }}>
        {assumptions.map((assumption, i) => {
          const isExpanded = expandedIndex === i;
          const confColor = assumption.confidence >= 70 ? '#22C55E' : assumption.confidence >= 40 ? '#EAB308' : '#E63946';

          return (
            <div
              key={i}
              style={{
                marginBottom: '8px',
                border: '1px solid #2A2A2A',
                borderLeft: '3px solid #E63946',
                borderRadius: '8px',
                overflow: 'hidden',
              }}
            >
              <div style={{ padding: '12px 14px' }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                  {/* Number */}
                  <span style={{
                    width: '22px', height: '22px', borderRadius: '50%',
                    background: '#2A2A2A', color: '#E63946', fontSize: '11px',
                    fontWeight: '700', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                  }}>
                    {i + 1}
                  </span>

                  <div style={{ flex: 1 }}>
                    <p style={{ fontSize: '13px', color: '#F1F1F1', margin: '0 0 8px', lineHeight: '1.5' }}>
                      {assumption.text}
                    </p>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
                      {/* Confidence */}
                      <span style={{
                        fontSize: '11px', fontWeight: '600', color: confColor,
                        background: `${confColor}22`, padding: '2px 8px', borderRadius: '9999px',
                      }}>
                        {assumption.confidence}% confidence
                      </span>

                      {/* Historical support */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', color: assumption.historical_support ? '#22C55E' : '#E63946' }}>
                        {assumption.historical_support
                          ? <><CheckCircle size={11} /> Historically supported</>
                          : <><XCircle size={11} /> No historical precedent</>
                        }
                      </div>

                      {/* Expand */}
                      {assumption.supporting_evidence?.length > 0 && (
                        <button
                          onClick={() => setExpandedIndex(isExpanded ? null : i)}
                          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#A0A0A0', display: 'flex', alignItems: 'center', gap: '3px', fontSize: '11px', marginLeft: 'auto' }}
                        >
                          Evidence <ChevronDown size={11} style={{ transform: isExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                        </button>
                      )}
                    </div>

                    {/* Evidence */}
                    {isExpanded && assumption.supporting_evidence?.length > 0 && (
                      <div style={{ marginTop: '10px', padding: '10px', background: '#0A0A0A', borderRadius: '6px' }}>
                        <p style={{ fontSize: '12px', color: '#A0A0A0', margin: 0, fontStyle: 'italic' }}>{assumption.supporting_evidence.join(' ')}</p>
                      </div>
                    )}
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
