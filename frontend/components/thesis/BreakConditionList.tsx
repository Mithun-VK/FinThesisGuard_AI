import type { BreakCondition } from '@/lib/types';
import { Zap } from 'lucide-react';

interface BreakConditionListProps {
  conditions: BreakCondition[];
}

const PROBABILITY_STYLES = {
  High: { color: '#E63946', bg: 'rgba(230, 57, 70, 0.12)' },
  Medium: { color: '#EAB308', bg: 'rgba(234, 179, 8, 0.12)' },
  Low: { color: '#22C55E', bg: 'rgba(34, 197, 94, 0.12)' },
};

export default function BreakConditionList({ conditions }: BreakConditionListProps) {
  if (!conditions || conditions.length === 0) return null;

  return (
    <div style={{
      background: '#1A1A1A',
      border: '1px solid #2A2A2A',
      borderRadius: '12px',
      overflow: 'hidden',
    }}>
      <div style={{ padding: '16px 20px', borderBottom: '1px solid #2A2A2A', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Zap size={16} color="#E63946" fill="#E63946" />
        <h3 style={{ fontSize: '15px', fontWeight: '600', color: '#E63946', margin: 0 }}>
          Break Conditions
        </h3>
      </div>

      <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {conditions.map((c, i) => {
          const probStyle = PROBABILITY_STYLES[c.probability] || PROBABILITY_STYLES.Medium;
          const isTriggered = c.triggered;

          return (
            <div
              key={i}
              style={{
                background: isTriggered ? 'rgba(230, 57, 70, 0.08)' : '#0A0A0A',
                border: `1px solid ${isTriggered ? '#E63946' : '#2A2A2A'}`,
                borderRadius: '10px',
                padding: '14px 16px',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '12px',
              }}
            >
              {/* Icon */}
              <Zap
                size={16}
                color="#E63946"
                fill={isTriggered ? "#E63946" : "none"}
                style={{
                  flexShrink: 0,
                  marginTop: '1px',
                  animation: isTriggered ? 'pulseRed 1.5s ease-in-out infinite' : 'none',
                }}
              />

              <div style={{ flex: 1 }}>
                <p style={{ fontSize: '13px', color: '#F1F1F1', margin: '0 0 8px', lineHeight: '1.5' }}>
                  {c.condition}
                </p>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                  {/* Probability */}
                  <span style={{
                    fontSize: '11px', fontWeight: '600',
                    color: probStyle.color,
                    background: probStyle.bg,
                    padding: '2px 8px',
                    borderRadius: '9999px',
                  }}>
                    {c.probability} likelihood
                  </span>

                  {/* Status */}
                  <span style={{
                    fontSize: '11px', fontWeight: '600',
                    color: isTriggered ? '#E63946' : '#22C55E',
                    background: isTriggered ? 'rgba(230, 57, 70, 0.15)' : 'rgba(34, 197, 94, 0.15)',
                    padding: '2px 8px',
                    borderRadius: '9999px',
                    animation: isTriggered ? 'pulseRed 1.5s ease-in-out infinite' : 'none',
                  }}>
                    {isTriggered ? '🔴 Triggered' : '🟢 Monitoring'}
                  </span>

                  {c.monitoring_frequency && (
                    <span style={{ fontSize: '11px', color: '#A0A0A0' }}>{c.monitoring_frequency}</span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
