// components/thesis/RiskScoreCard.tsx
import { RISK_THRESHOLDS } from '@/lib/constants';
import type { RiskScore } from '@/lib/types';

interface RiskScoreCardProps {
  risk: RiskScore;
}

function getRiskColor(score: number): string {
  if (score >= RISK_THRESHOLDS.high) return '#FF4444';
  if (score >= RISK_THRESHOLDS.medium) return '#FFA500';
  return '#22C55E';
}

function CircularRing({ score, color }: { score: number; color: string }) {
  const radius = 30;
  const circumference = 2 * Math.PI * radius;
  const filled = (score / 10) * circumference;
  const empty = circumference - filled;

  return (
    <div style={{ position: 'relative', width: '80px', height: '80px' }}>
      <svg width="80" height="80" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="40" cy="40" r={radius} fill="none" stroke="#2A2A2A" strokeWidth="6" />
        <circle
          cx="40" cy="40" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeDasharray={`${filled} ${empty}`}
          strokeLinecap="round"
        />
      </svg>
      <div style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
      }}>
        <span style={{
          fontSize: '22px',
          fontWeight: '800',
          color,
          lineHeight: 1,
          fontFamily: "'Space Grotesk', sans-serif",
        }}>
          {score}
        </span>
        <span style={{ fontSize: '9px', color: '#A0A0A0' }}>/10</span>
      </div>
    </div>
  );
}

export default function RiskScoreCard({ risk }: RiskScoreCardProps) {
  const color = getRiskColor(risk.score);

  return (
    <div
      style={{
        background: '#0A0A0A',
        border: '1px solid #2A2A2A',
        borderRadius: '10px',
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '10px',
        transition: 'all 0.2s ease',
        cursor: 'default',
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLElement).style.borderColor = color;
        (e.currentTarget as HTMLElement).style.boxShadow = `0 0 15px ${color}22`;
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.borderColor = '#2A2A2A';
        (e.currentTarget as HTMLElement).style.boxShadow = 'none';
      }}
    >
      <CircularRing score={risk.score} color={color} />

      <div style={{ textAlign: 'center' }}>
        <p style={{
          fontSize: '13px',
          fontWeight: '600',
          color: '#F1F1F1',
          margin: '0 0 4px',
        }}>
          {risk.dimension_label}
        </p>

        <span style={{
          fontSize: '10px',
          fontWeight: '600',
          color,
          background: `${color}22`,
          padding: '2px 8px',
          borderRadius: '9999px',
        }}>
          {risk.severity_label} Risk
        </span>
      </div>

      {risk.rationale && (
        <p style={{
          fontSize: '11px',
          color: '#A0A0A0',
          textAlign: 'center',
          margin: 0,
          lineHeight: '1.5',
        }}>
          {risk.rationale}
        </p>
      )}
    </div>
  );
}
