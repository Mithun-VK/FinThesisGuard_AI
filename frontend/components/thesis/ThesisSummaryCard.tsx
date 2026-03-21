import type { ThesisResponse } from '@/lib/types';
import { Download, Share2 } from 'lucide-react';
import ConfidenceBadge from '@/components/rag/ConfidenceBadge';
import type { ConfidenceLevel } from '@/lib/types';

interface ThesisSummaryCardProps {
  response: ThesisResponse;
  thesisText: string;
  onExport?: () => void;
  onShare?: () => void;
}

function getStrengthColor(strength: string): string {
  const lower = strength.toLowerCase();
  if (lower === 'strong' || lower === 'high') return '#22C55E';
  if (lower === 'medium' || lower === 'moderate') return '#EAB308';
  return '#E63946';
}

export default function ThesisSummaryCard({ response, thesisText, onExport, onShare }: ThesisSummaryCardProps) {
  const strengthColor = getStrengthColor(response.thesis_strength);
  const robustness = response.structural_robustness ?? 65;

  return (
    <div style={{
      background: '#1A1A1A',
      border: '1px solid #2A2A2A',
      borderRadius: '16px',
      padding: '24px',
      marginBottom: '24px',
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: '700', color: '#F1F1F1', margin: '0 0 4px', fontFamily: "'Space Grotesk', sans-serif" }}>
            Thesis Validation Report
          </h2>
          <p style={{ color: '#A0A0A0', fontSize: '13px', margin: 0, fontStyle: 'italic', maxWidth: '500px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            &ldquo;{thesisText.substring(0, 120)}{thesisText.length > 120 ? '...' : ''}&rdquo;
          </p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={onShare}
            style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              background: 'none', border: '1px solid #2A2A2A', borderRadius: '8px', padding: '8px 14px',
              cursor: 'pointer', color: '#A0A0A0', fontSize: '13px', transition: 'all 0.15s',
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = '#E63946'; (e.currentTarget as HTMLElement).style.color = '#E63946'; }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = '#2A2A2A'; (e.currentTarget as HTMLElement).style.color = '#A0A0A0'; }}
          >
            <Share2 size={13} /> Share
          </button>
          <button
            onClick={onExport}
            style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              background: 'none', border: '1px solid #E63946', borderRadius: '8px', padding: '8px 14px',
              cursor: 'pointer', color: '#E63946', fontSize: '13px', fontWeight: '500', transition: 'all 0.15s',
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(230,57,70,0.1)'; }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'none'; }}
          >
            <Download size={13} /> Export PDF
          </button>
        </div>
      </div>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px' }}>
        {/* Thesis Strength */}
        <div style={{ background: '#0A0A0A', borderRadius: '10px', padding: '16px' }}>
          <p style={{ fontSize: '11px', color: '#A0A0A0', margin: '0 0 6px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Thesis Strength</p>
          <p style={{ fontSize: '26px', fontWeight: '700', color: strengthColor, margin: 0, fontFamily: "'Space Grotesk', sans-serif" }}>
            {response.thesis_strength}
          </p>
        </div>

        {/* Confidence */}
        <div style={{ background: '#0A0A0A', borderRadius: '10px', padding: '16px' }}>
          <p style={{ fontSize: '11px', color: '#A0A0A0', margin: '0 0 10px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Confidence</p>
          <ConfidenceBadge confidence={response.confidence as ConfidenceLevel} />
        </div>

        {/* Structural Robustness */}
        <div style={{ background: '#0A0A0A', borderRadius: '10px', padding: '16px' }}>
          <p style={{ fontSize: '11px', color: '#A0A0A0', margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Structural Robustness</p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ flex: 1, height: '6px', background: '#2A2A2A', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${robustness}%`,
                background: robustness === 'High' ? '#22C55E' : robustness === 'Medium' ? '#EAB308' : '#E63946',
                borderRadius: '3px',
              }} />
            </div>
            <span style={{ fontSize: '14px', fontWeight: '700', color: '#F1F1F1' }}>{robustness}%</span>
          </div>
        </div>

        {/* Key Risk */}
        <div style={{ background: '#0A0A0A', borderRadius: '10px', padding: '16px' }}>
          <p style={{ fontSize: '11px', color: '#A0A0A0', margin: '0 0 6px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Key Risk</p>
          <p style={{ fontSize: '13px', fontWeight: '600', color: '#E63946', margin: 0 }}>
            {response.risks?.[0]?.dimension_label || 'N/A'}
          </p>
        </div>

        {/* Historical Analog */}
        {(response.historical_analogs?.[0]?.title || (response.historical_analogs && response.historical_analogs.length > 0)) && (
          <div style={{ background: '#0A0A0A', borderRadius: '10px', padding: '16px' }}>
            <p style={{ fontSize: '11px', color: '#A0A0A0', margin: '0 0 6px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Historical Analog</p>
            <p style={{ fontSize: '13px', fontWeight: '600', color: '#F1F1F1', margin: 0 }}>
              {response.historical_analogs?.[0]?.title}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
