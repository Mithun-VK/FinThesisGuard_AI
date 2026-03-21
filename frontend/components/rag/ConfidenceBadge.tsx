import type { ConfidenceLevel } from '@/lib/types';
import { CONFIDENCE_STYLES } from '@/lib/constants';
import Tooltip from '@/components/shared/Tooltip';

interface ConfidenceBadgeProps {
  confidence: ConfidenceLevel;
}

const CONFIDENCE_TOOLTIPS: Record<ConfidenceLevel, string> = {
  High: 'Multiple authoritative sources agree. High data consistency.',
  Medium: 'Sources partially agree. Some gaps in evidence.',
  Low: 'Conflicting sources or limited coverage. Verify manually.',
};

export default function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  const styles = CONFIDENCE_STYLES[confidence];
  const isLow = confidence === 'Low';

  return (
    <Tooltip content={CONFIDENCE_TOOLTIPS[confidence]}>
      <span
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '5px',
          background: styles.bg,
          color: styles.text,
          border: `1px solid ${styles.border}`,
          borderRadius: '9999px',
          fontSize: '11px',
          padding: '4px 10px',
          fontWeight: '600',
          whiteSpace: 'nowrap',
          cursor: 'default',
          animation: isLow ? 'pulseRed 1.5s ease-in-out infinite' : 'none',
        }}
      >
        <span style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          background: styles.text,
          display: 'inline-block',
          flexShrink: 0,
        }} />
        {styles.label}
      </span>
    </Tooltip>
  );
}
