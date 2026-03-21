type BadgeVariant = 'success' | 'warning' | 'danger' | 'info' | 'neutral';

interface StatusBadgeProps {
  variant: BadgeVariant;
  children: React.ReactNode;
  size?: 'sm' | 'md';
}

const VARIANT_STYLES: Record<BadgeVariant, { bg: string; text: string; border: string }> = {
  success: { bg: 'rgba(34, 197, 94, 0.15)', text: '#22C55E', border: 'rgba(34, 197, 94, 0.4)' },
  warning: { bg: 'rgba(234, 179, 8, 0.15)', text: '#EAB308', border: 'rgba(234, 179, 8, 0.4)' },
  danger: { bg: 'rgba(230, 57, 70, 0.15)', text: '#E63946', border: 'rgba(230, 57, 70, 0.4)' },
  info: { bg: 'rgba(59, 130, 246, 0.15)', text: '#60A5FA', border: 'rgba(59, 130, 246, 0.4)' },
  neutral: { bg: 'rgba(160, 160, 160, 0.15)', text: '#A0A0A0', border: 'rgba(160, 160, 160, 0.4)' },
};

export default function StatusBadge({ variant, children, size = 'sm' }: StatusBadgeProps) {
  const styles = VARIANT_STYLES[variant];
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        background: styles.bg,
        color: styles.text,
        border: `1px solid ${styles.border}`,
        borderRadius: '9999px',
        fontSize: size === 'sm' ? '11px' : '12px',
        padding: size === 'sm' ? '2px 8px' : '4px 12px',
        fontWeight: '500',
        lineHeight: 1.5,
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </span>
  );
}
