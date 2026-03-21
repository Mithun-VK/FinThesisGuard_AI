interface LoadingSpinnerProps {
  variant?: 'spinner' | 'skeleton' | 'dots';
  size?: 'sm' | 'md' | 'lg';
  text?: string;
}

export default function LoadingSpinner({ variant = 'spinner', size = 'md', text }: LoadingSpinnerProps) {
  const sizes = { sm: 24, md: 40, lg: 56 };
  const spinnerSize = sizes[size];

  if (variant === 'dots') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: '#E63946',
                animation: `bounceDot 1.4s ease-in-out ${i * 0.16}s infinite`,
              }}
            />
          ))}
        </div>
        {text && <p style={{ color: '#A0A0A0', fontSize: '13px', margin: 0 }}>{text}</p>}
        <style>{`
          @keyframes bounceDot {
            0%, 80%, 100% { transform: scale(0.4); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
          }
        `}</style>
      </div>
    );
  }

  if (variant === 'skeleton') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', width: '100%' }}>
        {/* Answer card skeleton */}
        <div style={{ background: '#1A1A1A', borderRadius: '12px', padding: '24px', border: '1px solid #2A2A2A' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div style={{ width: '30%', height: '20px', background: '#2A2A2A', borderRadius: '4px', animation: 'shimmer 1.5s infinite' }} />
            <div style={{ width: '80px', height: '24px', background: '#2A2A2A', borderRadius: '20px', animation: 'shimmer 1.5s infinite' }} />
          </div>
          {[100, 90, 80, 95, 70].map((w, i) => (
            <div
              key={i}
              style={{
                width: `${w}%`,
                height: '14px',
                background: '#2A2A2A',
                borderRadius: '4px',
                marginBottom: '10px',
                animation: `shimmer 1.5s ${i * 0.1}s infinite`,
              }}
            />
          ))}
          <div style={{ width: '60%', height: '14px', background: '#2A2A2A', borderRadius: '4px', animation: 'shimmer 1.5s infinite' }} />
        </div>
        {text && <p style={{ color: '#A0A0A0', fontSize: '13px', textAlign: 'center', margin: '4px 0 0' }}>{text}</p>}
        <style>{`
          @keyframes shimmer {
            0% { opacity: 0.5; }
            50% { opacity: 1; }
            100% { opacity: 0.5; }
          }
        `}</style>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
      <div style={{
        width: spinnerSize,
        height: spinnerSize,
        borderRadius: '50%',
        border: `3px solid #2A2A2A`,
        borderTop: `3px solid #E63946`,
        animation: 'spin 0.8s linear infinite',
      }} />
      {text && <p style={{ color: '#A0A0A0', fontSize: '13px', margin: 0 }}>{text}</p>}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
