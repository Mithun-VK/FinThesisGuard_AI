'use client';

import { useState } from 'react';

interface TooltipProps {
  content: string;
  children: React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
}

export default function Tooltip({ content, children, position = 'top' }: TooltipProps) {
  const [visible, setVisible] = useState(false);

  const getPositionStyle = () => {
    switch (position) {
      case 'top': return { bottom: '100%', left: '50%', transform: 'translateX(-50%)', marginBottom: '6px' };
      case 'bottom': return { top: '100%', left: '50%', transform: 'translateX(-50%)', marginTop: '6px' };
      case 'left': return { right: '100%', top: '50%', transform: 'translateY(-50%)', marginRight: '6px' };
      case 'right': return { left: '100%', top: '50%', transform: 'translateY(-50%)', marginLeft: '6px' };
    }
  };

  return (
    <div
      style={{ position: 'relative', display: 'inline-flex' }}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && (
        <div style={{
          position: 'absolute',
          ...getPositionStyle(),
          background: '#2A2A2A',
          color: '#F1F1F1',
          fontSize: '11px',
          padding: '6px 10px',
          borderRadius: '6px',
          whiteSpace: 'nowrap',
          zIndex: 100,
          border: '1px solid #3A3A3A',
          pointerEvents: 'none',
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
          maxWidth: '200px',
          whiteSpaceCollapse: 'break-spaces',
        }}>
          {content}
        </div>
      )}
    </div>
  );
}
