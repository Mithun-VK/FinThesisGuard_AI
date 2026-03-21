'use client';

import Link from 'next/link';
import { Github, BookOpen, Server } from 'lucide-react';

export default function Footer() {
  return (
    <footer style={{
      background: '#0A0A0A',
      borderTop: '1px solid rgba(230, 57, 70, 0.4)',
      padding: '24px',
      marginTop: 'auto',
    }}>
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px',
      }}>
        <p style={{ color: '#A0A0A0', fontSize: '13px', margin: 0 }}>
          <span style={{ color: '#E63946', fontWeight: '600' }}>ArkAngel</span>
          {' '}© 2026 | FinThesisGuard AI
        </p>

        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <Link
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#A0A0A0', textDecoration: 'none', fontSize: '13px', transition: 'color 0.2s' }}
            onMouseEnter={e => (e.currentTarget.style.color = '#E63946')}
            onMouseLeave={e => (e.currentTarget.style.color = '#A0A0A0')}
          >
            <Github size={14} />
            GitHub
          </Link>
          <Link
            href="/docs"
            style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#A0A0A0', textDecoration: 'none', fontSize: '13px', transition: 'color 0.2s' }}
            onMouseEnter={e => (e.currentTarget.style.color = '#E63946')}
            onMouseLeave={e => (e.currentTarget.style.color = '#A0A0A0')}
          >
            <BookOpen size={14} />
            Docs
          </Link>
          <Link
            href="/architecture"
            style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#A0A0A0', textDecoration: 'none', fontSize: '13px', transition: 'color 0.2s' }}
            onMouseEnter={e => (e.currentTarget.style.color = '#E63946')}
            onMouseLeave={e => (e.currentTarget.style.color = '#A0A0A0')}
          >
            <Server size={14} />
            Architecture
          </Link>
        </div>
      </div>
    </footer>
  );
}
