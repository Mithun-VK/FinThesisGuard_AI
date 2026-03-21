'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, X, Zap } from 'lucide-react';
import { NAV_LINKS } from '@/lib/constants';

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const pathname = usePathname();

  return (
    <nav style={{
      background: '#0A0A0A',
      borderBottom: '1px solid rgba(230, 57, 70, 0.3)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', height: '64px', gap: '16px' }}>
          {/* Logo */}
          <Link href="/" style={{ display: 'flex', alignItems: 'center', gap: '8px', textDecoration: 'none', flexShrink: 0 }}>
            <div style={{
              background: 'linear-gradient(135deg, #E63946, #B52A35)',
              borderRadius: '8px',
              padding: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <Zap size={18} color="white" fill="white" />
            </div>
            <span style={{ fontSize: '14px', fontWeight: '700', color: '#E63946', letterSpacing: '0.05em' }}>
              ARKANGEL
            </span>
          </Link>

          {/* Center wordmark */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center' }}>
            <span style={{
              fontSize: '18px',
              fontWeight: '700',
              color: '#F1F1F1',
              fontFamily: "'Space Grotesk', sans-serif",
              letterSpacing: '-0.02em',
            }}>
              FinThesisGuard AI
            </span>
          </div>

          {/* Desktop nav links */}
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }} className="desktop-nav">
            {NAV_LINKS.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  style={{
                    padding: '8px 16px',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '500',
                    color: isActive ? '#E63946' : '#A0A0A0',
                    textDecoration: 'none',
                    borderBottom: isActive ? '2px solid #E63946' : '2px solid transparent',
                    transition: 'color 0.2s ease',
                  }}
                  onMouseEnter={e => { if (!isActive) (e.target as HTMLElement).style.color = '#F1F1F1'; }}
                  onMouseLeave={e => { if (!isActive) (e.target as HTMLElement).style.color = '#A0A0A0'; }}
                >
                  {link.label}
                </Link>
              );
            })}
          </div>

          {/* Mobile hamburger */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#F1F1F1', padding: '4px' }}
            className="mobile-hamburger"
          >
            {mobileOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </div>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="animate-slide-down" style={{
          background: '#1A1A1A',
          borderBottom: '1px solid #2A2A2A',
          padding: '12px 24px',
        }}>
          {NAV_LINKS.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMobileOpen(false)}
                style={{
                  display: 'block',
                  padding: '12px 0',
                  fontSize: '16px',
                  fontWeight: '500',
                  color: isActive ? '#E63946' : '#F1F1F1',
                  textDecoration: 'none',
                  borderBottom: '1px solid #2A2A2A',
                }}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
      )}

      <style>{`
        @media (min-width: 768px) { .mobile-hamburger { display: none; } }
        @media (max-width: 767px) { .desktop-nav { display: none; } }
      `}</style>
    </nav>
  );
}
