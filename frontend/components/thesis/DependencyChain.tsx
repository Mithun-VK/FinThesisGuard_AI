'use client';

import { useState } from 'react';
import { ArrowRight } from 'lucide-react';
import type { RiskScore, DependencyNode } from '@/lib/types';

interface DependencyChainProps {
  chain: string[];
  nodes?: DependencyNode[];
  risks?: RiskScore[];
}

function getRiskColor(score: number): string {
  if (score >= 7) return '#FF4444';
  if (score >= 4) return '#FFA500';
  return '#22C55E';
}

export default function DependencyChain({
  chain,
  nodes = [],
  risks = [],
}: DependencyChainProps) {
  const [hoveredNode, setHoveredNode] = useState<number | null>(null);

  if (!chain || chain.length === 0) return null;

  // Try to match a RiskScore entry to this node label
  const getNodeRiskScore = (label: string): number => {
    const trimmed = label.toLowerCase().replace(/\s+/g, '');
    const match =
      risks.find(r =>
        trimmed.includes(r.dimension.replace('risk', '').toLowerCase()),
      ) ?? risks.find(r =>
        r.dimension_label.toLowerCase().includes(label.toLowerCase()),
      );
    return match ? match.score : 3;
  };

  return (
    <div
      style={{
        background: '#1A1A1A',
        border: '1px solid #2A2A2A',
        borderRadius: '12px',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          padding: '16px 20px',
          borderBottom: '1px solid #2A2A2A',
        }}
      >
        <h3
          style={{
            fontSize: '15px',
            fontWeight: '600',
            color: '#F1F1F1',
            margin: 0,
          }}
        >
          Dependency Chain
        </h3>
      </div>

      <div style={{ padding: '20px' }}>
        {/* Desktop: horizontal flow */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '4px',
          }}
          className="dep-chain-desktop"
        >
          {chain.map((label, i) => {
            const riskScore = getNodeRiskScore(label);
            const riskColor = getRiskColor(riskScore);
            const isHovered = hoveredNode === i;
            const isWeakLink = riskScore >= 7;

            return (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                {/* Node */}
                <div
                  onMouseEnter={() => setHoveredNode(i)}
                  onMouseLeave={() => setHoveredNode(null)}
                  style={{
                    position: 'relative',
                    background: '#0A0A0A',
                    border: `2px solid ${
                      isWeakLink ? '#E63946' : isHovered ? riskColor : '#2A2A2A'
                    }`,
                    borderRadius: '10px',
                    padding: '10px 16px',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    boxShadow: isHovered
                      ? `0 0 12px ${riskColor}44`
                      : 'none',
                    minWidth: '90px',
                    textAlign: 'center',
                  }}
                >
                  <p
                    style={{
                      fontSize: '13px',
                      fontWeight: '600',
                      color: '#F1F1F1',
                      margin: '0 0 2px',
                    }}
                  >
                    {label}
                  </p>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '3px',
                    }}
                  >
                    <div
                      style={{
                        width: '6px',
                        height: '6px',
                        borderRadius: '50%',
                        background: riskColor,
                      }}
                    />
                    <span
                      style={{
                        fontSize: '10px',
                        color: riskColor,
                      }}
                    >
                      Risk {riskScore}
                    </span>
                  </div>

                  {/* Tooltip */}
                  {isHovered && (
                    <div
                      style={{
                        position: 'absolute',
                        bottom: '110%',
                        left: '50%',
                        transform: 'translateX(-50%)',
                        background: '#2A2A2A',
                        border: `1px solid ${riskColor}`,
                        borderRadius: '6px',
                        padding: '8px 12px',
                        fontSize: '11px',
                        whiteSpace: 'nowrap',
                        color: '#F1F1F1',
                        zIndex: 10,
                      }}
                    >
                      {isWeakLink
                        ? '⚠️ High-risk dependency'
                        : `Risk Score: ${riskScore}/10`}
                    </div>
                  )}
                </div>

                {/* Animated arrow between nodes */}
                {i < chain.length - 1 && (
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      color: '#E63946',
                    }}
                    className="animate-flow-right"
                  >
                    <ArrowRight size={20} strokeWidth={2.5} />
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Legend */}
        <div
          style={{
            display: 'flex',
            gap: '16px',
            marginTop: '16px',
            paddingTop: '12px',
            borderTop: '1px solid #2A2A2A',
            flexWrap: 'wrap',
          }}
        >
          {[
            { color: '#22C55E', label: 'Strong link (low risk)' },
            { color: '#FFA500', label: 'Moderate risk' },
            { color: '#FF4444', label: 'Weak link (high risk)' },
          ].map(({ color, label }) => (
            <div
              key={label}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <div
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  background: color,
                }}
              />
              <span
                style={{
                  fontSize: '11px',
                  color: '#A0A0A0',
                }}
              >
                {label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
