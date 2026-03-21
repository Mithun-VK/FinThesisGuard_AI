import type { AuthorityType } from './types';

export const COLORS = {
  primary: '#E63946',
  darkRed: '#B52A35',
  bg: '#0A0A0A',
  card: '#1A1A1A',
  border: '#2A2A2A',
  textPrimary: '#F1F1F1',
  textSecondary: '#A0A0A0',
  riskHigh: '#FF4444',
  riskMedium: '#FFA500',
  riskLow: '#22C55E',
  success: '#22C55E',
  warning: '#EAB308',
  danger: '#E63946',
} as const;

export const EXAMPLE_QUERIES = [
  'What is HDFC Bank\'s NIM trend over the last 4 quarters?',
  'Summarize SEBI\'s latest guidelines on algo trading for retail investors',
  'What are the NRI tax implications for dividend income from Indian equities?',
  'Compare RBI\'s repo rate decisions in 2024 vs 2023',
  'What are the key risks in Reliance Industries\' O2C segment?',
  'How does Zomato\'s unit economics compare to Swiggy?',
] as const;

export const EXAMPLE_THESES = {
  nvidia: `NVIDIA will significantly outperform the broad market over the next 24 months, driven by:
1. Explosive AI infrastructure demand from hyperscalers (Microsoft, Google, AWS, Meta) spending $150B+ capex in 2024
2. Data center GPU monopoly with 80%+ market share in AI training chips (H100/H200/B100)
3. CUDA ecosystem moat creating 10+ year switching costs for ML engineers
4. Gross margins expanding to 75%+ as HBM memory supply normalizes
5. NIM software platform creating a recurring revenue stream beyond hardware

Core assumption: AI capex will sustain at current levels through 2026 as hyperscalers race for AGI infrastructure.`,

  hdfcBank: `HDFC Bank will deliver 18-22% earnings CAGR over the next 3 years, supported by:
1. Post-merger HDFC Ltd integration synergies of ₹5,000+ crore annually by FY26
2. NIM recovery from 3.4% to 3.8% as high-cost merger liabilities mature
3. Retail loan book growth of 20%+ driven by home loans, credit cards, and auto loans
4. Improving CASA ratio from 38% to 42% through branch expansion in Tier 2/3 cities
5. Asset quality remaining pristine with GNPA below 1.5% and PCR above 75%

Core assumption: RBI will cut rates by 75-100bps by end of FY26, supporting NIM expansion.`,
} as const;

export const AUTHORITY_STYLES: Record<AuthorityType, { bg: string; text: string; border: string }> = {
  rbi:   { bg: 'rgba(230, 57, 70, 0.15)', text: '#E63946', border: '#E63946' },
  sebi:  { bg: 'rgba(59, 130, 246, 0.15)', text: '#3B82F6', border: '#3B82F6' },
  mca:   { bg: 'rgba(234, 179, 8, 0.15)', text: '#EAB308', border: '#EAB308' },
  taxlaw: { bg: 'rgba(147, 51, 234, 0.15)', text: '#A855F7', border: '#A855F7' },
  annualreport: { bg: 'rgba(34, 197, 94, 0.15)', text: '#22C55E', border: '#22C55E' },
  earningstranscript: { bg: 'rgba(14, 165, 233, 0.15)', text: '#0EA5E9', border: '#0EA5E9' },
  brokerresearch: { bg: 'rgba(249, 115, 22, 0.15)', text: '#F97316', border: '#F97316' },
  news: { bg: 'rgba(148, 163, 184, 0.15)', text: '#94A3B8', border: '#94A3B8' },
  blog: { bg: 'rgba(148, 163, 184, 0.15)', text: '#94A3B8', border: '#94A3B8' },
  unknown: { bg: 'rgba(148, 163, 184, 0.15)', text: '#94A3B8', border: '#94A3B8' },
};

export const RISK_THRESHOLDS = {
  high: 7,
  medium: 4,
} as const;

export const NAV_LINKS = [
  { href: '/', label: 'RAG Query' },
  { href: '/thesis', label: 'Thesis Validator' },
] as const;

export const CONFIDENCE_STYLES = {
  High: {
    bg: '#052e16',
    text: '#22C55E',
    border: '#166534',
    label: 'High Confidence',
  },
  Medium: {
    bg: '#1c1400',
    text: '#EAB308',
    border: '#713f12',
    label: 'Medium Confidence',
  },
  Low: {
    bg: '#2A0A0A',
    text: '#E63946',
    border: '#7f1d1d',
    label: 'Low Confidence',
  },
} as const;
