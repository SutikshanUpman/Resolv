import type { ReactNode } from 'react';

type Tone = 'neutral' | 'accent' | 'match' | 'unsure' | 'danger';

const tones: Record<Tone, string> = {
  neutral: 'border-line text-muted',
  accent: 'border-accent/40 bg-accent/10 text-glow',
  match: 'border-match/30 bg-match/10 text-match',
  unsure: 'border-unsure/30 bg-unsure/10 text-unsure',
  danger: 'border-danger/30 bg-danger/10 text-danger',
};

interface ChipProps {
  children: ReactNode;
  tone?: Tone;
  title?: string;
  className?: string;
}

export function Chip({ children, tone = 'neutral', title, className = '' }: ChipProps) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[0.7rem] leading-5 font-medium whitespace-nowrap ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}
