import type { ReactNode } from 'react';

interface SectionHeadingProps {
  eyebrow: string;
  title: string;
  lead?: ReactNode;
  id?: string;
}

export function SectionHeading({ eyebrow, title, lead, id }: SectionHeadingProps) {
  return (
    <div className="max-w-3xl">
      <p className="text-sm font-medium tracking-[0.14em] text-glow uppercase">{eyebrow}</p>
      <h2 id={id} className="mt-3 text-h2 font-semibold text-balance">
        {title}
      </h2>
      {lead && <p className="mt-5 text-lead text-muted">{lead}</p>}
    </div>
  );
}
