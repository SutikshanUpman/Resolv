import type { ReactNode } from 'react';

interface PageHeaderProps {
  eyebrow: string;
  title: string;
  lead?: ReactNode;
}

export function PageHeader({ eyebrow, title, lead }: PageHeaderProps) {
  return (
    <header className="container-page pt-20 pb-12 md:pt-28">
      <p className="text-sm font-medium tracking-[0.14em] text-glow uppercase">{eyebrow}</p>
      <h1 className="mt-4 max-w-4xl text-h1 font-semibold text-balance">{title}</h1>
      {lead && <p className="mt-6 max-w-2xl text-lead text-muted">{lead}</p>}
    </header>
  );
}
