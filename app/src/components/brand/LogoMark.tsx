/** Three source streams converging on one entity. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 28 28" fill="none" aria-hidden="true" className={className}>
      <g stroke="var(--color-ink)" strokeWidth="2" strokeLinecap="round">
        <path d="M3 6c6 0 8 8 13 8" />
        <path d="M3 14h13" />
        <path d="M3 22c6 0 8-8 13-8" />
      </g>
      <circle cx="21" cy="14" r="4.5" fill="var(--color-accent)" />
    </svg>
  );
}
