import { useEffect, useRef } from 'react';
import { animate, useInView, useReducedMotion } from 'motion/react';

interface CountUpProps {
  /** Target value, already rounded as it should be shown. */
  value: number;
  decimals: number;
  /** Exact final text (from formatFact), so the last frame never differs from the doc. */
  final: string;
  className?: string;
}

const fmt = (n: number, d: number) =>
  new Intl.NumberFormat('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }).format(n);

export function CountUp({ value, decimals, final, className }: CountUpProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: '0px 0px -60px 0px' });
  const reduced = useReducedMotion();

  useEffect(() => {
    const el = ref.current;
    if (!el || !inView) return;
    if (reduced) {
      el.textContent = final;
      return;
    }
    const controls = animate(0, value, {
      duration: 1.6,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (n) => (el.textContent = fmt(Math.min(n, value), decimals)),
      onComplete: () => (el.textContent = final),
    });
    return () => controls.stop();
  }, [inView, reduced, value, decimals, final]);

  // Server/first render shows the true value, so the number is right even without JavaScript animation.
  return (
    <span ref={ref} className={className} aria-label={final}>
      {final}
    </span>
  );
}
