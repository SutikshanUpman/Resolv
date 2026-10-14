import { useMemo } from 'react';
import type { ResolveResult } from '../../lib/resolver/types';
import { validateResult } from '../../lib/validator';
import { Chip } from '../ui/Chip';

/** The official submission rules, checked in the browser on the two result files. */
export function ValidatorPanel({ result }: { result: ResolveResult }) {
  const checks = useMemo(() => validateResult(result), [result]);
  const passed = checks.every((c) => c.pass);
  return (
    <section aria-labelledby="validator" className="glass rounded-card p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 id="validator" className="font-semibold text-ink">
            Validator
          </h2>
          <p className="text-sm text-muted">The challenge’s submission rules, checked in your browser on both files.</p>
        </div>
        <Chip tone={passed ? 'match' : 'danger'}>{passed ? 'PASS' : 'FAIL'}</Chip>
      </div>
      <ul className="mt-5 grid gap-2 md:grid-cols-2">
        {checks.map((c) => (
          <li key={c.rule} className="flex gap-3 rounded-lg border border-line bg-canvas/50 p-3 text-sm">
            <span aria-hidden="true" className={c.pass ? 'text-match' : 'text-danger'}>
              {c.pass ? '✓' : '✗'}
            </span>
            <span className="min-w-0">
              <span className="block text-ink/90">
                <span className="sr-only">{c.pass ? 'Passed: ' : 'Failed: '}</span>
                {c.rule}
              </span>
              <span className="block text-xs text-muted">{c.detail}</span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
