import { glossary } from '../../content/overview';

/** Plain definitions of the terms the architecture page uses. */
export function Glossary() {
  return (
    <dl className="mt-12 grid gap-px overflow-hidden rounded-card border border-line bg-line sm:grid-cols-2">
      {glossary.map((g) => (
        <div key={g.term} className="bg-surface p-5">
          <dt className="font-medium text-ink">{g.term}</dt>
          <dd className="mt-2 text-sm text-muted">{typeof g.definition === 'string' ? g.definition : g.definition.text}</dd>
        </div>
      ))}
    </dl>
  );
}
