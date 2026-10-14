import { motion } from 'motion/react';
import { provenanceLabels, variantLabels } from '../../content/variants';
import { easeOutExpo } from '../../lib/motion';
import type { ShowcaseExample } from '../../lib/demo/types';
import { Chip } from '../ui/Chip';

/** One business and its records; the records slide in and settle into place. */
export function ShowcaseExampleView({ example, index }: { example: ShowcaseExample; index: number }) {
  const prov = provenanceLabels[example.provenance];
  return (
    <motion.div
      role="tabpanel"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
      className="mt-6 grid gap-6 lg:grid-cols-[1.4fr_1fr] lg:items-center"
    >
      <ul className="grid gap-3 sm:grid-cols-2">
        {example.records.map((r, i) => (
          <motion.li
            key={r.id}
            initial={{ opacity: 0, x: (i % 2 ? 1 : -1) * 40, y: 24, rotate: (i % 2 ? 1 : -1) * 4 }}
            animate={{ opacity: 1, x: 0, y: 0, rotate: 0 }}
            transition={{ duration: 0.8, ease: easeOutExpo, delay: 0.08 * i + 0.05 * (index % 2) }}
            className="glass rounded-xl p-4"
          >
            <div className="flex items-center justify-between gap-2">
              <Chip>{r.source}</Chip>
              <span className="font-mono text-[0.65rem] text-muted">{r.id}</span>
            </div>
            <p className="mt-3 font-medium break-words text-ink">{r.name}</p>
            {r.latin && (
              <p className="mt-1 text-xs text-muted">
                Transliterated: <span className="font-mono text-glow">{r.latin}</span>
              </p>
            )}
            <p className="mt-1 text-sm break-words text-muted">{r.address ?? 'No address'}</p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {r.kinds.map((k) => (
                <Chip key={k} tone={k === 'script' ? 'accent' : 'neutral'}>
                  {k === 'script' && r.script ? r.script : variantLabels[k]}
                </Chip>
              ))}
            </div>
          </motion.li>
        ))}
      </ul>

      <motion.div
        initial={{ opacity: 0, scale: 0.94 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.7, ease: easeOutExpo, delay: 0.25 + 0.06 * example.records.length }}
        className="rounded-card border border-accent/60 bg-surface p-6 shadow-glow"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs font-medium tracking-[0.14em] text-glow uppercase">One business</span>
          <Chip tone={example.provenance === 'verified' ? 'match' : 'accent'} title={prov.long}>
            {prov.short}
          </Chip>
        </div>
        <p className="mt-4 text-h3 font-semibold">{example.s1.name}</p>
        <p className="mt-2 text-sm text-muted">{example.s1.address}</p>
        <dl className="mt-6 grid grid-cols-2 gap-4 border-t border-line pt-5 text-sm">
          <div>
            <dt className="text-muted">Reference ID</dt>
            <dd className="mt-1 font-mono text-xs text-ink">{example.s1.id}</dd>
          </div>
          <div>
            <dt className="text-muted">Country</dt>
            <dd className="mt-1 text-ink">{example.country}</dd>
          </div>
          <div className="col-span-2">
            <dt className="text-muted">Records joined</dt>
            <dd className="mt-1 text-ink">
              <span className="font-mono text-match">{example.records.length}</span> from Sources 2 and 3
            </dd>
          </div>
        </dl>
        <p className="mt-5 text-xs text-muted">{prov.long}.</p>
      </motion.div>
    </motion.div>
  );
}
