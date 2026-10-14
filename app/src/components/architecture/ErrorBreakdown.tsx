import { diagnostics, france, limitations, lossByStage, lossByType, lossTotal, truePairsValidation } from '../../content/errors';
import { formatFact } from '../../lib/format';
import { SplitTag } from '../ui/SplitTag';

/** §5, stated as openly as the doc states it. */
export function ErrorBreakdown() {
  return (
    <div className="mt-12 grid gap-5 lg:grid-cols-2">
      <div className="glass rounded-card p-6">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-medium text-ink">Where the remaining F0.5 is lost</h3>
          <SplitTag split={lossTotal.split} />
        </div>
        <ul className="mt-5 space-y-4">
          {lossByType.map((row) => (
            <li key={row.label.text}>
              <div className="flex items-baseline justify-between gap-4 text-sm">
                <span className="text-ink/90">{row.label.text}</span>
                <span className="font-mono text-ink">{formatFact(row.points)}</span>
              </div>
              <div className="mt-1.5 h-2 rounded-full bg-canvas">
                <div
                  className="h-full rounded-full bg-unsure/80"
                  style={{ width: `${(row.points.value / lossTotal.value) * 100}%` }}
                />
              </div>
            </li>
          ))}
        </ul>
        <p className="mt-5 flex justify-between border-t border-line pt-4 text-sm">
          <span className="text-muted">Total (1 − validation F0.5)</span>
          <span className="font-mono text-ink">{formatFact(lossTotal)}</span>
        </p>
      </div>

      <div className="glass rounded-card p-6">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-medium text-ink">
            Where the <span className="font-mono">{formatFact(truePairsValidation)}</span> true pairs end up
          </h3>
          <SplitTag split={truePairsValidation.split} />
        </div>
        <table className="mt-5 w-full text-sm">
          <thead>
            <tr className="text-left text-muted">
              <th className="pb-2 font-normal">stage</th>
              <th className="pb-2 text-right font-normal">v2</th>
              <th className="pb-2 text-right font-normal">final</th>
            </tr>
          </thead>
          <tbody>
            {lossByStage.map((row) => (
              <tr key={row.stage.text} className="border-t border-line">
                <td className="py-2 text-ink/90">{row.stage.text}</td>
                <td className="py-2 text-right font-mono text-muted">{formatFact(row.v2)}</td>
                <td className="py-2 text-right font-mono text-ink">{formatFact(row.final)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="glass rounded-card p-6">
        <h3 className="font-medium text-ink">The hard case: address-less records with shared names</h3>
        <dl className="mt-5 grid grid-cols-2 gap-4 text-sm">
          <div>
            <dt className="text-muted first-letter:uppercase">{diagnostics.addresslessLabel.text}</dt>
            <dd className="mt-1 font-mono text-xl text-unsure">{formatFact(diagnostics.addresslessPrecision)}</dd>
          </div>
          <div>
            <dt className="text-muted">Businesses predicted empty that are true singletons</dt>
            <dd className="mt-1 font-mono text-xl text-ink">{formatFact(diagnostics.emptyAreSingletons)}</dd>
          </div>
          <div>
            <dt className="text-muted">Address-less misses with identical spelling</dt>
            <dd className="mt-1 font-mono text-xl text-ink">{formatFact(diagnostics.identicalSpellingMisses)}</dd>
          </div>
          <div>
            <dt className="text-muted">Sibling evidence helpful vs misleading</dt>
            <dd className="mt-1 font-mono text-xl text-ink">
              {formatFact(diagnostics.siblingHelpful)} vs {formatFact(diagnostics.siblingMisleading)}
            </dd>
          </div>
        </dl>
        <p className="mt-5 text-sm text-muted">{diagnostics.conclusion.text}</p>
      </div>

      <div className="glass rounded-card p-6">
        <h3 className="font-medium text-ink">Limitations</h3>
        <ul className="mt-5 space-y-4 text-sm">
          {limitations.map((l) => (
            <li key={l.title.text}>
              <span className="font-medium text-ink">{l.title.text}</span>{' '}
              <span className="text-muted">{l.text.text}</span>
            </li>
          ))}
        </ul>
        <p className="mt-5 border-t border-line pt-4 text-sm text-muted">{france.gap.text}</p>
      </div>
    </div>
  );
}
