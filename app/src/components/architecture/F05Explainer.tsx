import { breakEven, decision, fBeta } from '../../content/decision';
import { dataset } from '../../content/dataset';
import { formatFact } from '../../lib/format';
import { SplitTag } from '../ui/SplitTag';

/** Why precision counts double, and why a wrong match on a singleton costs a full point (§2.1, §4.4). */
export function F05Explainer() {
  return (
    <div className="mt-12 grid gap-5 lg:grid-cols-3 [&>*]:min-w-0">
      <div className="glass rounded-card p-6">
        <p className="text-xs font-medium tracking-[0.14em] text-glow uppercase">The metric</p>
        <p className="mt-4 font-mono text-sm text-ink">{fBeta.formula}</p>
        <p className="mt-4 text-muted">
          {fBeta.precisionWeight.text} It is computed for each business, then averaged over all of them.
        </p>
      </div>
      <div className="glass rounded-card p-6">
        <p className="text-xs font-medium tracking-[0.14em] text-glow uppercase">Singletons</p>
        <p className="mt-4 text-ink/90">{fBeta.singletonRule.text}</p>
        <p className="mt-3 text-muted">{fBeta.singletonCost.text}</p>
        <p className="mt-4 flex flex-wrap items-center gap-2 text-sm text-muted">
          Singletons are <span className="font-mono text-ink">{formatFact(dataset.singletonShareTrain)}</span> of businesses
          <SplitTag split={dataset.singletonShareTrain.split} />
        </p>
      </div>
      <div className="glass rounded-card p-6">
        <p className="text-xs font-medium tracking-[0.14em] text-glow uppercase">Why a high threshold</p>
        <p className="mt-4 text-muted">
          With n true records and n − 1 already found, adding one more uncertain record only helps if it is right at least
          this often:
        </p>
        <table className="mt-4 w-full text-sm">
          <thead>
            <tr className="text-left text-muted">
              <th className="py-1 font-normal">true records n</th>
              <th className="py-1 text-right font-normal">break-even precision</th>
            </tr>
          </thead>
          <tbody>
            {breakEven.map((b) => (
              <tr key={b.n} className="border-t border-line">
                <td className="py-1.5 font-mono">{b.n}</td>
                <td className="py-1.5 text-right font-mono text-ink">{formatFact(b.precision)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-4 text-sm text-muted">
          So a match is kept only at p ≥ <span className="font-mono text-ink">{formatFact(decision.threshold)}</span>.
        </p>
      </div>
    </div>
  );
}
