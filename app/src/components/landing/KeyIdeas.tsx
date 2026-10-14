import { keyIdeas } from '../../content/keyIdeas';
import { models } from '../../content/models';
import { formatFact } from '../../lib/format';
import { Reveal } from '../ui/Reveal';
import { SectionHeading } from '../ui/SectionHeading';
import { SplitTag } from '../ui/SplitTag';

const [ghost, exclusive, llm] = keyIdeas;

export function KeyIdeas() {
  return (
    <section className="container-page py-24 md:py-32" aria-labelledby="key-ideas">
      <SectionHeading
        id="key-ideas"
        eyebrow="Three key ideas"
        title="What moved our score the most"
        lead="Taken straight from our write-up."
      />
      <div className="mt-14 grid gap-5 lg:grid-cols-3">
        <Reveal className="glass rounded-card p-7">
          <p className="font-mono text-5xl font-medium text-glow">{formatFact(ghost.figure)}</p>
          <p className="mt-1 text-xs text-muted">of train S1 removed as “ghost” businesses</p>
          <h3 className="mt-6 text-h3 font-semibold">{ghost.title.text}</h3>
          <p className="mt-3 text-muted">{ghost.text.text}</p>
        </Reveal>

        <Reveal delay={0.12} className="glass rounded-card p-7">
          <div aria-hidden="true" className="flex items-center gap-2">
            {[0, 1, 2].map((i) => (
              <span key={i} className={`h-3 rounded-full ${i === 1 ? 'w-12 bg-match' : 'w-6 bg-line-strong'}`} />
            ))}
          </div>
          <p className="mt-4 text-xs text-muted">one record, one owner</p>
          <h3 className="mt-6 text-h3 font-semibold">{exclusive.title.text}</h3>
          <p className="mt-3 text-muted">{exclusive.text.text}</p>
        </Reveal>

        <Reveal delay={0.24} className="glass rounded-card p-7">
          <p className="font-mono text-3xl font-medium">
            <span className="text-muted">{formatFact(llm.before)}</span>
            <span className="mx-2 text-muted" aria-label="to">
              →
            </span>
            <span className="text-glow">{formatFact(llm.after)}</span>
          </p>
          <p className="mt-2 flex items-center gap-2 text-xs text-muted">
            the largest single step <SplitTag split={llm.after.split} />
          </p>
          <h3 className="mt-6 text-h3 font-semibold">{llm.title.text}</h3>
          <p className="mt-3 text-muted">
            <span className="font-mono text-sm text-ink">{models.qwen.name.text}</span>, {llm.text.text}
          </p>
        </Reveal>
      </div>
    </section>
  );
}
