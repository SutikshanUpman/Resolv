import { steps } from '../../content/landing';
import { Reveal } from '../ui/Reveal';
import { SectionHeading } from '../ui/SectionHeading';

export function HowItWorks() {
  return (
    <section className="container-page py-24 md:py-32" aria-labelledby="how-it-works">
      <SectionHeading id="how-it-works" eyebrow="How it works" title="You give it three files, it gives you the matches" />
      <ol className="mt-14 grid gap-5 md:grid-cols-3">
        {steps.map((step, i) => (
          <Reveal as="li" key={step.title} delay={i * 0.12} className="glass relative overflow-hidden rounded-card p-7">
            <span className="grid size-10 place-items-center rounded-full border-2 border-ink font-mono text-sm text-ink">
              {i + 1}
            </span>
            <h3 className="mt-6 text-h3 font-semibold">{step.title}</h3>
            <p className="mt-3 text-muted">{step.text}</p>
          </Reveal>
        ))}
      </ol>
    </section>
  );
}
