import { team, teamName } from '../../content/team';
import { brand } from '../../content/brand';
import { Reveal } from '../ui/Reveal';
import { SectionHeading } from '../ui/SectionHeading';

const initials = (name: string) =>
  name
    .split(' ')
    .map((p) => p[0])
    .join('');

export function TeamSection() {
  return (
    <section className="container-page py-24 md:py-32" aria-labelledby="team">
      <SectionHeading id="team" eyebrow="Team" title={teamName.text} lead={`Built for the ${brand.event}.`} />
      <ul className="mt-12 grid grid-cols-2 gap-4 md:grid-cols-4">
        {team.map((m, i) => (
          <Reveal as="li" key={m.name} delay={i * 0.08} className="glass rounded-card p-6">
            <span
              aria-hidden="true"
              className="grid size-12 place-items-center rounded-full bg-ink font-semibold text-canvas"
            >
              {initials(m.name)}
            </span>
            <p className="mt-5 font-medium text-ink">{m.name}</p>
            {m.role && <p className="mt-1 text-sm text-muted">{m.role}</p>}
            {(m.github || m.linkedin) && (
              <p className="mt-3 flex gap-3 text-sm">
                {m.github && (
                  <a href={m.github} className="text-glow hover:underline" target="_blank" rel="noreferrer">
                    GitHub
                  </a>
                )}
                {m.linkedin && (
                  <a href={m.linkedin} className="text-glow hover:underline" target="_blank" rel="noreferrer">
                    LinkedIn
                  </a>
                )}
              </p>
            )}
          </Reveal>
        ))}
      </ul>
    </section>
  );
}
