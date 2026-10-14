import { Link } from 'react-router';
import { brand } from '../../content/brand';
import { headline } from '../../content/metrics';
import { primaryNav, tryItNav } from '../../content/nav';
import { team, teamName } from '../../content/team';
import { splitLabels } from '../../content/labels';
import { formatFact } from '../../lib/format';
import { Logo } from '../brand/Logo';

const links = [tryItNav, ...primaryNav];

export function Footer() {
  return (
    <footer className="mt-32 border-t border-line">
      <div className="container-page grid gap-12 py-14 md:grid-cols-[1.6fr_1fr_1fr]">
        <div className="max-w-sm">
          <Logo />
          <p className="mt-4 text-muted">{brand.tagline}</p>
          <p className="mt-6 text-sm text-muted">
            {brand.event} · Team {teamName.text}
          </p>
        </div>

        <nav aria-label="Footer">
          <h2 className="text-xs font-medium tracking-[0.14em] text-muted uppercase">Product</h2>
          <ul className="mt-4 space-y-3">
            {links.map((item) => (
              <li key={item.to}>
                <Link to={item.to} className="text-ink/90 transition-colors hover:text-glow">
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        <div>
          <h2 className="text-xs font-medium tracking-[0.14em] text-muted uppercase">Team</h2>
          <ul className="mt-4 space-y-3">
            {team.map((member) => (
              <li key={member.name} className="text-ink/90">
                {member.name}
                {member.role && <span className="text-muted"> · {member.role}</span>}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="border-t border-line">
        <div className="container-page flex flex-col gap-2 py-6 text-xs text-muted sm:flex-row sm:items-center sm:justify-between">
          <p className="flex flex-wrap gap-x-3 gap-y-1">
            <span className="whitespace-nowrap">
              {splitLabels.leaderboard.short} F0.5{' '}
              <span className="font-mono text-ink">{formatFact(headline.leaderboardF05)}</span>
            </span>
            <span className="whitespace-nowrap">
              {splitLabels.validation.short} F0.5{' '}
              <span className="font-mono text-ink">{formatFact(headline.validationF05)}</span>
            </span>
          </p>
          <p>Demo mode replays precomputed results from our real submission.</p>
        </div>
      </div>
    </footer>
  );
}
