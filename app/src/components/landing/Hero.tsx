import { brand } from '../../content/brand';
import { splitLabels } from '../../content/labels';
import { headline } from '../../content/metrics';
import { tryItNav } from '../../content/nav';
import { formatFact } from '../../lib/format';
import { ButtonLink } from '../ui/ButtonLink';
import { HeroVisual } from './HeroVisual';

export function Hero() {
  return (
    <section className="container-page grid items-center gap-12 pt-16 pb-20 md:pt-24 lg:grid-cols-[1.05fr_1fr] lg:gap-8 lg:pb-28">
      <div>
        <p className="text-sm font-medium tracking-[0.14em] text-glow uppercase">{brand.event}</p>
        <h1 className="mt-5 text-display font-semibold text-balance">{brand.hook}</h1>
        <p className="mt-7 max-w-xl text-lead text-muted">
          {brand.subline} It scored{' '}
          {/* CLAUDE.md allows the leaderboard 0.985978 to be shown as 0.986. */}
          <strong className="font-semibold text-ink">
            {formatFact(headline.leaderboardF05, { digits: 3, rounding: 'nearest' })} macro F0.5
          </strong>{' '}
          on the {brand.event} {splitLabels.leaderboard.short.toLowerCase()}.
        </p>
        <div className="mt-10 flex flex-wrap gap-3">
          <ButtonLink to="/architecture" variant="secondary" size="lg">
            See how it works
          </ButtonLink>
          <ButtonLink to={tryItNav.to} size="lg">
            See it on sample data
          </ButtonLink>
        </div>
      </div>
      <HeroVisual />
    </section>
  );
}
