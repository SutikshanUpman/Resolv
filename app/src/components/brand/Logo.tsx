import { Link } from 'react-router';
import { brand } from '../../content/brand';
import { LogoMark } from './LogoMark';

export function Logo() {
  return (
    <Link
      to="/"
      className="group inline-flex items-center gap-2.5 rounded-control text-ink"
      aria-label={`${brand.name}, home`}
    >
      <LogoMark className="size-7 transition-transform duration-300 ease-out-expo group-hover:scale-110" />
      <span className="text-lg font-semibold tracking-[-0.03em]">{brand.name}</span>
    </Link>
  );
}
