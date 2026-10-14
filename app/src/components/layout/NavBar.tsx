import { NavLink } from 'react-router';
import { primaryNav, tryItNav } from '../../content/nav';
import { Logo } from '../brand/Logo';
import { ButtonLink } from '../ui/ButtonLink';
import { MobileNav } from './MobileNav';

export function NavBar() {
  return (
    <header className="sticky top-0 z-50 h-(--nav-height) border-b border-line bg-canvas/95">
      <nav aria-label="Primary" className="container-page flex h-full items-center justify-between gap-4">
        <Logo />
        <div className="flex items-center gap-1 md:gap-2">
          <ul className="hidden items-center gap-1 md:flex">
            {primaryNav.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    `relative rounded-control px-3 py-2 text-sm transition-colors ${
                      isActive ? 'text-ink' : 'text-muted hover:text-ink'
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      {item.label}
                      {isActive && (
                        <span
                          aria-hidden="true"
                          className="absolute inset-x-3 -bottom-px h-0.5 bg-accent"
                        />
                      )}
                    </>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
          <ButtonLink to={tryItNav.to} size="sm" className="ml-1">
            {tryItNav.label}
          </ButtonLink>
          <MobileNav />
        </div>
      </nav>
    </header>
  );
}
