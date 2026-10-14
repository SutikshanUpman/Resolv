import { MotionConfig } from 'motion/react';
import { Footer } from './Footer';
import { NavBar } from './NavBar';
import { PageTransition } from './PageTransition';
import { SkipLink } from './SkipLink';

export function AppShell() {
  return (
    // reducedMotion="user": with prefers-reduced-motion, Motion drops transform animations and keeps fades.
    <MotionConfig reducedMotion="user">
      <SkipLink />
      <div className="relative flex min-h-dvh flex-col">
        <NavBar />
        <main id="main" tabIndex={-1} className="flex-1 overflow-x-clip outline-none">
          <PageTransition />
        </main>
        <Footer />
      </div>
    </MotionConfig>
  );
}
