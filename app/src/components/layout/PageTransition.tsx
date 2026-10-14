import { useState, type ReactNode } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { useLocation, useOutlet } from 'react-router';
import { pageEnter, pageExit } from '../../lib/motion';

/** Keeps the page that is leaving on screen during its exit animation, instead of swapping in the next one. */
function FrozenOutlet({ element }: { element: ReactNode }) {
  const [frozen] = useState(element);
  return frozen;
}

function onPageChange() {
  // With a #target in the address, the new page scrolls to it; otherwise start at the top.
  if (!window.location.hash) window.scrollTo(0, 0);
  document.getElementById('main')?.focus({ preventScroll: true });
}

export function PageTransition() {
  const location = useLocation();
  const element = useOutlet();

  return (
    <AnimatePresence mode="wait" initial={false} onExitComplete={onPageChange}>
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0, transition: pageEnter }}
        exit={{ opacity: 0, y: -6, transition: pageExit }}
      >
        <FrozenOutlet element={element} />
      </motion.div>
    </AnimatePresence>
  );
}
