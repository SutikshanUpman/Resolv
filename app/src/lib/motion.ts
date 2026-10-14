import type { Transition } from 'motion/react';

/** Mirrors --ease-out-expo in src/styles/tokens.css. */
export const easeOutExpo = [0.16, 1, 0.3, 1] as const;
const easeIn = [0.4, 0, 1, 1] as const;

export const pageEnter: Transition = { duration: 0.45, ease: easeOutExpo };
export const pageExit: Transition = { duration: 0.16, ease: easeIn };
