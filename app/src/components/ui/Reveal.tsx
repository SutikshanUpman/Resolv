import type { ReactNode } from 'react';
import { motion } from 'motion/react';
import { easeOutExpo } from '../../lib/motion';

interface RevealProps {
  children: ReactNode;
  delay?: number;
  className?: string;
  as?: 'div' | 'li' | 'section';
}

/** Fades and lifts its content in once, when it scrolls into view. */
export function Reveal({ children, delay = 0, className, as = 'div' }: RevealProps) {
  const Tag = motion[as];
  return (
    <Tag
      className={className}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '0px 0px -80px 0px' }}
      transition={{ duration: 0.7, ease: easeOutExpo, delay }}
    >
      {children}
    </Tag>
  );
}
