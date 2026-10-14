export interface NavItem {
  readonly label: string;
  readonly to: string;
  /** Match the path exactly, so /architecture is not active on /architecture/docs. */
  readonly end?: boolean;
}

export const primaryNav: readonly NavItem[] = [
  { label: 'How it works', to: '/architecture', end: true },
  { label: 'Docs', to: '/architecture/docs' },
];

export const tryItNav: NavItem = { label: 'Try it', to: '/app' };
