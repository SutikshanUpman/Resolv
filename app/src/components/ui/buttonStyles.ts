export type ButtonVariant = 'primary' | 'secondary' | 'ghost';
export type ButtonSize = 'sm' | 'md' | 'lg';

const base =
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-control font-medium tracking-[-0.01em] transition-[transform,background-color,border-color,color,box-shadow] duration-200 ease-out-expo active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50';

const variants: Record<ButtonVariant, string> = {
  primary: 'bg-accent text-white shadow-glow hover:bg-accent-hover hover:-translate-y-px',
  secondary: 'glass text-ink hover:border-line-strong hover:bg-raised/70 hover:-translate-y-px',
  ghost: 'text-muted hover:text-ink hover:bg-raised/60',
};

const sizes: Record<ButtonSize, string> = {
  sm: 'h-9 px-3.5 text-sm',
  md: 'h-11 px-5 text-[0.95rem]',
  lg: 'h-13 px-7 text-base',
};

export function buttonClasses(variant: ButtonVariant = 'primary', size: ButtonSize = 'md', extra = ''): string {
  return [base, variants[variant], sizes[size], extra].filter(Boolean).join(' ');
}
