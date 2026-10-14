/** Shown only while the first page chunk loads: the bare canvas, so nothing flashes. */
export function ShellFallback() {
  return <div className="min-h-dvh bg-canvas" aria-busy="true" />;
}
