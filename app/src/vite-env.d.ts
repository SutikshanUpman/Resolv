/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 'demo' (default) replays precomputed results; 'api' uses ApiResolver. */
  readonly VITE_RESOLVER?: 'demo' | 'api';
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
