import { defineConfig, searchForWorkspaceRoot, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { brand } from './src/content/brand.ts';

/** Fills %BRAND_NAME% and %BRAND_DESCRIPTION% in index.html, so the name lives only in brand.ts. */
function brandHtml(): Plugin {
  return {
    name: 'brand-html',
    transformIndexHtml: (html) =>
      html.replaceAll('%BRAND_NAME%', brand.name).replaceAll('%BRAND_DESCRIPTION%', brand.description),
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), brandHtml()],
  // The docs page imports Documentation_template.md from the sibling code folder.
  server: { fs: { allow: [searchForWorkspaceRoot(process.cwd()), '../code'] } },
});
