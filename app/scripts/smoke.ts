/**
 * Opens every route in the installed Chrome at three widths, records console errors and failed requests,
 * checks keyboard basics, and saves full-page screenshots to .checks/ for review.
 *
 *   npm run build && npx vite preview --port 4173 &   then   npm run smoke
 *   SMOKE_URL=http://localhost:5173 npm run smoke     (against the dev server)
 */
import { mkdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { chromium, type Page } from 'playwright';
import { APP_DIR } from './lib/paths.ts';

const BASE = process.env.SMOKE_URL ?? 'http://localhost:4173';
const OUT = join(APP_DIR, '.checks');
const ROUTES = ['/', '/architecture', '/architecture/docs', '/app', '/app/run', '/app/results', '/no-such-page'];
const WIDTHS = [360, 768, 1440];

const problems: string[] = [];

async function visit(page: Page, route: string, width: number): Promise<void> {
  const errors: string[] = [];
  page.removeAllListeners('console');
  page.removeAllListeners('requestfailed');
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('requestfailed', (r) => errors.push(`request failed: ${r.url()}`));
  await page.setViewportSize({ width, height: 900 });
  await page.goto(BASE + route, { waitUntil: 'networkidle' });
  await page.waitForTimeout(600);

  const title = await page.title();
  const h1 = await page.locator('h1').count();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  if (h1 !== 1) problems.push(`${route} @${width}: ${h1} h1 elements`);
  if (overflow > 0) problems.push(`${route} @${width}: horizontal overflow of ${overflow}px`);
  for (const e of errors) problems.push(`${route} @${width}: ${e}`);
  const name = `${route === '/' ? 'home' : route.slice(1).replaceAll('/', '_')}@${width}.png`;
  await page.screenshot({ path: join(OUT, name), fullPage: true });
  console.log(`${route.padEnd(20)} ${String(width).padStart(5)}px  "${title}"  h1=${h1}  overflow=${overflow}`);
}

async function keyboard(page: Page): Promise<void> {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.keyboard.press('Tab');
  const first = await page.evaluate(() => document.activeElement?.textContent?.trim());
  if (first !== 'Skip to content') problems.push(`keyboard: first Tab lands on "${first}", expected the skip link`);
  await page.screenshot({ path: join(OUT, 'keyboard-skip-link.png') });
  await page.keyboard.press('Tab');
  await page.keyboard.press('Tab');
  await page.screenshot({ path: join(OUT, 'keyboard-focus.png') });

  // Mobile menu: opens, moves focus into the panel, closes on Escape and returns focus.
  await page.setViewportSize({ width: 360, height: 800 });
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: 'Open menu' }).click();
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(OUT, 'mobile-menu-open.png') });
  const focused = await page.evaluate(() => document.activeElement?.textContent?.trim());
  if (focused !== 'How it works') problems.push(`mobile menu: focus went to "${focused}"`);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(300);
  const back = await page.evaluate(() => document.activeElement?.getAttribute('aria-label'));
  if (back !== 'Open menu') problems.push(`mobile menu: focus after Escape is on "${back}"`);

  // Client-side navigation with the page transition.
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.getByRole('link', { name: 'See how it works' }).click();
  await page.waitForURL('**/architecture');
  await page.waitForTimeout(800);
  const heading = await page.locator('h1').innerText();
  if (heading !== 'How it works') problems.push(`navigation: /architecture shows "${heading}"`);
}

/** Landing to downloaded files: sample data → Refine → Skip → results → preview → every download format. */
async function workspaceFlow(page: Page): Promise<void> {
  const fail = (msg: string) => problems.push(`workspace: ${msg}`);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.getByRole('link', { name: 'Try it' }).first().click();
  await page.waitForURL('**/app');
  await page.getByRole('button', { name: 'Load sample data' }).click();
  await page.getByText('Sample data is ready.').waitFor({ timeout: 20000 });
  await page.screenshot({ path: join(OUT, 'flow-1-upload.png'), fullPage: true });

  await page.getByRole('button', { name: 'Refine', exact: true }).click();
  await page.waitForURL('**/app/run');
  await page.waitForTimeout(2500);
  await page.screenshot({ path: join(OUT, 'flow-2-run.png'), fullPage: true });
  await page.getByRole('button', { name: 'Skip to results' }).click();
  await page.waitForURL('**/app/results', { timeout: 20000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(OUT, 'flow-3-results.png'), fullPage: true });

  if ((await page.getByText('PASS', { exact: true }).count()) !== 1) fail('validator did not show PASS');

  // The result survives a reload (sessionStorage).
  const stored = await page.evaluate(() => sessionStorage.getItem('resolv.lastResult.v1')?.length ?? 0);
  await page.reload({ waitUntil: 'networkidle' });
  await page.getByText('PASS', { exact: true }).waitFor({ timeout: 15000 }).catch(() => fail('results did not survive a reload'));
  console.log(`reload: results restored (${stored.toLocaleString('en-US')} characters in sessionStorage)`);
  // The Safari path: only IDs fit in storage, so the records are rebuilt from the sample files.
  await page.evaluate(() => {
    const key = 'resolv.lastResult.v1';
    const saved = JSON.parse(sessionStorage.getItem(key) ?? '{}') as Record<string, unknown>;
    sessionStorage.setItem(key, JSON.stringify({ ...saved, records: null }));
  });
  await page.reload({ waitUntil: 'networkidle' });
  await page.getByText('PASS', { exact: true }).waitFor({ timeout: 15000 }).catch(() => fail('results did not rebuild from the sample files'));
  const explorerName = await page.locator('#explorer').count();
  console.log(`reload: rebuilt from IDs only${explorerName ? '' : ' (explorer missing)'}`);
  if ((await page.getByText('Demo mode: precomputed results').count()) < 1) fail('demo-mode badge missing');

  await page.getByRole('button', { name: 'Preview' }).first().click();
  const dialog = page.getByRole('dialog');
  await dialog.waitFor();
  await dialog.getByRole('searchbox').fill('France');
  await page.waitForTimeout(300);
  await dialog.getByRole('columnheader', { name: /record_name/ }).click();
  await page.screenshot({ path: join(OUT, 'flow-4-preview.png') });
  await page.keyboard.press('Escape');

  const downloads = join(OUT, 'downloads');
  mkdirSync(downloads, { recursive: true });
  const formats: [RegExp, string][] = [
    [/^CSV$/, 'text'],
    [/^Excel/, 'PK'],
    [/^PDF/, '%PDF'],
    [/^Word/, 'PK'],
    [/^Challenge format/, 'source1_entity_id\tcandidate_entity_ids'],
  ];
  for (const [label, magic] of formats) {
    await page.getByRole('button', { name: 'Download' }).first().click();
    const wait = page.waitForEvent('download', { timeout: 60000 });
    await page.getByRole('menuitem', { name: label }).click();
    const dl = await wait;
    const path = join(downloads, dl.suggestedFilename());
    await dl.saveAs(path);
    const head = readFileSync(path).subarray(0, 64).toString('utf8');
    const size = statSync(path).size;
    const ok = magic === 'text' ? head.includes('source1_entity_id') : head.startsWith(magic);
    console.log(`download ${dl.suggestedFilename().padEnd(22)} ${String(size).padStart(9)} bytes ${ok ? 'ok' : 'UNEXPECTED CONTENT'}`);
    if (!ok || size < 200) fail(`${dl.suggestedFilename()} looks wrong (${size} bytes)`);
  }

  // Results with data at phone width.
  await page.setViewportSize({ width: 360, height: 800 });
  await page.waitForTimeout(400);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  if (overflow > 0) fail(`results @360: horizontal overflow of ${overflow}px`);
  await page.screenshot({ path: join(OUT, 'flow-results@360.png'), fullPage: true });
  await page.setViewportSize({ width: 1440, height: 900 });

  await page.getByLabel('Country').selectOption('France');
  await page.waitForTimeout(300);
  await page.locator('#explorer').scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(OUT, 'flow-5-explorer.png') });

  // Honest demo: a visitor's own files are validated for real, then stopped before any "inference".
  await page.goto(BASE + '/app', { waitUntil: 'networkidle' });
  await page.locator('input[type=file]').first().setInputFiles({ name: 'bad.csv', mimeType: 'text/csv', buffer: Buffer.from('id,name\n1,Acme\n') });
  await page.getByText(/Missing columns/).waitFor({ timeout: 10000 });
  const sample = (n: number) => readFileSync(join(APP_DIR, 'public', 'demo', `sample_source${n}.tsv`));
  for (const n of [1, 2, 3]) {
    await page.locator('input[type=file]').nth(n - 1).setInputFiles({ name: `mine_${n}.tsv`, mimeType: 'text/tab-separated-values', buffer: sample(n) });
  }
  await page.getByText('All three files passed the checks.').waitFor({ timeout: 20000 });
  await page.getByRole('button', { name: 'Refine', exact: true }).click();
  await page.getByText('Live inference isn’t connected').waitFor({ timeout: 5000 });
  if (!page.url().endsWith('/app')) fail('uploading own files navigated away from /app');
  await page.screenshot({ path: join(OUT, 'flow-6-own-files.png'), fullPage: true });
}

/** A heading counts as landed when it sits just under the sticky nav, or the page cannot scroll further. */
async function landedOn(page: Page, id: string): Promise<string | null> {
  return page.evaluate((target) => {
    const el = document.getElementById(target);
    if (!el) return `no element with id "${target}"`;
    const top = el.getBoundingClientRect().top;
    const atBottom = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2;
    return (top >= 0 && top <= 160) || (atBottom && top >= 0 && top < window.innerHeight) ? null : `heading top at ${Math.round(top)}px`;
  }, id);
}

/** Clicks every "Read more" link on /architecture and every contents link on the docs page. */
async function linkChecks(page: Page): Promise<void> {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(BASE + '/architecture', { waitUntil: 'networkidle' });
  const stages = await page.locator('ol > li > button[aria-controls^="stage-"]').count();
  let readMore = 0;
  for (let i = 0; i < stages; i++) {
    await page.goto(BASE + '/architecture', { waitUntil: 'networkidle' });
    const button = page.locator('ol > li > button[aria-controls^="stage-"]').nth(i);
    if ((await button.getAttribute('aria-expanded')) !== 'true') await button.click();
    const panel = page.locator(`#${await button.getAttribute('aria-controls')}`);
    const links = await panel.locator('a[href*="/architecture/docs#"]').count();
    for (let j = 0; j < links; j++) {
      if (j > 0) {
        await page.goto(BASE + '/architecture', { waitUntil: 'networkidle' });
        const again = page.locator('ol > li > button[aria-controls^="stage-"]').nth(i);
        if ((await again.getAttribute('aria-expanded')) !== 'true') await again.click();
      }
      const link = page.locator(`#${await button.getAttribute('aria-controls')} a[href*="/architecture/docs#"]`).nth(j);
      const href = (await link.getAttribute('href')) ?? '';
      await link.click();
      await page.waitForURL('**/architecture/docs#*');
      await page.waitForTimeout(900);
      const miss = await landedOn(page, href.split('#')[1] ?? '');
      readMore++;
      if (miss) problems.push(`read-more link ${href}: ${miss}`);
    }
  }

  await page.goto(BASE + '/architecture/docs', { waitUntil: 'networkidle' });
  const toc = page.locator('nav[aria-label="Table of contents"] a:visible');
  const count = await toc.count();
  for (let i = 0; i < count; i++) {
    const href = (await toc.nth(i).getAttribute('href')) ?? '';
    await toc.nth(i).click();
    await page.waitForTimeout(250);
    const miss = await landedOn(page, href.slice(1));
    if (miss) problems.push(`contents link ${href}: ${miss}`);
  }
  console.log(`links: clicked ${readMore} read-more links and ${count} contents links`);
}

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ channel: 'chrome' });
  const page = await browser.newPage({ acceptDownloads: true });
  for (const route of ROUTES) for (const width of WIDTHS) await visit(page, route, width);
  await keyboard(page);
  await linkChecks(page);
  await workspaceFlow(page);

  const reduced = await browser.newPage({ reducedMotion: 'reduce' });
  await visit(reduced, '/', 1440);
  await browser.close();

  if (problems.length > 0) {
    console.error(`\n${problems.length} problem(s):`);
    for (const p of problems) console.error(`  ✗ ${p}`);
    process.exit(1);
  }
  console.log(`\nAll routes rendered without console errors. Screenshots in ${OUT}`);
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
