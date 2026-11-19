// Records the self-contained REAL-LOG replay, visibly labelled as non-live.
// Requires Playwright and its ffmpeg binary. No API credentials are accessed.
const { chromium } = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

async function main() {
  const replay = process.argv[2];
  const output = process.argv[3];
  if (!replay || !output) throw new Error('Usage: node record_agent_replay.cjs REPLAY.html OUTPUT.webm');
  const dest = path.resolve(output);
  const build = path.join(path.dirname(dest), '.video-build');
  await fs.mkdir(build, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  const context = await browser.newContext({ viewport: { width:1280, height:720 },
    recordVideo: { dir:build, size:{ width:1280, height:720 } } });
  const page = await context.newPage();
  await page.goto(pathToFileURL(path.resolve(replay)).href);
  const count = await page.evaluate(() => window.screenCount);
  for(let index=0; index<count; index++) {
    await page.evaluate(i => window.show(i), index);
    await page.waitForTimeout(4500);
  }
  await page.screenshot({ path:path.join(build,'preview.png') });
  const video = page.video();
  await context.close();
  await video.saveAs(dest);
  await browser.close();
  console.log(JSON.stringify({video:dest,screens:count,kind:'actual_recording_of_real_log_replay_not_live_API'}));
}
main().catch(error => { console.error(error.message); process.exitCode=1; });
