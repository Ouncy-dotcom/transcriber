// End-to-end test of the real app window: picks a file, transcribes it and checks the result.
// Usage: node test/app-test.mjs [path-to-Transcriber.exe]   (no argument = run from source)
import { _electron as electron } from 'playwright';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const exe = process.argv[2];
const audio = path.join(here, 'audio', 'interview-2-speakers.wav');
const shots = path.join(here, '..', 'test-output'); fs.mkdirSync(shots, { recursive: true });
const args = exe ? [] : [path.join(here, '..')];
if (process.platform === 'linux') args.push('--no-sandbox');

const t0 = Date.now();
const app = await electron.launch({ executablePath: exe || undefined, args, timeout: 120000 });
app.process().stdout?.on('data', (d) => process.stdout.write(d));
app.process().stderr?.on('data', (d) => process.stderr.write(d));
const page = await app.firstWindow();
page.on('pageerror', (e) => console.error('PAGE ERROR', e));
await page.waitForSelector('#drop');
console.log('window open after', ((Date.now() - t0) / 1000).toFixed(1), 's, title:', await page.title());
await page.setInputFiles('#pick', audio);
await page.click('#go');
let status = '', last = '';
for (let i = 0; i < 1800; i++) {
  status = await page.innerText('#status');
  if (status !== last) { console.log(((Date.now() - t0) / 1000).toFixed(0) + 's', status); last = status; }
  if (/^(Done|Error|No speech|Could not|This file|Something)/.test(status)) break;
  await page.waitForTimeout(1000);
}
await page.screenshot({ path: path.join(shots, 'app-result.png') });
const out = await page.innerText('#out');
console.log('---\n' + out + '\n---');
let fail = false;
const check = (ok, what) => { console.log((ok ? 'ok   ' : 'FAIL ') + what); if (!ok) fail = true; };
check(status.startsWith('Done'), 'finished: ' + status);
check(/Found 2 speakers/.test(status), 'found 2 speakers');
for (const w of ['Navy ordered me', 'pocket model', 'feet long']) check(out.toLowerCase().includes(w.toLowerCase()), 'contains "' + w + '"');
// rename a speaker and make sure every label for that speaker changes
const who = await page.$$('.who');
if (who.length) {
  await who[0].click(); await page.fill('#pop input', 'Interviewer'); await page.click('#pop .r button');
  const names = await page.$$eval('.who', els => els.map(e => e.textContent));
  check(names.includes('Interviewer') && !names.includes('Speaker 1'), 'rename speaker');
}
// stop button: start again and cancel straight away, then make sure a new run still works
await page.click('#go'); await page.waitForTimeout(1500); await page.click('#cancel');
check((await page.innerText('#status')).startsWith('Stopped'), 'stop button');
await page.click('#go');
for (let i = 0; i < 900; i++) { status = await page.innerText('#status'); if (/^(Done|Error|Could not)/.test(status)) break; await page.waitForTimeout(1000); }
check(status.startsWith('Done'), 'runs again after stopping: ' + status);
// other file types people actually have: phone recordings, WhatsApp voice notes, videos
for (const f of ['clip.mp3', 'clip.m4a', 'voicenote.ogg', 'clip.mp4']) {
  await page.setInputFiles('#pick', path.join(here, 'audio', f));
  await page.click('#go');
  let st = '';
  for (let i = 0; i < 600; i++) { st = await page.innerText('#status'); if (/^(Done|Error|Could not|This file|No speech)/.test(st)) break; await page.waitForTimeout(1000); }
  const txt = (await page.innerText('#out')).toLowerCase();
  check(st.startsWith('Done') && txt.includes('fortunate'), 'opens and transcribes ' + f + ': ' + st);
}
await app.close();
console.log(fail ? 'APP TEST FAILED' : 'APP TEST PASSED', 'in', ((Date.now() - t0) / 1000).toFixed(0) + 's');
process.exit(fail ? 1 : 0);
