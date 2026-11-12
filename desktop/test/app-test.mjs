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
import os from 'node:os';
const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'transcriber-test-'));
const docxPath = path.join(dataDir, 'export.docx');
const app = await electron.launch({ executablePath: exe || undefined, args, timeout: 120000, env: { ...process.env, TRANSCRIBER_DATA: dataDir, TRANSCRIBER_SAVE_PATH: docxPath, TRANSCRIBER_FAKE_MIC: path.join(here, 'audio', 'interview-2-speakers.wav') } });
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
await page.screenshot({ path: path.join(shots, 'app-result.png') });
for (const w of ['Navy ordered me', 'pocket model', 'feet long']) check(out.toLowerCase().includes(w.toLowerCase()), 'contains "' + w + '"');
const waitDone = async () => { let st = ''; for (let i = 0; i < 900; i++) { st = await page.innerText('#status'); if (/^(Done|Error|Could not|This file|No speech)/.test(st)) break; await page.waitForTimeout(1000); } return st; };
// unsure words: underlined, and "Next to check" jumps to one
const unsure = await page.$$eval('.w.unsure', els => els.map(e => e.textContent));
check(unsure.length > 0 && unsure.length < 15, 'underlines a few unsure words (' + unsure.join(', ') + ')');
await page.click('#check'); await page.waitForTimeout(600);
await page.locator('#out').screenshot({ path: path.join(shots, 'transcript.png') });
check(await page.$('.w.cur') !== null, '"Next to check" jumps to an unsure word');
// saved to recent transcripts automatically
check((await page.$$('#recentList .item')).length === 1, 'finished transcript appears in Recent');
// rename a speaker: every label changes, and the voice is remembered
const who = await page.$$('.who');
await who[0].click(); await page.fill('#pop input', 'Interviewer'); await page.click('#pop .r button');
const names = await page.$$eval('.who', els => els.map(e => e.textContent));
check(names.some(n => n.startsWith('Interviewer')) && !names.some(n => n.startsWith('Speaker 1')), 'rename speaker');
await page.waitForTimeout(500);
check(/saved voices \(1\)/.test(await page.innerText('#voicesBtn')), 'renamed voice is saved');
// find and replace
await page.keyboard.press('Control+f'); await page.fill('#findIn', 'computer');
await page.waitForTimeout(200);
const n1 = await page.innerText('#findN');
check(/of [1-9]/.test(n1), 'find shows matches: ' + n1);
await page.fill('#replIn', 'machine'); await page.click('#replAll');
const after = await page.innerText('#out');
check(!/computer/i.test(after) && /machine/i.test(after), 'replace all');
await page.keyboard.press('Escape');
// Word export
await page.click('#docx'); await page.waitForTimeout(1500);
const docxOk = fs.existsSync(docxPath) && fs.statSync(docxPath).size > 3000;
let docxText = '';
if (docxOk) { const { execSync } = await import('node:child_process'); try { docxText = execSync(process.platform === 'win32' ? `tar -xOf "${docxPath}" word/document.xml` : `unzip -p "${docxPath}" word/document.xml`).toString(); } catch {} }
check(docxOk && docxText.includes('Interviewer') && docxText.includes('machine'), 'Save Word: document has speaker names and edits');
await page.waitForTimeout(1200);   // let the edits auto-save
// transcribe the same recording again: the renamed voice should be recognised
await page.click('#go'); status = await waitDone();
const names2 = await page.$$eval('.who', els => els.map(e => e.textContent));
check(names2.some(n => n.startsWith('Interviewer') && n.includes('recognised')), 'recognises the saved voice in a new transcription: ' + status);
check((await page.$$('#recentList .item')).length === 2, 'second transcript in Recent');
// reopen the first one: edits were kept
const items = await page.$$('#recentList .item');
await (await items[1].$('button')).click(); await page.waitForTimeout(1500);
const reopened = await page.innerText('#out');
check(/machine/i.test(reopened) && /Interviewer/.test(reopened), 'reopened transcript keeps edits and names');
check(!(await page.$eval('#player', el => el.hidden)), 'reopened transcript can play the original recording');
// delete it (two clicks: Delete, then confirm)
const del = await (await page.$$('#recentList .item'))[1].$('button.del');
await del.click(); await del.click(); await page.waitForTimeout(500);
check((await page.$$('#recentList .item')).length === 1, 'delete from Recent');
await page.setInputFiles('#pick', audio);
// stop button: start again and cancel straight away, then make sure a new run still works
await page.click('#go'); await page.waitForTimeout(1500); await page.click('#cancel');
check((await page.innerText('#status')).startsWith('Stopped'), 'stop button');
await page.click('#go'); status = await waitDone();
check(status.startsWith('Done'), 'runs again after stopping: ' + status);
// live: record from the (fake) microphone, which plays the 34 s interview, then stop
{
  const before = (await page.$$('#recentList .item')).length;
  await page.click('#recBtn');
  await page.waitForTimeout(2000);
  check(!(await page.$eval('#recPanel', el => el.hidden)), 'recording starts (microphone opened)');
  let liveText = '';
  for (let i = 0; i < 60; i++) { await page.waitForTimeout(1000); liveText = await page.innerText('#out'); if (/feet/i.test(liveText) && i > 36) break; }
  check(/navy/i.test(liveText), 'text appears live while recording');
  await page.screenshot({ path: path.join(shots, 'live.png') });
  await page.click('#recStop');
  const st = await waitDone();
  const t = (await page.innerText('#out')).toLowerCase();
  check(st.startsWith('Recording finished') && /found [23] speakers/i.test(st), 'stop: speakers labelled: ' + st);
  for (const w of ['navy ordered me', 'pocket model', 'feet long']) check(t.includes(w), 'live transcript contains "' + w + '"');
  check(!(await page.$eval('#player', el => el.hidden)), 'recording can be played back');
  check((await page.$$('#recentList .item')).length === before + 1, 'recording saved to Recent');
  const recFiles = fs.readdirSync(path.join(dataDir, 'recordings'));
  check(recFiles.length === 1 && fs.statSync(path.join(dataDir, 'recordings', recFiles[0])).size > 50000, 'audio file saved: ' + recFiles.join(', '));
}
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
