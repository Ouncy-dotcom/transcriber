// Before/after test for audio cleanup on deliberately damaged audio. Reference = transcript of the clean recording.
import { fork } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createCoordinator } from '../src/coordinator.mjs';
const here = path.dirname(fileURLToPath(import.meta.url));
const raw = fs.readFileSync(process.argv[2]); const clean = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const SR = 16000;
// damage: every other 20 s block is distant (-24 dB), plus 50 Hz hum and hiss
let seed = 1; const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5);
const damaged = clean.map((x, i) => (Math.floor(i / (20 * SR)) % 2 ? x * 0.063 : x) + 0.02 * Math.sin(2 * Math.PI * 50 * i / SR) + 0.006 * Math.sin(2 * Math.PI * 100 * i / SR) + 0.004 * rnd());
const spawn = (threads) => { const c = fork(path.join(here, '..', 'src', 'engine.mjs'), [], { env: { ...process.env, TRANSCRIBER_MODELS: path.join(here, '..', 'models'), TRANSCRIBER_THREADS: String(threads) }, serialization: 'advanced' }); return { send: m => c.send(m), onMessage: f => c.on('message', f), onExit: f => c.on('exit', f), kill: () => c.kill() }; };
const coord = createCoordinator(spawn, () => {});
function run(audio, cleanup) {
  return new Promise((resolve, reject) => {
    const words = []; let p = 0;
    const c = createCoordinator(spawn, (m) => {
      if (m.type === 'segments') for (const s of m.chunks) for (const w of s.words) words.push(w);
      if (m.type === 'done') { c.cancel(); resolve({ words, m }); }
      if (m.type === 'error') reject(new Error(m.text));
    });
    c.start({ audio, language: 'english', speakers: 'off', cleanup });
  });
}
const norm = (ws) => ws.map(w => w.t.toLowerCase().replace(/[^a-z0-9']/g, '')).filter(Boolean);
function wer(ref, hyp) {
  const R = ref.length, H = hyp.length; let prev = Array.from({ length: H + 1 }, (_, j) => j);
  for (let i = 1; i <= R; i++) { const cur = [i]; for (let j = 1; j <= H; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1)); prev = cur; }
  return prev[H] / R;
}
// which hypothesis words are wrong (not aligned to an identical reference word) -> confidence check
function wrongFlags(ref, hyp) {
  const R = ref.length, H = hyp.length; const D = Array.from({ length: R + 1 }, () => new Int32Array(H + 1));
  for (let i = 0; i <= R; i++) D[i][0] = i; for (let j = 0; j <= H; j++) D[0][j] = j;
  for (let i = 1; i <= R; i++) for (let j = 1; j <= H; j++) D[i][j] = Math.min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1));
  const wrong = new Array(H).fill(true); let i = R, j = H;
  while (i > 0 && j > 0) { if (D[i][j] === D[i - 1][j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1)) { if (ref[i - 1] === hyp[j - 1]) wrong[j - 1] = false; i--; j--; } else if (D[i][j] === D[i - 1][j] + 1) i--; else j--; }
  return wrong;
}
const truth = process.argv[3] ? JSON.parse(fs.readFileSync(process.argv[3])) : null;
let R;
if (truth) R = truth.toLowerCase().split(/\s+/).map(w => w.replace(/[^a-z0-9']/g, '')).filter(Boolean);
else R = norm((await run(clean, false)).words);
console.log('reference words:', R.length);
const results = {};
for (const [name, audio, cl] of [...(truth ? [['clean, no cleanup', clean, false]] : []), ['clean audio + cleanup', clean, true], ['damaged, no cleanup', damaged, false], ['damaged + cleanup', damaged, true]]) {
  const r = await run(audio, cl); results[name] = r;
  console.log(`${name.padEnd(22)} word error rate: ${(100 * wer(R, norm(r.words))).toFixed(1)}%  (${(r.m.elapsed / 1000).toFixed(0)}s)`);
}
// do low-confidence words line up with real mistakes?
for (const which of [...(truth ? ['clean, no cleanup'] : []), 'damaged, no cleanup']) { console.log('-- confidence vs mistakes:', which);
const r = results[which]; const ws = r.words.filter(w => w.t.replace(/[^a-zA-Z0-9']/g, '')); const flags = wrongFlags(R, norm(ws));
for (const th of [0.3, 0.4, 0.5]) {
  const unsure = ws.map(w => (w.p ?? 1) < th);
  const caught = flags.filter((f, i) => f && unsure[i]).length, nWrong = flags.filter(Boolean).length, nUnsure = unsure.filter(Boolean).length;
  console.log(`threshold ${th}: underlines ${(100 * nUnsure / ws.length).toFixed(1)}% of words, catches ${nWrong ? (100 * caught / nWrong).toFixed(0) : '-'}% of mistakes; ${nUnsure ? (100 * caught / nUnsure).toFixed(0) : '-'}% of underlined words really were wrong`);
}
}
process.exit(0);
