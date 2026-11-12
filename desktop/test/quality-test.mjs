// Unit and end-to-end checks for the mistake guards.
import { fork } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isLoop, collapseRepeats, dropInvented, cleanAudio, fixDriftedWords } from '../src/quality.mjs';
import { createCoordinator } from '../src/coordinator.mjs';
const here = path.dirname(fileURLToPath(import.meta.url));
let fail = false;
const check = (ok, what) => { console.log((ok ? 'ok   ' : 'FAIL ') + what); if (!ok) fail = true; };
const W = (s, p = 0.9) => s.split(' ').map((t, i) => ({ text: t, start: i * 0.3, end: i * 0.3 + 0.25, p }));
const txt = (ws) => ws.map(w => w.text).join(' ');

// loops
check(isLoop(W('I think so. I think so. I think so. I think so. I think so.')), 'detects a repeated phrase');
check(isLoop(W('the the the the the the cat sat')), 'detects a stuck word');
check(!isLoop(W('no no I said we should go to the shop and then come back home')), 'normal speech is not a loop');
check(!isLoop(W('very very good indeed and we liked it')), '"very very" is not a loop');
check(txt(collapseRepeats(W('okay I think so. I think so. I think so. I think so. then we left'))) === 'okay I think so. I think so. then we left', 'collapses a loop, keeps two');
check(txt(collapseRepeats(W('no no no no no no no stop'))) === 'no no no stop', 'collapses a stuck word, keeps three');

// invented words in silence
const sr = 16000, audio = new Float32Array(sr * 6);
for (let i = 0; i < sr * 3; i++) audio[i] = 0.2 * Math.sin(i * 0.05) * Math.sin(i * 0.0007);   // "speech" for 3 s, then silence
const ws = [{ text: 'hello', start: 0.2, end: 0.6, p: 0.9 }, { text: 'there', start: 0.7, end: 1.2, p: 0.9 }, { text: 'friend.', start: 1.3, end: 1.9, p: 0.8 },
  { text: 'Thanks', start: 4.0, end: 4.3, p: 0.3 }, { text: 'for', start: 4.3, end: 4.5, p: 0.3 }, { text: 'watching!', start: 4.5, end: 5.0, p: 0.3 }];
check(txt(dropInvented(ws, audio, 0)) === 'hello there friend.', 'drops "Thanks for watching!" invented in silence');
const real = ws.map((w, i) => i > 2 ? { ...w, start: w.start - 2.5, end: w.end - 2.5, p: 0.9 } : w);
check(dropInvented(real, audio, 0).length === 6, 'keeps a confident "thanks for watching" that was really said');

// stranded first word is moved next to its sentence
const drift = fixDriftedWords([{ text: 'Harvard.', start: 7.0, end: 7.4 }, { text: 'The', start: 22.4, end: 22.6 }, { text: 'thing', start: 33.3, end: 33.5 }, { text: 'was', start: 33.5, end: 33.7 }]);
check(drift[1].start > 32.5 && drift[1].end <= 33.3, 'moves a stranded first word back next to its sentence');
check(fixDriftedWords([{ text: 'The', start: 22.4, end: 33.2 }, { text: 'thing', start: 33.3, end: 33.5 }])[0].start > 32.5, 'shrinks a word that swallowed a long silence');
check(fixDriftedWords(W('we went home and then we slept')).every((w, i) => w.start === i * 0.3), 'leaves normal timing alone');

// cleanup keeps length and doesn't clip
const c = cleanAudio(audio);
check(c.length === audio.length && Math.max(...c.map(Math.abs)) <= 1, 'audio cleanup keeps length, no clipping');

// end to end: speech, 25 s of near silence, speech. No words may appear in the gap.
const once = (() => { const b = fs.readFileSync(path.join(here, 'audio', 'interview-2-speakers.wav')); return Float32Array.from({ length: (b.length - 44) / 2 }, (_, i) => b.readInt16LE(44 + i * 2) / 32768); })();
const a = once.subarray(0, 8 * sr), z = once.subarray(10 * sr, 18 * sr);
let seed = 7; const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5);
const gap = Float32Array.from({ length: 25 * sr }, () => 0.0015 * rnd());
const mix = new Float32Array(a.length + gap.length + z.length); mix.set(a); mix.set(gap, a.length); mix.set(z, a.length + gap.length);
const spawn = (threads) => { const ch = fork(path.join(here, '..', 'src', 'engine.mjs'), [], { env: { ...process.env, TRANSCRIBER_MODELS: path.join(here, '..', 'models'), TRANSCRIBER_THREADS: String(threads) }, serialization: 'advanced' }); return { send: m => ch.send(m), onMessage: f => ch.on('message', f), onExit: f => ch.on('exit', f), kill: () => ch.kill() }; };
for (const speakers of ['off', '0']) {
  const words = await new Promise((resolve, reject) => {
    const out = []; const co = createCoordinator(spawn, (m) => {
      if (m.type === 'segments') for (const s of m.chunks) for (const w of s.words) out.push({ ...w, seg: s.start });
      if (m.type === 'done') { co.cancel(); resolve(out); } if (m.type === 'error') reject(new Error(m.text));
    });
    co.start({ audio: mix, language: 'english', speakers });
  });
  const inGap = words.filter(w => w.s > 8.5 && w.s < 8 + 25 - 0.5);
  if (process.env.DEBUG) console.log(words.filter(w => w.s > 6 && w.s < 36).map(w => w.t + '@' + w.s + (w.p != null ? '(' + w.p.toFixed(2) + ')' : '')).join(' '));
  check(words.length > 15 && inGap.length === 0, `speakers ${speakers === 'off' ? 'off' : 'on '}: no invented words in a 25 s silent gap (${words.length} words, ${inGap.length} in gap${inGap.length ? ': ' + inGap.map(w => w.t).join(' ') : ''})`);
}
console.log(fail ? 'QUALITY TEST FAILED' : 'QUALITY TEST PASSED');
process.exit(fail ? 1 : 0);
