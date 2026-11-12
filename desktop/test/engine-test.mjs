// Runs the engine as a child process on a 16 kHz mono WAV and prints the labelled transcript.
// Usage: node test/engine-test.mjs file.wav [speakers] [expectSpeakers] [expectText...]
import { fork } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const [file, speakers = '0', expectK, ...expect] = process.argv.slice(2);

function readWav(p) {
  const b = fs.readFileSync(p);
  let o = 12, fmt = null;
  while (o < b.length) {
    const id = b.toString('ascii', o, o + 4), size = b.readUInt32LE(o + 4);
    if (id === 'fmt ') fmt = { ch: b.readUInt16LE(o + 10), sr: b.readUInt32LE(o + 12), bits: b.readUInt16LE(o + 22), tag: b.readUInt16LE(o + 8) };
    if (id === 'data') {
      if (!fmt || fmt.sr !== 16000) throw new Error('need 16 kHz WAV');
      const n = size / (fmt.bits / 8) / fmt.ch, out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let c = 0; c < fmt.ch; c++) {
          const at = o + 8 + (i * fmt.ch + c) * (fmt.bits / 8);
          s += fmt.bits === 16 ? b.readInt16LE(at) / 32768 : b.readFloatLE(at);
        }
        out[i] = s / fmt.ch;
      }
      return out;
    }
    o += 8 + size + (size & 1);
  }
  throw new Error('no data chunk');
}

const once = readWav(file);
const repeat = +(process.env.REPEAT || 1);
const audio = new Float32Array(once.length * repeat); for (let r = 0; r < repeat; r++) audio.set(once, r * once.length);
const { createCoordinator } = await import('../src/coordinator.mjs');
const spawn = (threads) => {
  const c = fork(path.join(here, '..', 'src', 'engine.mjs'), [], {
    env: { ...process.env, TRANSCRIBER_MODELS: process.env.TRANSCRIBER_MODELS || path.join(here, '..', 'models'), TRANSCRIBER_THREADS: String(threads) },
    serialization: 'advanced',
  });
  return { send: (m) => c.send(m), onMessage: (fn) => c.on('message', fn), onExit: (fn) => c.on('exit', fn), kill: () => c.kill() };
};
const segs = []; const t0 = Date.now();
const timer = setTimeout(() => { console.error('TIMEOUT'); process.exit(2); }, 30 * 60 * 1000);
const coord = createCoordinator(spawn, (m) => {
  if (m.type === 'plan') console.log(`plan: ${m.workers} speech worker(s) x ${m.threads} thread(s), speaker worker ${m.speakerThreads} thread(s); ${m.logical} logical CPUs, ${m.freeGB} GB free`);
  else if (m.type === 'status') console.log(((Date.now() - t0) / 1000).toFixed(1) + 's', m.text);
  else if (m.type === 'speakers') console.log(((Date.now() - t0) / 1000).toFixed(1) + 's speakers found:', m.count);
  else if (m.type === 'segments') for (const c of m.chunks) { segs.push(c); console.log(`[${c.start.toFixed(1)}] S${c.speaker + 1}: ${c.text}`); }
  else if (m.type === 'error') { console.error('ENGINE ERROR', m.text); process.exit(1); }
  else if (m.type === 'done') {
    clearTimeout(timer);
    const secs = audio.length / 16000;
    console.log(`DONE in ${(m.elapsed / 1000).toFixed(1)}s for ${secs.toFixed(1)}s of audio (${(m.elapsed / 1000 / secs).toFixed(2)}x real time)`);
    coord.cancel();
    const text = segs.map(s => s.text).join(' ').toLowerCase();
    let fail = false;
    const k = new Set(segs.map(s => s.speaker)).size;
    if (expectK && +expectK !== k) { console.error('FAIL: expected', expectK, 'speakers, got', k); fail = true; }
    for (const w of expect) if (!text.includes(w.toLowerCase())) { console.error('FAIL: missing text:', w); fail = true; }
    if (!segs.length) { console.error('FAIL: empty transcript'); fail = true; }
    for (let i = 1; i < segs.length; i++) if (segs[i].start + 0.01 < segs[i - 1].start) { console.error('FAIL: lines out of order at', segs[i].start); fail = true; break; }
    const allWords = segs.flatMap(s => s.words || []);
    const withP = allWords.filter(w => w.p != null);
    console.log(`word confidence on ${withP.length}/${allWords.length} words; least sure:`, [...withP].sort((a, b) => a.p - b.p).slice(0, 6).map(w => `${w.t}(${w.p.toFixed(2)})`).join(' '));
    console.log(`voice prints: ${(m.voiceprints || []).filter(Boolean).length}, loops fixed: ${m.loopsFixed}, invented words dropped: ${m.wordsDropped}`);
    if (expectK && withP.length < allWords.length * 0.9) { console.error('FAIL: word confidence missing'); fail = true; }
    if (expectK && (m.voiceprints || []).filter(Boolean).length !== +expectK) { console.error('FAIL: voice prints missing'); fail = true; }
    console.log(fail ? 'TEST FAILED' : 'TEST PASSED');
    setTimeout(() => process.exit(fail ? 1 : 0), 200);
  }
});
coord.start({ type: 'run', audio, language: 'english', speakers });
