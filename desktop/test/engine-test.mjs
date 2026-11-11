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

const audio = readWav(file);
const child = fork(path.join(here, '..', 'src', 'engine.mjs'), [], {
  env: { ...process.env, TRANSCRIBER_MODELS: process.env.TRANSCRIBER_MODELS || path.join(here, '..', 'models') },
  serialization: 'advanced',
});
const segs = []; let K = 0; const t0 = Date.now();
const timer = setTimeout(() => { console.error('TIMEOUT'); child.kill(); process.exit(2); }, 30 * 60 * 1000);
child.on('message', (m) => {
  if (m.type === 'ready') { console.log('engine ready, threads', m.threads); child.send({ type: 'run', audio, language: 'english', speakers }); }
  else if (m.type === 'status') console.log(((Date.now() - t0) / 1000).toFixed(1) + 's', m.text);
  else if (m.type === 'speakers') { K = m.count; console.log('speakers found:', K); }
  else if (m.type === 'segments') for (const c of m.chunks) { segs.push(c); console.log(`[${c.start.toFixed(1)}] S${c.speaker + 1}: ${c.text}`); }
  else if (m.type === 'error') { console.error('ENGINE ERROR', m.text); process.exit(1); }
  else if (m.type === 'done') {
    clearTimeout(timer);
    const secs = audio.length / 16000;
    console.log(`DONE in ${(m.elapsed / 1000).toFixed(1)}s for ${secs.toFixed(1)}s of audio (${(m.elapsed / 1000 / secs).toFixed(2)}x real time)`);
    child.kill();
    const text = segs.map(s => s.text).join(' ').toLowerCase();
    let fail = false;
    if (expectK && +expectK !== new Set(segs.map(s => s.speaker)).size) { console.error('FAIL: expected', expectK, 'speakers, got', new Set(segs.map(s => s.speaker)).size); fail = true; }
    for (const w of expect) if (!text.includes(w.toLowerCase())) { console.error('FAIL: missing text:', w); fail = true; }
    if (!segs.length) { console.error('FAIL: empty transcript'); fail = true; }
    console.log(fail ? 'TEST FAILED' : 'TEST PASSED');
    process.exit(fail ? 1 : 0);
  }
});
