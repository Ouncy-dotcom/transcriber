import { pipeline, env } from './transformers.min.js';

const BASE = new URL('./', import.meta.url).href;
const realFetch = self.fetch.bind(self);
let partsMap = null;
let transcriber = null;
let loadedModel = null;
let cancelled = false;

const post = (type, data = {}) => self.postMessage({ type, ...data });

async function getParts() {
  if (!partsMap) partsMap = await (await realFetch(BASE + 'parts.json')).json();
  return partsMap;
}

// Large files are hosted in pieces; put them back together, with progress.
async function assemble(key, label) {
  const parts = (await getParts())[key];
  const bufs = [];
  for (let i = 0; i < parts.length; i++) {
    const r = await realFetch(BASE + parts[i]);
    if (!r.ok) throw new Error('Could not load ' + parts[i]);
    bufs.push(await r.arrayBuffer());
    post('load', { label, done: i + 1, total: parts.length });
  }
  return new Blob(bufs);
}

self.fetch = async (input, init) => {
  const url = typeof input === 'string' ? input : input.url;
  if (url.startsWith(BASE)) {
    const key = url.slice(BASE.length).split('?')[0];
    const parts = await getParts();
    if (parts[key]) {
      const label = key.includes('encoder') ? 'encoder' : 'decoder';
      const blob = await assemble(key, label);
      return new Response(blob, { status: 200, headers: { 'content-length': String(blob.size), 'content-type': 'application/octet-stream' } });
    }
  }
  return realFetch(input, init);
};

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = BASE + 'models/';
env.useBrowserCache = true;

let wasmReady = false;
async function prepWasm() {
  if (wasmReady) return;
  const blob = await assemble('ort/ort.wasm', 'engine');
  env.backends.onnx.wasm.wasmBinary = await blob.arrayBuffer();
  env.backends.onnx.wasm.wasmPaths = BASE + 'ort/';
  env.backends.onnx.wasm.proxy = false;
  if (!self.crossOriginIsolated) env.backends.onnx.wasm.numThreads = 1;
  wasmReady = true;
}

async function loadModel(model) {
  if (transcriber && loadedModel === model) return;
  if (transcriber) { try { await transcriber.dispose(); } catch {} transcriber = null; }
  await prepWasm();
  post('status', { text: 'Loading model…' });
  transcriber = await pipeline('automatic-speech-recognition', model, {
    dtype: { encoder_model: 'q4', decoder_model_merged: 'q8' },
    device: 'wasm',
  });
  loadedModel = model;
}

function rms(a, s, e) {
  let sum = 0;
  for (let i = s; i < e; i++) sum += a[i] * a[i];
  return Math.sqrt(sum / Math.max(1, e - s));
}

// Split into windows of up to 30 s, cutting at the quietest moment so words are not chopped.
function makeWindows(audio, sr) {
  const wins = [];
  const max = 29.5 * sr, min = 18 * sr, frame = Math.round(0.05 * sr);
  let start = 0;
  while (start < audio.length) {
    if (audio.length - start <= max) { wins.push([start, audio.length]); break; }
    let best = start + max, bestVal = Infinity;
    for (let p = start + min; p + frame <= start + max; p += frame) {
      const v = rms(audio, p, p + frame);
      if (v < bestVal) { bestVal = v; best = p + Math.floor(frame / 2); }
    }
    wins.push([start, best]);
    start = best;
  }
  return wins;
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'cancel') { cancelled = true; return; }
  if (msg.type !== 'run') return;
  cancelled = false;
  try {
    await loadModel(msg.model);
    const sr = 16000;
    const audio = msg.audio;
    const wins = makeWindows(audio, sr);
    post('status', { text: 'Transcribing…' });
    const t0 = performance.now();
    for (let i = 0; i < wins.length; i++) {
      if (cancelled) { post('cancelled'); return; }
      const [s, en] = wins[i];
      const slice = audio.subarray(s, en);
      const offset = s / sr, segEnd = en / sr;
      if (rms(slice, 0, slice.length) < 0.0015) { post('progress', { done: i + 1, total: wins.length, elapsed: performance.now() - t0 }); continue; }
      const opts = { return_timestamps: true, task: 'transcribe' };
      if (msg.language) opts.language = msg.language;
      const out = await transcriber(slice, opts);
      const chunks = (out.chunks && out.chunks.length ? out.chunks : [{ timestamp: [0, segEnd - offset], text: out.text }])
        .map(c => ({
          start: offset + (c.timestamp[0] ?? 0),
          end: Math.min(segEnd, offset + (c.timestamp[1] ?? (segEnd - offset))),
          text: (c.text || '').trim(),
        }))
        .filter(c => c.text && !/^\[.*\]$|^\(.*\)$/.test(c.text));
      post('segments', { chunks });
      post('progress', { done: i + 1, total: wins.length, elapsed: performance.now() - t0 });
    }
    post('done');
  } catch (err) {
    post('error', { text: String(err && err.message || err) });
  }
};
