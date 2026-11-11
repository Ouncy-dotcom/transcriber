import { pipeline, env, AutoProcessor, AutoModel, AutoModelForAudioFrameClassification } from './transformers.min.js';
import { diarize, anySpeech } from './diarize.js';
import { cleanWords, assignSpeakers, groupSegments } from './words.js';

const BASE = new URL('./', import.meta.url).href;
const realFetch = self.fetch.bind(self);
const SR = 16000;
const EMB_DTYPE = 'q8';   // the 8-bit voice print model scored the same as the full one in testing, at half the time
let partsMap = null;
let transcriber = null, loadedModel = null;
let seg = null, emb = null;

const post = (type, data = {}) => self.postMessage({ type, ...data });

async function getParts() {
  if (!partsMap) partsMap = await (await realFetch(BASE + 'parts.json')).json();
  return partsMap;
}

async function fetchRetry(url, tries = 4) {
  let last;
  for (let i = 0; i < tries; i++) {
    try {
      const r = await realFetch(url);
      if (r.ok) return await r.arrayBuffer();
      last = new Error('HTTP ' + r.status);
    } catch (e) { last = e; }
    await new Promise(r => setTimeout(r, 800 * (i + 1)));
  }
  throw new Error('Could not download ' + url.slice(BASE.length) + ' (' + (last && last.message) + '). Check your connection and try again.');
}

// Large files are hosted in pieces; put them back together, with progress.
async function assemble(key, label) {
  const parts = (await getParts())[key];
  const bufs = [];
  for (let i = 0; i < parts.length; i++) {
    bufs.push(await fetchRetry(BASE + parts[i]));
    post('load', { label, done: i + 1, total: parts.length });
  }
  return new Blob(bufs);
}

function labelFor(key) {
  if (key.includes('pyannote') || key.includes('wespeaker')) return 'speakers';
  return key.includes('encoder') ? 'encoder' : 'decoder';
}

self.fetch = async (input, init) => {
  const url = typeof input === 'string' ? input : input.url;
  if (url.startsWith(BASE)) {
    const key = url.slice(BASE.length).split('?')[0];
    const parts = await getParts();
    if (parts[key]) {
      const blob = await assemble(key, labelFor(key));
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
  post('status', { text: 'Loading speech model…' });
  transcriber = await pipeline('automatic-speech-recognition', model, {
    dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' },
    device: 'wasm',
  });
  loadedModel = model;
}

async function loadSpeakerModels() {
  if (seg && emb) return;
  post('status', { text: 'Loading speaker model…' });
  const segId = 'pyannote-segmentation-3.0', embId = 'wespeaker-voxceleb-resnet34-LM';
  const sp = await AutoProcessor.from_pretrained(segId);
  const sm = await AutoModelForAudioFrameClassification.from_pretrained(segId, { dtype: 'fp32', device: 'wasm' });
  const ep = await AutoProcessor.from_pretrained(embId);
  const em = await AutoModel.from_pretrained(embId, { dtype: EMB_DTYPE, device: 'wasm' });
  seg = async (buf) => { const { logits } = await sm(await sp(buf)); return { data: logits.data, dims: logits.dims }; };
  emb = async (buf) => { const out = await em(await ep(buf)); return (out.embeds ?? out.last_hidden_state ?? Object.values(out)[0]).data; };
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
  if (msg.type !== 'run') return;
  try {
    const audio = msg.audio;
    await prepWasm();
    let tl = null;
    const order = new Map();   // speakers are numbered in the order they first appear in the transcript
    const t0 = performance.now();
    if (msg.speakers !== 'off') {
      await loadSpeakerModels();
      post('status', { text: 'Working out who is speaking…' });
      const res = await diarize(audio, {
        segment: seg, embed: emb, numSpeakers: +msg.speakers || 0,
        onProgress: (phase, done, total) => post('diar', { phase, done, total }),
      });
      if (res.K > 0) tl = res;
      post('speakers', { count: res.K });
    }
    await loadModel(msg.model);
    const wins = makeWindows(audio, SR);
    post('status', { text: 'Transcribing…' });
    const t1 = performance.now();
    for (let i = 0; i < wins.length; i++) {
      const [s, en] = wins[i];
      const slice = audio.subarray(s, en);
      const offset = s / SR, segEnd = en / SR;
      const silent = rms(slice, 0, slice.length) < 0.0015 || (tl && !anySpeech(tl, offset, segEnd));
      if (!silent) {
        const opts = { return_timestamps: 'word', task: 'transcribe' };
        if (msg.language) opts.language = msg.language;
        const out = await transcriber(slice, opts);
        const words = cleanWords(out.chunks, offset, segEnd);
        if (tl && words.length) assignSpeakers(words, tl);
        const chunks = groupSegments(words, !!tl);
        for (const c of chunks) if (c.speaker >= 0) { if (!order.has(c.speaker)) order.set(c.speaker, order.size); c.speaker = order.get(c.speaker); }
        if (chunks.length) post('segments', { chunks });
      }
      post('progress', { done: i + 1, total: wins.length, elapsed: performance.now() - t1 });
    }
    post('done', { elapsed: performance.now() - t0 });
  } catch (err) {
    post('error', { text: String(err && err.message || err) });
  }
};
