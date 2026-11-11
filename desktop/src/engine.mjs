// Transcription engine. Runs in its own background process so the window never freezes.
// Uses the same speaker-detection and word-labelling code as the website, but runs the
// models natively on all CPU cores.
import path from 'node:path';
import os from 'node:os';
import { env, pipeline, AutoProcessor, AutoModel, AutoModelForAudioFrameClassification } from '@huggingface/transformers';
import { diarize, anySpeech } from './diarize.js';
import { cleanWords, assignSpeakers, groupSegments } from './words.js';

const SR = 16000;
const MODELS = process.env.TRANSCRIBER_MODELS;
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = MODELS.endsWith(path.sep) ? MODELS : MODELS + path.sep;
env.useFSCache = false;

// leave one core free so the computer stays usable while it works
const logical = os.availableParallelism?.() ?? os.cpus().length;
// roughly one thread per physical core (hyper-threads add little here), keeping the PC responsive
const threads = logical >= 4 ? Math.min(8, Math.floor(logical / 2)) : Math.max(1, logical - 1);
const session_options = { intraOpNumThreads: threads, interOpNumThreads: 1 };

let transcriber = null, seg = null, emb = null;
let port = null;
const post = (type, data = {}) => port.postMessage({ type, ...data });

async function loadTranscriber() {
  if (transcriber) return;
  post('status', { text: 'Loading speech model…' });
  transcriber = await pipeline('automatic-speech-recognition', 'whisper-small_timestamped', {
    dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' },
    device: 'cpu',
    session_options,
  });
}

async function loadSpeakerModels() {
  if (seg && emb) return;
  post('status', { text: 'Loading speaker model…' });
  const segId = 'pyannote-segmentation-3.0', embId = 'wespeaker-voxceleb-resnet34-LM';
  const sp = await AutoProcessor.from_pretrained(segId);
  const sm = await AutoModelForAudioFrameClassification.from_pretrained(segId, { dtype: 'fp32', device: 'cpu', session_options });
  const ep = await AutoProcessor.from_pretrained(embId);
  const em = await AutoModel.from_pretrained(embId, { dtype: 'q8', device: 'cpu', session_options });
  seg = async (buf) => { const { logits } = await sm(await sp(buf)); return { data: logits.data, dims: logits.dims }; };
  emb = async (buf) => { const out = await em(await ep(buf)); return (out.embeds ?? out.last_hidden_state ?? Object.values(out)[0]).data; };
}

function rms(a, s, e) {
  let sum = 0;
  for (let i = s; i < e; i++) sum += a[i] * a[i];
  return Math.sqrt(sum / Math.max(1, e - s));
}

// Split into windows of up to 30 s, cutting at the quietest moment so words are not chopped.
function makeWindows(audio) {
  const wins = [];
  const max = 29.5 * SR, min = 18 * SR, frame = Math.round(0.05 * SR);
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

async function run(msg) {
  const audio = msg.audio;
  const t0 = performance.now();
  let tl = null;
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
  await loadTranscriber();
  const wins = makeWindows(audio);
  post('status', { text: 'Transcribing…' });
  const order = new Map();
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
}

// Works both as an Electron utility process (process.parentPort) and as a plain Node child (for tests).
function attach(p, onMessage) {
  port = p;
  p.on('message', (e) => onMessage(e && e.data !== undefined && e.ports !== undefined ? e.data : e));
}
const handler = async (msg) => {
  if (!msg || msg.type !== 'run') return;
  try {
    if (!(msg.audio instanceof Float32Array)) msg.audio = new Float32Array(msg.audio.buffer ?? msg.audio);
    await run(msg);
  } catch (err) {
    post('error', { text: String(err && err.message || err) });
  }
};
if (process.parentPort) attach(process.parentPort, handler);
else attach({ postMessage: (m) => process.send(m), on: (ev, fn) => process.on(ev, fn) }, handler);
post('ready', { threads });
