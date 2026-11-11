// Runs one transcription job across several worker processes:
//  - one worker works out who is speaking (speaker detection) while
//  - the others transcribe 30-second chunks in parallel.
// Results are sent to the window in order, with speaker labels, as soon as each chunk is ready.
import os from 'node:os';
import { anySpeech } from './diarize.js';
import { assignSpeakers, groupSegments } from './words.js';

const SR = 16000;
const GB = 1024 ** 3;
const PER_WORKER = 1.8 * GB;     // measured peak memory of one speech worker
const SPEAKER_WORKER = 0.5 * GB;

// How many speech workers and threads to use on this computer.
export function plan({ speakers = true, freeMem = os.freemem(), logical = os.availableParallelism?.() ?? os.cpus().length } = {}) {
  // Each speech worker is limited mostly by per-word overhead rather than raw maths, so several
  // workers with 1-2 threads each beat one worker with many threads (measured: 2x1 is 1.3x faster than 1x2).
  const cores = logical >= 4 ? Math.floor(logical / 2) : logical;                    // ~physical cores
  const spare = freeMem - 1.0 * GB - (speakers ? SPEAKER_WORKER : 0);                 // leave room for Windows and other apps
  const byMem = Math.max(1, Math.floor(spare / PER_WORKER));
  let workers = Math.max(1, Math.min(4, cores, byMem));
  let threads = Math.max(1, Math.floor(cores / workers));
  const force = /^(\d+)x(\d+)$/.exec(process.env.TRANSCRIBER_PLAN || '');   // for testing, e.g. TRANSCRIBER_PLAN=2x1
  if (force) { workers = +force[1]; threads = +force[2]; }
  return { workers, threads, speakerThreads: 1, cores, logical, freeGB: +(freeMem / GB).toFixed(1) };
}

function rms(a, s, e) {
  let sum = 0;
  for (let i = s; i < e; i++) sum += a[i] * a[i];
  return Math.sqrt(sum / Math.max(1, e - s));
}

// Split into windows of up to 30 s, cutting at the quietest moment so words are not chopped.
export function makeWindows(audio) {
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

/**
 * spawn(threads) must return a worker: { send(msg), onMessage(fn), kill() } that sends 'ready' when started.
 * emit(msg) receives the same messages the window understands: status, diar, speakers, segments, progress, done, error.
 */
export function createCoordinator(spawn, emit) {
  let pool = [], speakerWorker = null, job = null, poolKey = '';
  // decided once, before any worker is running (our own workers would otherwise eat into the free memory figure)
  let cfg = null;
  const getPlan = () => (cfg ??= plan());

  function startWorker(threads) {
    const w = spawn(threads);
    w.threads = threads;
    let readyResolve, current = null;
    w.ready = new Promise((resolve) => { readyResolve = resolve; });
    w.alive = true;
    w.onMessage((m) => {
      if (m.type === 'ready') return readyResolve();
      if (m.type === 'diar') return emit({ type: 'diar', phase: m.phase, done: m.done, total: m.total });
      if (!current) return;
      const c = current; current = null;
      if (m.type === 'error') c.reject(new Error(m.text)); else c.resolve(m);
    });
    w.onExit((code) => {
      w.alive = false;
      if (current) { const c = current; current = null; c.reject(new Error(w.killed ? 'stopped' : 'A transcription worker stopped unexpectedly (code ' + code + '). The computer may be running low on memory.')); }
    });
    // a worker does one thing at a time, so requests queue up behind each other
    let chain = Promise.resolve();
    w.request = (msg) => {
      const p = chain.then(() => new Promise((resolve, reject) => {
        if (!w.alive) return reject(new Error('stopped'));
        current = { resolve, reject };
        w.send(msg);
      }));
      chain = p.catch(() => {});
      return p;
    };
    const kill = w.kill;
    w.kill = () => { w.killed = true; kill(); };
    return w;
  }

  function ensurePool(p, withSpeakers) {
    const key = p.workers + 'x' + p.threads;
    if (key !== poolKey) { for (const w of pool) w.kill(); pool = []; poolKey = key; }
    while (pool.length < p.workers) pool.push(startWorker(p.threads));
    if (withSpeakers && !speakerWorker) speakerWorker = startWorker(p.speakerThreads);
  }

  function killAll() {
    for (const w of pool) w.kill();
    if (speakerWorker) speakerWorker.kill();
    pool = []; speakerWorker = null; poolKey = '';
  }

  async function run(msg) {
    const myJob = job = { cancelled: false };
    const audio = msg.audio instanceof Float32Array ? msg.audio : new Float32Array(msg.audio.buffer ?? msg.audio);
    const wantSpeakers = msg.speakers !== 'off';
    getPlan();
    emit({ type: 'plan', ...cfg });
    ensurePool(cfg, wantSpeakers);
    const t0 = performance.now();

    // 1. speaker detection starts straight away on its own worker
    let timeline = null;
    const timelineReady = wantSpeakers
      ? speakerWorker.ready
          .then(() => { emit({ type: 'status', text: 'Working out who is speaking…' }); return speakerWorker.request({ type: 'diarize', audio, speakers: msg.speakers }); })
          .then((m) => { timeline = m.tl.K > 0 ? m.tl : null; emit({ type: 'speakers', count: m.tl.K }); })
      : Promise.resolve();

    // 2. speech workers take chunks from a shared queue
    const wins = makeWindows(audio);
    const results = new Array(wins.length);
    let next = 0, finished = 0, sent = 0;
    const order = new Map();
    const t1 = performance.now();

    const flush = () => {
      if (wantSpeakers && !timeline && !timelineDone) return;   // wait for speaker labels before showing text
      while (sent < wins.length && results[sent] !== undefined) {
        const words = results[sent];
        if (words && words.length) {
          const labelled = words.map(w => ({ ...w, spk: -1 }));
          if (timeline) assignSpeakers(labelled, timeline);
          const chunks = groupSegments(labelled, !!timeline);
          for (const c of chunks) if (c.speaker >= 0) { if (!order.has(c.speaker)) order.set(c.speaker, order.size); c.speaker = order.get(c.speaker); }
          if (chunks.length) emit({ type: 'segments', chunks });
        }
        sent++;
      }
    };
    let timelineDone = !wantSpeakers;
    const tlPromise = timelineReady.then(() => { timelineDone = true; if (!myJob.cancelled) flush(); });

    let firstStatus = false;
    const workerLoop = async (w) => {
      await w.ready;
      if (!firstStatus && !myJob.cancelled) { firstStatus = true; if (!wantSpeakers || timelineDone) emit({ type: 'status', text: 'Transcribing…' }); }
      while (!myJob.cancelled) {
        const i = next++;
        if (i >= wins.length) return;
        const [s, e] = wins[i];
        const slice = audio.slice(s, e);
        let words = [];
        if (rms(slice, 0, slice.length) >= 0.0015 && !(timeline && !anySpeech(timeline, s / SR, e / SR))) {
          const m = await w.request({ type: 'chunk', i, audio: slice, offset: s / SR, segEnd: e / SR, language: msg.language });
          words = m.words;
        }
        if (myJob.cancelled) return;
        results[i] = words; finished++;
        emit({ type: 'progress', done: finished, total: wins.length, elapsed: performance.now() - t1 });
        flush();
      }
    };
    await Promise.all([...pool.map(workerLoop), tlPromise]);
    if (myJob.cancelled) return;
    flush();
    emit({ type: 'done', elapsed: performance.now() - t0, workers: cfg.workers, threads: cfg.threads });
  }

  return {
    async start(msg) {
      try { await run(msg); }
      catch (err) { if (!job?.cancelled) { killAll(); emit({ type: 'error', text: String(err && err.message || err) }); } }
    },
    cancel() { if (job) job.cancelled = true; killAll(); },
    // warm up one speech worker in the background so the first run starts sooner
    warmUp() {
      ensurePool(getPlan(), false);
      for (const w of pool) w.ready.then(() => w.request({ type: 'load', what: 'speech' })).catch(() => {});
    },
    plan,
  };
}
