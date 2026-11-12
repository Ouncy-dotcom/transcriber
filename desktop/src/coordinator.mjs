// Runs one transcription job across several worker processes:
//  - one worker works out who is speaking (speaker detection) while
//  - the others transcribe 30-second chunks in parallel.
// Results are sent to the window in order, with speaker labels, as soon as each chunk is ready.
import os from 'node:os';
import { anySpeech } from './diarize.js';
import { assignSpeakers, groupSegments } from './words.js';
import { cleanAudio, dropOutsideSpeech } from './quality.mjs';

// Saved voices: a speaker is given a saved name when their voice print is clearly closest to it.
// Calibrated on 50 labelled recordings: the same person scores 0.55-0.99, strangers almost never pass 0.6.
const MATCH = 0.6, MARGIN = 0.08;
export function matchVoices(voiceprints, saved) {
  const names = new Array(voiceprints.length).fill(null);
  if (!saved || !saved.length) return names;
  const pairs = [];
  voiceprints.forEach((v, k) => {
    const sims = saved.map(sv => { let d = 0; for (let i = 0; i < v.length; i++) d += v[i] * sv.emb[i]; return d; });
    const order = [...sims.keys()].sort((a, b) => sims[b] - sims[a]);
    const best = sims[order[0]], second = order.length > 1 ? sims[order[1]] : -1;
    if (best >= MATCH && best - second >= MARGIN) pairs.push([best, k, order[0]]);
  });
  pairs.sort((a, b) => b[0] - a[0]);
  const usedK = new Set(), usedS = new Set();
  for (const [sim, k, si] of pairs) if (!usedK.has(k) && !usedS.has(si)) { usedK.add(k); usedS.add(si); names[k] = { name: saved[si].name, sim: +sim.toFixed(3) }; }
  return names;
}

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
    // cleaned copy for the speech model (hum removed, quiet voices lifted); speaker detection uses the original
    const clean = msg.cleanup === false ? audio : cleanAudio(audio);
    let loopsFixed = 0, wordsDropped = 0;
    getPlan();
    emit({ type: 'plan', ...cfg });
    ensurePool(cfg, wantSpeakers);
    const t0 = performance.now();

    // 1. speaker detection starts straight away on its own worker
    let timeline = null, voiceNames = [];
    const timelineReady = wantSpeakers
      ? speakerWorker.ready
          .then(() => { emit({ type: 'status', text: 'Working out who is speaking…' }); return speakerWorker.request({ type: 'diarize', audio, speakers: msg.speakers }); })
          .then((m) => {
            timeline = m.tl.K > 0 ? m.tl : null;
            if (timeline) voiceNames = matchVoices(timeline.voiceprints || [], msg.voices);
            emit({ type: 'speakers', count: m.tl.K });
          })
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
          let labelled = words.map(w => ({ ...w, spk: -1 }));
          if (timeline) {
            const before = labelled.length;
            labelled = dropOutsideSpeech(labelled, (a, b) => anySpeech(timeline, a, b));
            wordsDropped += before - labelled.length;
            assignSpeakers(labelled, timeline);
          }
          const chunks = groupSegments(labelled, !!timeline);
          for (const c of chunks) if (c.speaker >= 0) {
            if (!order.has(c.speaker)) {
              order.set(c.speaker, order.size);
              const vn = voiceNames[c.speaker];
              if (vn) emit({ type: 'speaker-name', speaker: order.get(c.speaker), name: vn.name, sim: vn.sim });
            }
            c.speaker = order.get(c.speaker);
          }
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
        const slice = clean.slice(s, e);
        let words = [];
        if (rms(slice, 0, slice.length) >= 0.0015 && !(timeline && !anySpeech(timeline, s / SR, e / SR))) {
          const m = await w.request({ type: 'chunk', i, audio: slice, offset: s / SR, segEnd: e / SR, language: msg.language });
          words = m.words; loopsFixed += m.loopFixed || 0; wordsDropped += m.dropped || 0;
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
    // voice prints in display order, so names the user gives can be remembered for next time
    const voiceprints = [];
    if (timeline) for (const [c, d] of order) voiceprints[d] = timeline.voiceprints?.[c] ?? null;
    emit({ type: 'done', elapsed: performance.now() - t0, workers: cfg.workers, threads: cfg.threads, loopsFixed, wordsDropped, voiceprints });
  }

  // ---------- live: chunks arrive from the microphone while the person is still talking ----------
  let live = null;
  function liveStart(msg) {
    getPlan();
    ensurePool(cfg, msg.speakers !== 'off');
    live = { msg, results: [], sent: 0, next: 0, words: [], pending: [], cancelled: false, loopsFixed: 0 };
    job = live;
    emit({ type: 'plan', ...cfg });
  }
  function liveFlush() {
    const L = live;
    while (L.sent < L.results.length && L.results[L.sent] !== undefined) {
      const words = L.results[L.sent];
      if (words.length) {
        L.words.push(...words);
        const chunks = groupSegments(words.map(w => ({ ...w, spk: -1 })), false);
        if (chunks.length) emit({ type: 'segments', chunks, live: true });
      }
      L.sent++;
    }
    emit({ type: 'live-queue', waiting: L.results.length - L.sent });
  }
  function liveChunk({ audio, offset }) {
    const L = live; if (!L || L.cancelled) return;
    const i = L.results.length; L.results.push(undefined);
    const a = audio instanceof Float32Array ? audio : new Float32Array(audio.buffer ?? audio);
    const w = pool[L.next++ % pool.length];
    const clean = L.msg.cleanup === false ? a : cleanAudio(a);
    const p = (rms(clean, 0, clean.length) < 0.0015 ? Promise.resolve({ words: [] })
      : w.ready.then(() => w.request({ type: 'chunk', i, audio: clean, offset, segEnd: offset + a.length / SR, language: L.msg.language })))
      .then((m) => { if (L.cancelled) return; L.results[i] = m.words || []; L.loopsFixed += m.loopFixed || 0; liveFlush(); })
      .catch((err) => { if (!L.cancelled) { L.results[i] = []; emit({ type: 'live-warning', text: String(err && err.message || err) }); liveFlush(); } });
    L.pending.push(p);
  }
  // Recording stopped: finish any chunks still being transcribed, then label speakers over the whole recording.
  async function liveFinish({ audio }) {
    const L = live; if (!L) return;
    const t0 = performance.now();
    emit({ type: 'status', text: 'Finishing the last few words…' });
    await Promise.all(L.pending);
    if (L.cancelled) return;
    const full = audio instanceof Float32Array ? audio : new Float32Array(audio.buffer ?? audio);
    let voiceprints = [];
    if (L.msg.speakers !== 'off' && L.words.length) {
      emit({ type: 'status', text: 'Working out who said what…' });
      await speakerWorker.ready;
      const m = await speakerWorker.request({ type: 'diarize', audio: full, speakers: L.msg.speakers });
      if (L.cancelled) return;
      const tl = m.tl.K > 0 ? m.tl : null;
      emit({ type: 'speakers', count: m.tl.K });
      if (tl) {
        const voiceNames = matchVoices(tl.voiceprints || [], L.msg.voices);
        let labelled = dropOutsideSpeech(L.words.map(w => ({ ...w, spk: -1 })), (a, b) => anySpeech(tl, a, b));
        assignSpeakers(labelled, tl);
        const chunks = groupSegments(labelled, true);
        const order = new Map();
        for (const c of chunks) if (c.speaker >= 0) {
          if (!order.has(c.speaker)) {
            order.set(c.speaker, order.size);
            const vn = voiceNames[c.speaker];
            if (vn) emit({ type: 'speaker-name', speaker: order.get(c.speaker), name: vn.name, sim: vn.sim });
          }
          c.speaker = order.get(c.speaker);
        }
        for (const [c, d] of order) voiceprints[d] = tl.voiceprints?.[c] ?? null;
        emit({ type: 'replace', chunks });
      }
    }
    live = null;
    emit({ type: 'done', live: true, elapsed: performance.now() - t0, loopsFixed: L.loopsFixed, wordsDropped: 0, voiceprints });
  }

  return {
    liveStart(msg) { try { liveStart(msg); } catch (err) { emit({ type: 'error', text: String(err && err.message || err) }); } },
    liveChunk,
    async liveFinish(msg) {
      try { await liveFinish(msg); }
      catch (err) { if (!job?.cancelled) { killAll(); live = null; emit({ type: 'error', text: String(err && err.message || err) }); } }
    },
    async start(msg) {
      try { await run(msg); }
      catch (err) { if (!job?.cancelled) { killAll(); emit({ type: 'error', text: String(err && err.message || err) }); } }
    },
    cancel() { if (job) job.cancelled = true; live = null; killAll(); },
    // warm up one speech worker in the background so the first run starts sooner
    warmUp() {
      ensurePool(getPlan(), false);
      for (const w of pool) w.ready.then(() => w.request({ type: 'load', what: 'speech' })).catch(() => {});
    },
    plan,
  };
}
