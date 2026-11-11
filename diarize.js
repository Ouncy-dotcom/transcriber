// Speaker diarization: works out who spoke when.
// Pipeline: sliding-window speaker segmentation -> one voice print per (window, local speaker)
// -> global clustering of voice prints -> per-frame speaker timeline.
// Pure logic, no model code: the caller injects `segment` and `embed`, so the same file
// runs in the browser worker and in Node (for accuracy testing).

export const SR = 16000;
export const WIN = 160000;        // 10 s, the window the segmentation model was trained on
const STEP = 270;                 // samples per output frame
const FRAME0 = 360;               // sample where frame 0's tile starts (centre 495 minus half a step)
const SETS = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]]; // powerset classes -> local speakers

export const DEFAULTS = {
  hop: 80000,          // 5 s between windows
  minClean: 0.4,       // s of non-overlapped speech needed to prefer it over all speech
  minEmb: 0.4,         // s of speech needed to compute a voice print at all
  minClust: 1.5,       // s of speech needed for a voice print to take part in clustering
  maxEmb: 6,           // s of audio fed to the voice print model per (window, speaker)
  threshold: 0.75,     // centroid-linkage distance at which two clusters are different people
  maxMinSize: 3,       // clusters smaller than this are folded into the nearest big one
  maxClusterItems: 3500,
  embEvery: 1,         // compute voice prints on every Nth window; the rest are matched by overlap
};

// ---------- stage 1: segmentation ----------
export async function segmentAll(audio, segment, opts = {}, onProgress) {
  const o = { ...DEFAULTS, ...opts };
  const nWin = Math.max(1, Math.ceil((audio.length - WIN) / o.hop) + 1);
  const wins = [];
  for (let w = 0; w < nWin; w++) {
    const start = w * o.hop;
    const buf = new Float32Array(WIN);
    buf.set(audio.subarray(start, Math.min(audio.length, start + WIN)));
    const { data, dims } = await segment(buf);
    const F = dims[dims.length - 2], C = dims[dims.length - 1];
    const valid = Math.max(0, Math.min(F, Math.ceil((audio.length - start - FRAME0) / STEP)));
    const prob = new Float32Array(3 * F), act = new Uint8Array(3 * F), nact = new Uint8Array(F);
    for (let f = 0; f < valid; f++) {
      let mx = -Infinity, arg = 0;
      for (let c = 0; c < C; c++) { const v = data[f * C + c]; if (v > mx) { mx = v; arg = c; } }
      let sum = 0; const e = new Float32Array(C);
      for (let c = 0; c < C; c++) { e[c] = Math.exp(data[f * C + c] - mx); sum += e[c]; }
      for (let c = 1; c < C; c++) for (const k of SETS[c]) prob[k * F + f] += e[c] / sum;
      for (const k of SETS[arg]) act[k * F + f] = 1;
      nact[f] = SETS[arg].length;
    }
    wins.push({ start, F, valid, prob, act, nact });
    if (onProgress) onProgress(w + 1, nWin);
  }
  return wins;
}

// ---------- stage 2: voice prints ----------
function runsOf(mask, n) {
  const runs = []; let s = -1;
  for (let f = 0; f <= n; f++) {
    const on = f < n && mask[f];
    if (on && s < 0) s = f;
    else if (!on && s >= 0) { runs.push([s, f]); s = -1; }
  }
  return runs;
}

export async function embedAll(audio, wins, embed, opts = {}, onProgress) {
  const o = { ...DEFAULTS, ...opts };
  const items = [];
  const jobs = [];
  for (let w = 0; w < wins.length; w++) {
    const { F, valid, act, nact } = wins[w];
    for (let k = 0; k < 3; k++) {
      let all = 0, clean = 0;
      for (let f = 0; f < valid; f++) if (act[k * F + f]) { all++; if (nact[f] === 1) clean++; }
      const allDur = all * STEP / SR, cleanDur = clean * STEP / SR;
      const skip = o.embEvery > 1 && w % o.embEvery !== 0 && w !== wins.length - 1;
      if (allDur < o.minEmb || skip) { if (all > 0) items.push({ w, k, emb: null, dur: allDur, clean: cleanDur }); continue; }
      jobs.push({ w, k, useClean: cleanDur >= o.minClean, allDur, cleanDur });
    }
  }
  for (let j = 0; j < jobs.length; j++) {
    const { w, k, useClean, allDur, cleanDur } = jobs[j];
    const { start, F, valid, act, nact } = wins[w];
    const mask = new Uint8Array(valid);
    for (let f = 0; f < valid; f++) mask[f] = act[k * F + f] && (!useClean || nact[f] === 1) ? 1 : 0;
    let runs = runsOf(mask, valid);
    // longest stretches first, up to the cap, then back in time order
    const cap = Math.round(o.maxEmb * SR);
    runs.sort((a, b) => (b[1] - b[0]) - (a[1] - a[0]));
    const picked = []; let total = 0;
    for (const r of runs) {
      if (total >= cap) break;
      let s = start + r[0] * STEP + FRAME0, e = Math.min(audio.length, start + r[1] * STEP + FRAME0);
      if (e - s > cap - total) { const mid = (s + e) >> 1, half = (cap - total) >> 1; s = mid - half; e = mid + half; }
      if (e > s) { picked.push([s, e]); total += e - s; }
    }
    picked.sort((a, b) => a[0] - b[0]);
    const buf = new Float32Array(total); let p = 0;
    for (const [s, e] of picked) { buf.set(audio.subarray(s, e), p); p += e - s; }
    let emb = null;
    if (total >= o.minEmb * SR * 0.9) {
      emb = Float32Array.from(await embed(buf));
      let n = 0; for (let i = 0; i < emb.length; i++) n += emb[i] * emb[i];
      n = Math.sqrt(n);
      if (n > 0 && isFinite(n)) for (let i = 0; i < emb.length; i++) emb[i] /= n; else emb = null;
    }
    items.push({ w, k, emb, dur: useClean ? cleanDur : allDur, clean: cleanDur });
    if (onProgress) onProgress(j + 1, jobs.length);
  }
  return items;
}

// ---------- stage 3: clustering ----------
function linkage(embs) {
  // agglomerative clustering, centroid linkage on unit vectors (Lance-Williams update)
  const n = embs.length, d = embs[0].length;
  const D = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    let dot = 0; const a = embs[i], b = embs[j];
    for (let t = 0; t < d; t++) dot += a[t] * b[t];
    D[i * n + j] = D[j * n + i] = Math.max(0, 2 - 2 * dot);
  }
  const alive = new Uint8Array(n).fill(1), size = new Float64Array(n).fill(1);
  const nn = new Int32Array(n).fill(-1), nnd = new Float32Array(n).fill(Infinity);
  const findNN = (i) => {
    let b = -1, bd = Infinity;
    for (let j = 0; j < n; j++) if (alive[j] && j !== i && D[i * n + j] < bd) { bd = D[i * n + j]; b = j; }
    nn[i] = b; nnd[i] = bd;
  };
  for (let i = 0; i < n; i++) findNN(i);
  const merges = [];
  for (let m = 0; m < n - 1; m++) {
    let i = -1, bd = Infinity;
    for (let t = 0; t < n; t++) if (alive[t] && nnd[t] < bd) { bd = nnd[t]; i = t; }
    if (i < 0) break;
    const j = nn[i];
    const ni = size[i], nj = size[j], dij = D[i * n + j];
    for (let k = 0; k < n; k++) {
      if (!alive[k] || k === i || k === j) continue;
      const v = Math.max(0, (ni * D[k * n + i] + nj * D[k * n + j]) / (ni + nj) - ni * nj * dij / ((ni + nj) * (ni + nj)));
      D[k * n + i] = D[i * n + k] = v;
    }
    alive[j] = 0; size[i] = ni + nj;
    merges.push({ a: i, b: j, dist: Math.sqrt(dij), size: size[i] });
    findNN(i);
    for (let k = 0; k < n; k++) {
      if (!alive[k] || k === i) continue;
      if (nn[k] === i || nn[k] === j) findNN(k);
      else if (D[k * n + i] < nnd[k]) { nnd[k] = D[k * n + i]; nn[k] = i; }
    }
  }
  return merges;
}

function cut(n, merges, steps) {
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  for (let m = 0; m < steps; m++) parent[find(merges[m].b)] = find(merges[m].a);
  const map = new Map(), labels = new Int32Array(n);
  for (let i = 0; i < n; i++) { const r = find(i); if (!map.has(r)) map.set(r, map.size); labels[i] = map.get(r); }
  return { labels, count: map.size };
}

function countLarge(labels, count, minSize) {
  const c = new Int32Array(count);
  for (const l of labels) c[l]++;
  let big = 0; for (const v of c) if (v >= minSize) big++;
  return { sizes: c, big };
}

export function cluster(items, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const withEmb = items.filter(it => it.emb);
  if (!withEmb.length) return { centroids: [], k: 0 };
  let pool = withEmb.filter(it => it.clean >= o.minClust);
  if (pool.length < 2) pool = withEmb.filter(it => it.dur >= o.minClust);
  if (pool.length < 2) pool = withEmb;
  if (pool.length > o.maxClusterItems) pool = [...pool].sort((a, b) => b.clean - a.clean).slice(0, o.maxClusterItems);
  const n = pool.length, d = pool[0].emb.length;
  const want = o.numSpeakers > 0 ? Math.min(o.numSpeakers, n) : 0;
  let labels, count;
  const minSize = Math.min(o.maxMinSize, Math.max(1, Math.round(0.1 * n)));
  if (n === 1) { labels = new Int32Array(1); count = 1; }
  else {
    const merges = linkage(pool.map(it => it.emb));
    let steps = 0;
    while (steps < merges.length && merges[steps].dist <= o.threshold) steps++;
    ({ labels, count } = cut(n, merges, steps));
    if (want) {
      let { big } = countLarge(labels, count, minSize);
      if (big !== want) {
        // pick the cut closest to the default threshold that yields the requested number of real clusters
        const order = [...merges.keys()].sort((x, y) => Math.abs(merges[x].dist - o.threshold) - Math.abs(merges[y].dist - o.threshold));
        let best = null, bestDiff = Infinity;
        for (const m of [-1, ...order]) {
          const c = cut(n, merges, m + 1);
          const b = countLarge(c.labels, c.count, minSize).big;
          const diff = Math.abs(b - want);
          if (diff < bestDiff) { bestDiff = diff; best = c; }
          if (diff === 0) break;
        }
        if (best) ({ labels, count } = best);
      }
    }
  }
  // fold small clusters into the nearest big one
  let { sizes, big } = countLarge(labels, count, minSize);
  const keep = [];
  for (let c = 0; c < count; c++) if (big === 0 || sizes[c] >= minSize) keep.push(c);
  if (want && keep.length > want) { keep.sort((a, b) => sizes[b] - sizes[a]); keep.length = want; }
  const centroids = keep.map(() => new Float32Array(d));
  const cnt = new Float64Array(keep.length);
  const idx = new Map(keep.map((c, i) => [c, i]));
  for (let i = 0; i < n; i++) {
    const t = idx.get(labels[i]); if (t === undefined) continue;
    const e = pool[i].emb; for (let x = 0; x < d; x++) centroids[t][x] += e[x]; cnt[t]++;
  }
  for (let t = 0; t < centroids.length; t++) {
    let nrm = 0; for (let x = 0; x < d; x++) { centroids[t][x] /= cnt[t]; nrm += centroids[t][x] ** 2; }
    nrm = Math.sqrt(nrm) || 1; for (let x = 0; x < d; x++) centroids[t][x] /= nrm;
  }
  return { centroids, k: centroids.length };
}

// ---------- stage 4: timeline ----------
export function buildTimeline(audioLen, wins, items, clustering, opts = {}) {
  const { centroids, k: K } = clustering;
  const G = Math.ceil(audioLen / STEP) + 2;
  const empty = { K: 0, G, active: [], turns: [], frameTime: (g) => (g * STEP + FRAME0) / SR };
  if (!K) return empty;
  // assign each (window, local speaker) to a cluster; two speakers in one window prefer different clusters
  const byWin = new Map();
  for (const it of items) { if (!byWin.has(it.w)) byWin.set(it.w, []); byWin.get(it.w).push(it); }
  const assign = new Map(); // "w:k" -> cluster
  for (const [w, list] of byWin) {
    const pairs = [];
    for (const it of list) {
      if (!it.emb) continue;
      for (let c = 0; c < K; c++) { let s = 0; const ce = centroids[c]; for (let x = 0; x < ce.length; x++) s += it.emb[x] * ce[x]; pairs.push([s, it.k, c]); }
    }
    pairs.sort((a, b) => b[0] - a[0]);
    const usedK = new Set(), usedC = new Set();
    for (const [, k, c] of pairs) if (!usedK.has(k) && !usedC.has(c)) { usedK.add(k); usedC.add(c); assign.set(w + ':' + k, c); }
    for (const [, k, c] of pairs) if (!usedK.has(k)) { usedK.add(k); assign.set(w + ':' + k, c); }
  }
  const score = Array.from({ length: K }, () => new Float32Array(G));
  const wsum = new Float32Array(G), nsum = new Float32Array(G);
  const pending = [];
  for (let w = 0; w < wins.length; w++) {
    const { start, F, valid, prob, act, nact } = wins[w];
    const g0 = Math.round(start / STEP);
    for (let f = 0; f < valid; f++) {
      const hw = 0.54 - 0.46 * Math.cos(2 * Math.PI * f / (F - 1));
      const g = g0 + f; if (g >= G) break;
      wsum[g] += hw; nsum[g] += hw * nact[f];
    }
    for (let k = 0; k < 3; k++) {
      const c = assign.get(w + ':' + k);
      if (c === undefined) { pending.push([w, k]); continue; }
      for (let f = 0; f < valid; f++) {
        const g = g0 + f; if (g >= G) break;
        const hw = 0.54 - 0.46 * Math.cos(2 * Math.PI * f / (F - 1));
        score[c][g] += hw * prob[k * F + f];
      }
    }
  }
  // local speakers too brief for a voice print: give them to whichever cluster is already talking there
  // Local speakers with no voice print (too brief, or in a window we skipped): match them to whoever
  // is already talking at those moments. Different local speakers in one window are different people,
  // so they may not share a cluster while another choice exists.
  const adds = [];
  const pendByWin = new Map();
  for (const [w, k] of pending) { if (!pendByWin.has(w)) pendByWin.set(w, []); pendByWin.get(w).push(k); }
  for (const [w, ks] of pendByWin) {
    const { start, F, valid, act } = wins[w];
    const g0 = Math.round(start / STEP);
    const usedC = new Set();
    for (let k = 0; k < 3; k++) { const c = assign.get(w + ':' + k); if (c !== undefined) usedC.add(c); }
    const pairs = [], lone = [];
    for (const k of ks) {
      const ov = new Float64Array(K); let n = 0, fsum = 0;
      for (let f = 0; f < valid; f++) if (act[k * F + f]) { n++; fsum += f; const g = g0 + f; if (g < G) for (let c = 0; c < K; c++) ov[c] += score[c][g]; }
      if (!n) continue;
      lone.push([k, ov, g0 + Math.round(fsum / n)]);
      for (let c = 0; c < K; c++) if (ov[c] > 0) pairs.push([ov[c] * (usedC.has(c) ? 0.25 : 1), k, c]);
    }
    pairs.sort((x, y) => y[0] - x[0]);
    const doneK = new Set(), takenC = new Set();
    for (const [, k, c] of pairs) if (!doneK.has(k) && !takenC.has(c) && !usedC.has(c)) { doneK.add(k); takenC.add(c); adds.push([w, k, c]); }
    for (const [k, ov, gc] of lone) {
      if (doneK.has(k)) continue;
      const free = []; for (let c = 0; c < K; c++) if (!usedC.has(c) && !takenC.has(c)) free.push(c);
      let pick = -1;
      if (free.length === 1) pick = free[0];
      else if (free.length > 1) {
        // the free speaker heard most recently around here
        const span = Math.round(15 * SR / STEP); let bd = Infinity;
        for (const c of free) for (let d = 0; d <= span; d++) { const a = gc - d, b = gc + d; if ((a >= 0 && score[c][a] > 0) || (b < G && score[c][b] > 0)) { if (d < bd) { bd = d; pick = c; } break; } }
      }
      if (pick < 0) { let bv = 0; for (let c = 0; c < K; c++) if (ov[c] > bv) { bv = ov[c]; pick = c; } }
      if (pick >= 0) { doneK.add(k); takenC.add(pick); adds.push([w, k, pick]); }
    }
  }
  for (const [w, k, bc] of adds) {
    const { start, F, valid, prob } = wins[w];
    const g0 = Math.round(start / STEP);
    for (let f = 0; f < valid; f++) {
      const g = g0 + f; if (g >= G) break;
      const hw = 0.54 - 0.46 * Math.cos(2 * Math.PI * f / (F - 1));
      score[bc][g] += hw * prob[k * F + f];
    }
  }
  let active = Array.from({ length: K }, () => new Uint8Array(G));
  const order = new Int32Array(K);
  for (let g = 0; g < G; g++) {
    if (!wsum[g]) continue;
    const cnt = Math.min(K, Math.round(nsum[g] / wsum[g]));
    if (cnt <= 0) continue;
    for (let c = 0; c < K; c++) order[c] = c;
    const ord = Array.from(order).sort((a, b) => score[b][g] - score[a][g]);
    for (let t = 0; t < cnt; t++) if (score[ord[t]][g] > 0) active[ord[t]][g] = 1;
  }
  // number speakers by order of first appearance, dropping any that never speak
  const first = active.map(a => { const i = a.indexOf(1); return i < 0 ? Infinity : i; });
  const perm = [...first.keys()].filter(c => first[c] < Infinity).sort((a, b) => first[a] - first[b]);
  active = perm.map(c => active[c]);
  const frameTime = (g) => Math.min(audioLen, g * STEP + FRAME0) / SR;
  const turns = [];
  for (let c = 0; c < active.length; c++) for (const [s, e] of runsOf(active[c], G)) turns.push({ start: s === 0 ? 0 : frameTime(s), end: frameTime(e), speaker: c });
  turns.sort((a, b) => a.start - b.start);
  return { K: active.length, G, active, turns, frameTime };
}

export async function diarize(audio, { segment, embed, numSpeakers = 0, opts = {}, onProgress } = {}) {
  const o = { ...DEFAULTS, ...opts, numSpeakers };
  const wins = await segmentAll(audio, segment, o, onProgress && ((d, t) => onProgress('listen', d, t)));
  const items = await embedAll(audio, wins, embed, o, onProgress && ((d, t) => onProgress('voices', d, t)));
  const clustering = cluster(items, o);
  return buildTimeline(audio.length, wins, items, clustering, o);
}

// ---------- using the timeline ----------
// Seconds each speaker is active inside [t0, t1].
export function speakerOverlap(tl, t0, t1) {
  const out = new Float64Array(tl.K);
  const g0 = Math.max(0, Math.floor((t0 * SR - FRAME0) / STEP)), g1 = Math.min(tl.G, Math.ceil((t1 * SR - FRAME0) / STEP));
  for (let c = 0; c < tl.K; c++) { let n = 0; for (let g = g0; g < Math.max(g1, g0 + 1) && g < tl.G; g++) n += tl.active[c][g]; out[c] = n * STEP / SR; }
  return out;
}

// Nearest active speaker to time t within maxGap seconds, or -1.
export function nearestSpeaker(tl, t, maxGap = 2) {
  const g = Math.max(0, Math.min(tl.G - 1, Math.round((t * SR - FRAME0) / STEP)));
  const lim = Math.round(maxGap * SR / STEP);
  for (let dlt = 0; dlt <= lim; dlt++) {
    for (const x of [g - dlt, g + dlt]) if (x >= 0 && x < tl.G) for (let c = 0; c < tl.K; c++) if (tl.active[c][x]) return c;
  }
  return -1;
}

// True if anyone is speaking between t0 and t1.
export function anySpeech(tl, t0, t1) {
  const o = speakerOverlap(tl, t0, t1);
  for (const v of o) if (v > 0) return true;
  return false;
}
