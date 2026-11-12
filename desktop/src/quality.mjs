// Accuracy helpers: how sure the model is about each word, and guards against Whisper's
// two classic mistakes (getting stuck repeating itself, and inventing words in silence).
import zlib from 'node:zlib';

// ---------- word confidence ----------
// A logits processor that records how likely the chosen token was at every step (greedy decoding,
// so the chosen token is the most likely one after Whisper's own rules have been applied).
export function makeConfidenceRecorder() {
  const probs = [];
  const recorder = (input_ids, logits) => {
    const d = logits.data, V = logits.dims[logits.dims.length - 1];
    let mx = -Infinity;
    for (let i = 0; i < V; i++) if (d[i] > mx) mx = d[i];
    let sum = 0;
    for (let i = 0; i < V; i++) { const v = d[i]; if (v !== -Infinity) sum += Math.exp(v - mx); }
    probs.push(sum > 0 ? 1 / sum : 1);
    return logits;
  };
  return { recorder, probs, reset: () => { probs.length = 0; } };
}

// Map token probabilities onto the word chunks the pipeline returned. A word's confidence is
// the lowest confidence of the tokens it is made of.
export function attachConfidence(chunks, seqTokens, probs, tokenizer) {
  if (!chunks || !chunks.length || !probs.length) return chunks;
  const gen = seqTokens.slice(seqTokens.length - probs.length).map(Number);  // tokens chosen during generation
  const special = new Set(tokenizer.all_special_ids ?? []);
  const toks = [], tp = [];
  gen.forEach((t, i) => { if (!special.has(t) && t < 50257) { toks.push(t); tp.push(probs[i]); } });
  if (!toks.length) return chunks;
  // character span of every token inside the decoded text
  const ends = [];
  for (let i = 0; i < toks.length; i++) ends.push(tokenizer.decode(toks.slice(0, i + 1), { skip_special_tokens: true }).length);
  const full = tokenizer.decode(toks, { skip_special_tokens: true });
  let pos = 0;
  for (const c of chunks) {
    const w = (c.text || '').trim();
    if (!w) continue;
    const at = full.indexOf(w, pos);
    if (at < 0) continue;
    const a = at, b = at + w.length;
    pos = b;
    let p = 1, start = 0;
    for (let i = 0; i < toks.length; i++) {
      const s = start, e = ends[i]; start = e;
      if (e > a && s < b) p = Math.min(p, tp[i]);
    }
    c.p = p;
  }
  return chunks;
}

// ---------- repetition loops ----------
// Whisper's own test: text that compresses unusually well is usually a loop.
export function compressionRatio(text) {
  const b = Buffer.from(text, 'utf8');
  if (b.length < 40) return 1;
  return b.length / zlib.deflateSync(b).length;
}

// Longest run of a phrase repeated back to back, e.g. "I don't know. I don't know. I don't know."
export function maxRepeat(words) {
  const t = words.map(w => w.text.toLowerCase().replace(/[^\p{L}\p{N}']/gu, ''));
  let best = { count: 1, n: 1, at: 0 };
  for (let n = 1; n <= 8; n++) {
    for (let i = 0; i + 2 * n <= t.length; i++) {
      let count = 1;
      while (i + (count + 1) * n <= t.length) {
        let same = true;
        for (let k = 0; k < n; k++) if (t[i + k] !== t[i + count * n + k] || !t[i + k]) { same = false; break; }
        if (!same) break;
        count++;
      }
      if (count > best.count || (count === best.count && n > best.n && count > 1)) best = { count, n, at: i };
    }
  }
  return best;
}

export function isLoop(words) {
  if (words.length < 6) return false;
  const r = maxRepeat(words);
  const text = words.map(w => w.text).join(' ');
  // a word said 4+ times in a row, or a phrase 3+ times, or text that compresses like a loop
  return (r.n === 1 && r.count >= 4) || (r.n > 1 && r.count >= 3) || compressionRatio(text) > 2.4;
}

// Keep the first two copies of any phrase repeated 3+ times in a row (people do say "no, no").
export function collapseRepeats(words) {
  let out = words;
  for (let guard = 0; guard < 20; guard++) {
    const r = maxRepeat(out);
    const limit = r.n === 1 ? 3 : 2;
    if (r.count <= limit) break;
    out = [...out.slice(0, r.at + limit * r.n), ...out.slice(r.at + r.count * r.n)];
  }
  return out;
}

// ---------- invented words in silence ----------
const STOCK = /^(thanks? (you )?(so much )?for watching|please (like and )?subscribe|subtitles? by|subtitled by|transcribed by|amara\.org|thank you\.?$)/i;

function rmsOf(audio, s, e) {
  s = Math.max(0, s); e = Math.min(audio.length, e);
  let sum = 0; for (let i = s; i < e; i++) sum += audio[i] * audio[i];
  return Math.sqrt(sum / Math.max(1, e - s));
}

// Drop words that sit in near-silence and that the model wasn't sure about, plus Whisper's
// stock "thanks for watching" style phrases when they are not backed by real speech.
export function dropInvented(words, audio, offset, sr = 16000) {
  if (!words.length) return words;
  const level = words.map(w => rmsOf(audio, Math.round((w.start - offset) * sr), Math.round((Math.max(w.end, w.start + 0.15) - offset) * sr)));
  const sorted = [...level].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] || 0;
  const keep = words.map((w, i) => !(level[i] < 0.12 * median && (w.p ?? 1) < 0.5));
  // stock phrases: check the whole phrase
  const text = words.map(w => w.text).join(' ');
  for (let i = 0; i < words.length; i++) {
    const rest = words.slice(i).map(w => w.text).join(' ');
    const m = STOCK.exec(rest);
    if (!m) continue;
    const nWords = m[0].trim().split(/\s+/).length;
    const span = level.slice(i, i + nWords);
    const avg = span.reduce((a, b) => a + b, 0) / Math.max(1, span.length);
    const conf = Math.min(...words.slice(i, i + nWords).map(w => w.p ?? 1));
    // "thank you" is often real; only drop it when it's quiet and the model was unsure
    if (avg < 0.35 * median || (conf < 0.4 && !/^thank you/i.test(m[0])) || (words.length === nWords && conf < 0.6 && avg < 0.6 * median)) {
      for (let k = i; k < i + nWords; k++) keep[k] = false;
    }
  }
  void text;
  return words.filter((_, i) => keep[i]);
}

// With a speaker timeline: words with no speech anywhere near them are not real.
export function dropOutsideSpeech(words, hasSpeech) {
  return words.filter(w => hasSpeech(w.start - 1.0, Math.max(w.end, w.start + 0.1) + 1.0) || (w.p ?? 1) >= 0.7);
}

// ---------- audio cleanup ----------
// 4th-order high-pass at 80 Hz (removes mains hum at 50 Hz and rumble), then gentle automatic
// volume levelling so a quiet or distant speaker is lifted towards the level of a near one.
export function cleanAudio(input, sr = 16000) {
  const out = Float32Array.from(input);
  // remove DC offset
  let mean = 0; for (let i = 0; i < out.length; i++) mean += out[i]; mean /= Math.max(1, out.length);
  for (let i = 0; i < out.length; i++) out[i] -= mean;
  // two cascaded 2nd-order Butterworth high-pass sections
  const f0 = 80, Q = [0.5412, 1.3066];
  for (const q of Q) {
    const w0 = 2 * Math.PI * f0 / sr, cos = Math.cos(w0), alpha = Math.sin(w0) / (2 * q);
    const a0 = 1 + alpha;
    const b0 = (1 + cos) / 2 / a0, b1 = -(1 + cos) / a0, b2 = (1 + cos) / 2 / a0, a1 = -2 * cos / a0, a2 = (1 - alpha) / a0;
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < out.length; i++) {
      const x = out[i];
      const y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1; x1 = x; y2 = y1; y1 = y; out[i] = y;
    }
  }
  // levelling: gain per 100 ms block towards a target level, only where there is speech
  const B = Math.round(0.1 * sr), nb = Math.ceil(out.length / B);
  const lv = new Float32Array(nb);
  for (let b = 0; b < nb; b++) lv[b] = rmsOf(out, b * B, (b + 1) * B);
  const sorted = Float32Array.from(lv).sort();
  const floor = sorted[Math.floor(nb * 0.1)] || 0;
  const loud = sorted[Math.floor(nb * 0.95)] || 0;
  if (loud <= 0) return out;
  const target = Math.min(0.1, Math.max(0.03, loud * 0.6));
  // smooth level over ~0.5 s so the gain doesn't pump on every syllable
  const sm = new Float32Array(nb);
  for (let b = 0; b < nb; b++) { let s = 0, n = 0; for (let k = Math.max(0, b - 2); k <= Math.min(nb - 1, b + 2); k++) { s += lv[k]; n++; } sm[b] = s / n; }
  const gain = new Float32Array(nb);
  let g = 1;
  for (let b = 0; b < nb; b++) {
    const speech = sm[b] > Math.max(floor * 3, 0.002);
    const want = speech ? Math.min(6, Math.max(1, target / sm[b])) : Math.min(g, 1.5);   // never boost background noise much
    g = want > g ? g + (want - g) * 0.5 : g + (want - g) * 0.3;
    gain[b] = g;
  }
  for (let i = 0; i < out.length; i++) {
    const p = i / B - 0.5, b = Math.max(0, Math.min(nb - 1, Math.floor(p))), b2 = Math.min(nb - 1, b + 1), f = Math.max(0, Math.min(1, p - b));
    let v = out[i] * (gain[b] * (1 - f) + gain[b2] * f);
    if (v > 0.98) v = 0.98 + (v - 0.98) * 0.1; else if (v < -0.98) v = -0.98 + (v + 0.98) * 0.1;   // soft limit
    out[i] = v;
  }
  return out;
}

// Whisper's word timing sometimes leaves the first word of a sentence stranded seconds earlier,
// in the silence before the speech (or the last word stranded after it). Move such words next
// to the words they belong with.
const ENDS = /[.!?…]["'”’)\]]?$/;
export function fixDriftedWords(words) {
  const ws = words.map(w => ({ ...w }));
  // nobody takes more than ~1.5 s to say one word: an over-long word has swallowed a silence.
  // Keep the end next to the following word (or the start, for the last word of a sentence).
  for (let i = 0; i < ws.length; i++) {
    const w = ws[i], max = 1.2 + 0.06 * w.text.length;
    if (w.end - w.start > max) {
      const d = Math.min(0.6, 0.12 + 0.05 * w.text.length);
      if (ENDS.test(w.text) && i > 0) w.end = w.start + d; else w.start = w.end - d;
    }
  }
  for (let i = 0; i < ws.length; i++) {
    const w = ws[i], prev = ws[i - 1], next = ws[i + 1];
    const gapNext = next ? next.start - w.end : Infinity, gapPrev = prev ? w.start - prev.end : Infinity;
    const startsSentence = !prev || ENDS.test(prev.text);
    if (next && gapNext > 1.0 && !ENDS.test(w.text) && (startsSentence || gapPrev > gapNext)) {
      const d = Math.min(0.4, Math.max(0.15, w.end - w.start));
      w.end = Math.max(next.start - 0.02, 0); w.start = Math.max(w.end - d, prev ? prev.end : 0);
    } else if (prev && gapPrev > 1.0 && ENDS.test(w.text) && gapNext > gapPrev && !ENDS.test(prev.text)) {
      const d = Math.min(0.5, Math.max(0.15, w.end - w.start));
      w.start = prev.end + 0.02; w.end = w.start + d;
    }
  }
  return ws;
}
