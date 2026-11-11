// Turns Whisper's word list into speaker-labelled transcript lines.
import { speakerOverlap, nearestSpeaker } from './diarize.js';

const SENT_END = /[.!?…。？！]["'”’)\]]?$/;

// Whisper's word list -> clean words with absolute times.
export function cleanWords(chunks, offset, segEnd) {
  const words = []; let depth = 0, since = 0, prevEnd = 0;
  for (const c of chunks || []) {
    const raw = c.text || '';
    // drop sound notes such as (audience laughing) or [MUSIC], however the words were split
    let t = '';
    for (const ch of raw) {
      if (ch === '(' || ch === '[') { depth++; since = 0; }
      else if (ch === ')' || ch === ']') depth = Math.max(0, depth - 1);
      else if (depth === 0) t += ch;
    }
    if (depth > 0 && ++since > 8) depth = 0;                            // unmatched bracket: give up on it
    t = t.trim().replace(/^[-–—]+\s*/, '').replace(/--+$/, '…');         // dialogue dashes
    if (!t || /^[-–—♪…,.!?]+$/.test(t)) continue;
    let s = c.timestamp && c.timestamp[0], e = c.timestamp && c.timestamp[1];
    if (s == null || !isFinite(s)) s = prevEnd;
    if (e == null || !isFinite(e) || e < s) e = s + 0.2;
    prevEnd = e;
    words.push({ text: t, sp: /^\s/.test(raw) || words.length === 0, start: Math.min(segEnd, offset + s), end: Math.min(segEnd, offset + e), spk: -1 });
  }
  return words;
}

// Give every word a speaker from the timeline, then tidy within sentences.
export function assignSpeakers(words, tl) {
  const K = tl.K;
  let prev = -1;
  for (const w of words) {
    // a word's end time is more trustworthy than its start, which can stretch back over a pause
    const ov = speakerOverlap(tl, Math.max(w.start, w.end - 0.6), Math.max(w.end, w.start + 0.02));
    let best = -1, bv = 0;
    for (let c = 0; c < K; c++) { const v = ov[c] + (c === prev ? 1e-4 : 0); if (ov[c] > 0 && v > bv) { bv = v; best = c; } }
    w.ov = ov; w.spk = best;
    if (best >= 0) prev = best;
  }
  for (const w of words) if (w.spk < 0) w.spk = nearestSpeaker(tl, (w.start + w.end) / 2, 1.5);
  for (let i = 1; i < words.length; i++) if (words[i].spk < 0) words[i].spk = words[i - 1].spk;
  for (let i = words.length - 2; i >= 0; i--) if (words[i].spk < 0) words[i].spk = words[i + 1].spk;
  let i = 0;
  while (i < words.length) {
    let j = i;
    while (j < words.length - 1 && !SENT_END.test(words[j].text) && words[j + 1].start - words[j].end <= 0.7) j++;
    const tot = new Float64Array(K); let sum = 0;
    for (let x = i; x <= j; x++) {
      const w = words[x]; let any = false;
      for (let c = 0; c < K; c++) if (w.ov[c] > 0) { tot[c] += w.ov[c]; sum += w.ov[c]; any = true; }
      if (!any && w.spk >= 0) { const d = Math.max(0.05, w.end - w.start); tot[w.spk] += d; sum += d; }
    }
    let top = -1, tv = 0;
    for (let c = 0; c < K; c++) if (tot[c] > tv) { tv = tot[c]; top = c; }
    const n = j - i + 1;
    if (top >= 0 && (tv / sum >= 0.65 || n <= 3)) { for (let x = i; x <= j; x++) words[x].spk = top; }
    else {
      // a real change of speaker inside the sentence: keep it, but no one-word flips
      let x = i;
      while (x <= j) {
        let y = x; while (y < j && words[y + 1].spk === words[x].spk) y++;
        if (y - x + 1 < 2) { const to = x > i ? words[x - 1].spk : (y < j ? words[y + 1].spk : words[x].spk); for (let z = x; z <= y; z++) words[z].spk = to; }
        x = y + 1;
      }
    }
    i = j + 1;
  }
}

// perSentence: one line per sentence (used when speakers are labelled, so a wrong label is easy to fix)
export function groupSegments(words, perSentence = false) {
  const segs = []; let cur = null;
  for (const w of words) {
    const brk = cur && (w.spk !== cur.speaker
      || (cur.sentEnd && (perSentence || cur.end - cur.start >= 4 || cur.n >= 12 || w.start - cur.end > 0.9))
      || cur.n >= 70);
    if (!cur || brk) { cur = { start: w.start, end: w.end, text: w.text, speaker: w.spk, n: 1 }; segs.push(cur); }
    else { cur.text += (w.sp ? ' ' : '') + w.text; cur.end = Math.max(cur.end, w.end); cur.n++; }
    cur.sentEnd = SENT_END.test(w.text);
  }
  return segs.map(({ start, end, text, speaker }) => ({ start, end, text, speaker }));
}
