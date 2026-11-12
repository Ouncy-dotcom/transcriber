// Everything the app keeps on this computer: recent transcripts, saved voices, Word export.
// All of it lives in the app's own data folder (%APPDATA%\Transcriber on Windows).
const fs = require('node:fs');
const path = require('node:path');

function createStore(dir) {
  const tdir = path.join(dir, 'transcripts');
  const voicesFile = path.join(dir, 'voices.json');
  fs.mkdirSync(tdir, { recursive: true });

  const safeId = (id) => String(id).replace(/[^a-zA-Z0-9_-]/g, '');
  const writeJson = (file, data) => {
    const tmp = file + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(data));
    fs.renameSync(tmp, file);   // never leave a half-written file behind
  };
  const readJson = (file, fallback) => { try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch { return fallback; } };

  // ---------- transcripts ----------
  function listTranscripts() {
    const out = [];
    for (const f of fs.readdirSync(tdir)) {
      if (!f.endsWith('.json')) continue;
      const t = readJson(path.join(tdir, f), null);
      if (!t || !t.id) continue;
      const words = (t.segs || []).reduce((n, s) => n + (s.text ? s.text.split(/\s+/).filter(Boolean).length : 0), 0);
      const speakers = new Set((t.segs || []).map(s => s.speaker).filter(k => k >= 0)).size;
      out.push({ id: t.id, title: t.title, created: t.created, updated: t.updated, duration: t.duration, words, speakers, names: t.names || [] });
    }
    return out.sort((a, b) => (b.updated || b.created || 0) - (a.updated || a.created || 0));
  }
  const getTranscript = (id) => readJson(path.join(tdir, safeId(id) + '.json'), null);
  function saveTranscript(t) {
    if (!t || !t.id) throw new Error('missing id');
    t.updated = Date.now();
    writeJson(path.join(tdir, safeId(t.id) + '.json'), t);
    return true;
  }
  function deleteTranscript(id) { try { fs.unlinkSync(path.join(tdir, safeId(id) + '.json')); } catch {} return true; }

  // ---------- saved voices ----------
  const listVoices = () => readJson(voicesFile, []);
  function saveVoice(name, emb) {
    name = String(name || '').trim().slice(0, 60);
    if (!name || !Array.isArray(emb) || emb.length < 16) return listVoices();
    const voices = listVoices();
    const v = voices.find(x => x.name.toLowerCase() === name.toLowerCase());
    if (v) {
      // blend with what we already know of this voice, so it gets more reliable over time
      const n = Math.min(v.count || 1, 9);
      const mixed = v.emb.map((x, i) => (x * n + emb[i]) / (n + 1));
      const norm = Math.sqrt(mixed.reduce((a, b) => a + b * b, 0)) || 1;
      v.emb = mixed.map(x => x / norm); v.count = (v.count || 1) + 1; v.updated = Date.now();
    } else {
      const norm = Math.sqrt(emb.reduce((a, b) => a + b * b, 0)) || 1;
      voices.push({ name, emb: emb.map(x => x / norm), count: 1, created: Date.now(), updated: Date.now() });
    }
    writeJson(voicesFile, voices);
    return voices;
  }
  function deleteVoice(name) { writeJson(voicesFile, listVoices().filter(v => v.name !== name)); return listVoices(); }
  function renameVoice(from, to) {
    to = String(to || '').trim().slice(0, 60);
    const voices = listVoices(); const v = voices.find(x => x.name === from);
    if (v && to && !voices.some(x => x !== v && x.name.toLowerCase() === to.toLowerCase())) { v.name = to; writeJson(voicesFile, voices); }
    return listVoices();
  }

  return { listTranscripts, getTranscript, saveTranscript, deleteTranscript, listVoices, saveVoice, deleteVoice, renameVoice };
}

// ---------- Word document ----------
async function buildDocx({ title, created, duration, turns, stamped }) {
  const d = await import('docx');
  const { Document, Packer, Paragraph, TextRun, HeadingLevel } = d;
  const fmt = (s) => { s = Math.max(0, s || 0); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = Math.floor(s % 60); return (h ? h + ':' + String(m).padStart(2, '0') : m) + ':' + String(x).padStart(2, '0'); };
  const when = created ? new Date(created).toLocaleString('en-ZA', { dateStyle: 'long', timeStyle: 'short' }) : '';
  const kids = [
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(title || 'Transcript')] }),
    new Paragraph({ children: [new TextRun({ text: [when, duration ? 'Length ' + fmt(duration) : ''].filter(Boolean).join('  ·  '), color: '6D6A63', size: 20 })] }),
    new Paragraph({ children: [] }),
  ];
  for (const t of turns) {
    const runs = [];
    if (stamped) runs.push(new TextRun({ text: '[' + fmt(t.start) + ']  ', color: '9A3412', size: 18 }));
    if (t.name) runs.push(new TextRun({ text: t.name + ':  ', bold: true }));
    runs.push(new TextRun(t.text));
    kids.push(new Paragraph({ children: runs, spacing: { after: 160 } }));
  }
  const doc = new Document({
    creator: 'Transcriber', title: title || 'Transcript',
    styles: { default: { document: { run: { font: 'Calibri', size: 22 } } } },
    sections: [{ children: kids }],
  });
  return Packer.toBuffer(doc);
}

module.exports = { createStore, buildDocx };
