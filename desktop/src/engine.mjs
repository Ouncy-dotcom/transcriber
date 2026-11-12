// Worker process. Several of these run side by side; each one holds its own copy of the models
// and handles whatever job the coordinator gives it (speaker detection, or one 30-second chunk).
import path from 'node:path';
import { env, pipeline, AutoProcessor, AutoModel, AutoModelForAudioFrameClassification } from '@huggingface/transformers';
import { diarize } from './diarize.js';
import { cleanWords } from './words.js';
import { makeConfidenceRecorder, attachConfidence, isLoop, collapseRepeats, compressionRatio, dropInvented, fixDriftedWords } from './quality.mjs';

const MODELS = process.env.TRANSCRIBER_MODELS;
const threads = Math.max(1, +process.env.TRANSCRIBER_THREADS || 1);
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = MODELS.endsWith(path.sep) ? MODELS : MODELS + path.sep;
env.useFSCache = false;
const session_options = { intraOpNumThreads: threads, interOpNumThreads: 1 };

let transcriber = null, seg = null, emb = null;
let port = null;
const post = (type, data = {}) => port.postMessage({ type, ...data });

let lastSequence = null;
const conf = makeConfidenceRecorder();

async function loadTranscriber() {
  if (transcriber) return;
  transcriber = await pipeline('automatic-speech-recognition', 'whisper-small_timestamped', {
    dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' },
    device: 'cpu',
    session_options,
  });
  // keep the token ids of each run so word confidence can be matched to words
  const generate = transcriber.model.generate.bind(transcriber.model);
  transcriber.model.generate = async (args) => { const out = await generate(args); lastSequence = out.sequences ? out.sequences.tolist()[0] : null; return out; };
}

// One pass of Whisper over a chunk, returning clean words with confidence.
async function transcribeOnce(audio, msg, extra = {}) {
  conf.reset(); lastSequence = null;
  const opts = { return_timestamps: 'word', task: 'transcribe', logits_processor: [conf.recorder], ...extra };
  if (msg.language) opts.language = msg.language;
  const out = await transcriber(audio, opts);
  try { if (lastSequence) attachConfidence(out.chunks, lastSequence, conf.probs, transcriber.tokenizer); } catch {}
  return fixDriftedWords(cleanWords(out.chunks, msg.offset, msg.segEnd));
}

async function loadSpeakerModels() {
  if (seg && emb) return;
  const segId = 'pyannote-segmentation-3.0', embId = 'wespeaker-voxceleb-resnet34-LM';
  const sp = await AutoProcessor.from_pretrained(segId);
  const sm = await AutoModelForAudioFrameClassification.from_pretrained(segId, { dtype: 'fp32', device: 'cpu', session_options });
  const ep = await AutoProcessor.from_pretrained(embId);
  const em = await AutoModel.from_pretrained(embId, { dtype: 'q8', device: 'cpu', session_options });
  seg = async (buf) => { const { logits } = await sm(await sp(buf)); return { data: logits.data, dims: logits.dims }; };
  emb = async (buf) => { const out = await em(await ep(buf)); return (out.embeds ?? out.last_hidden_state ?? Object.values(out)[0]).data; };
}

const asF32 = (a) => (a instanceof Float32Array ? a : new Float32Array(a.buffer ?? a));

async function handle(msg) {
  if (msg.type === 'load') {
    if (msg.what === 'speech') await loadTranscriber(); else await loadSpeakerModels();
    post('loaded', { what: msg.what });
  } else if (msg.type === 'diarize') {
    await loadSpeakerModels();
    const tl = await diarize(asF32(msg.audio), {
      segment: seg, embed: emb, numSpeakers: +msg.speakers || 0,
      onProgress: (phase, done, total) => post('diar', { phase, done, total }),
    });
    post('timeline', { tl: { K: tl.K, G: tl.G, active: tl.active, turns: tl.turns, voiceprints: tl.voiceprints } });
  } else if (msg.type === 'chunk') {
    await loadTranscriber();
    const audio = asF32(msg.audio);
    let words = await transcribeOnce(audio, msg);
    let fixed = 0;
    if (isLoop(words)) {
      // Whisper got stuck repeating itself: try again with settings that discourage repeats
      const retry = await transcribeOnce(audio, msg, { no_repeat_ngram_size: 4, repetition_penalty: 1.15 });
      const score = (w) => compressionRatio(w.map(x => x.text).join(' '));
      if (retry.length && (!isLoop(retry) || score(retry) < score(words))) words = retry;
      if (isLoop(words)) words = collapseRepeats(words);
      fixed++;
    }
    const before = words.length;
    words = dropInvented(words, audio, msg.offset);
    post('words', { i: msg.i, loopFixed: fixed, dropped: before - words.length, words: words.map(({ text, sp, start, end, p }) => ({ text, sp, start, end, p })) });
  }
}

// Works as an Electron utility process (process.parentPort) and as a plain Node child (tests).
const onMessage = async (raw) => {
  const msg = raw && raw.data !== undefined && raw.ports !== undefined ? raw.data : raw;
  if (!msg || !msg.type) return;
  try { await handle(msg); }
  catch (err) { post('error', { text: String(err && err.message || err), job: msg.type, i: msg.i }); }
};
if (process.parentPort) { port = process.parentPort; port.on('message', onMessage); }
else { port = { postMessage: (m) => process.send(m) }; process.on('message', onMessage); }
post('ready', { threads });
