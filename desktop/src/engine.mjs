// Worker process. Several of these run side by side; each one holds its own copy of the models
// and handles whatever job the coordinator gives it (speaker detection, or one 30-second chunk).
import path from 'node:path';
import { env, pipeline, AutoProcessor, AutoModel, AutoModelForAudioFrameClassification } from '@huggingface/transformers';
import { diarize } from './diarize.js';
import { cleanWords } from './words.js';

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

async function loadTranscriber() {
  if (transcriber) return;
  transcriber = await pipeline('automatic-speech-recognition', 'whisper-small_timestamped', {
    dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' },
    device: 'cpu',
    session_options,
  });
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
    post('timeline', { tl: { K: tl.K, G: tl.G, active: tl.active, turns: tl.turns } });
  } else if (msg.type === 'chunk') {
    await loadTranscriber();
    const opts = { return_timestamps: 'word', task: 'transcribe' };
    if (msg.language) opts.language = msg.language;
    const out = await transcriber(asF32(msg.audio), opts);
    post('words', { i: msg.i, words: cleanWords(out.chunks, msg.offset, msg.segEnd).map(({ text, sp, start, end }) => ({ text, sp, start, end })) });
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
