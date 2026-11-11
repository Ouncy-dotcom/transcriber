// Downloads the speech and speaker models into ./models (run once before building).
import fs from 'node:fs';
import path from 'node:path';

const HF = 'https://huggingface.co/onnx-community/';
const MODELS = {
  'whisper-small_timestamped': ['config.json', 'generation_config.json', 'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'normalizer.json', 'onnx/encoder_model_quantized.onnx', 'onnx/decoder_model_merged_quantized.onnx'],
  'pyannote-segmentation-3.0': ['config.json', 'preprocessor_config.json', 'onnx/model.onnx'],
  'wespeaker-voxceleb-resnet34-LM': ['config.json', 'preprocessor_config.json', 'onnx/model_quantized.onnx'],
};
const root = path.resolve('models');
for (const [id, files] of Object.entries(MODELS)) {
  for (const f of files) {
    const dest = path.join(root, id, f);
    if (fs.existsSync(dest) && fs.statSync(dest).size > 0) continue;
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    let ok = false;
    for (let i = 0; i < 4 && !ok; i++) {
      try {
        const r = await fetch(HF + id + '/resolve/main/' + f);
        if (r.status === 404 && !f.startsWith('onnx/') && f !== 'config.json') { console.log('skip (not in repo)', id, f); ok = true; break; }
        if (!r.ok) throw new Error('HTTP ' + r.status);
        fs.writeFileSync(dest + '.tmp', Buffer.from(await r.arrayBuffer()));
        fs.renameSync(dest + '.tmp', dest);
        console.log('got', id, f, (fs.statSync(dest).size / 1e6).toFixed(1) + ' MB');
        ok = true;
      } catch (e) { console.log('retry', id, f, e.message); await new Promise(r => setTimeout(r, 2000 * (i + 1))); }
    }
    if (!ok) { console.error('FAILED to download', id, f); process.exit(1); }
  }
}
console.log('models ready in', root);
