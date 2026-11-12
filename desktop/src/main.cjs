// Electron main process: opens the window and runs transcription in background worker processes.
const { app, BrowserWindow, ipcMain, utilityProcess, shell, Menu, dialog, session } = require('electron');
const { createStore, buildDocx } = require('./store.cjs');
const path = require('node:path');
const fs = require('node:fs');

const MODELS = app.isPackaged ? path.join(process.resourcesPath, 'models') : path.join(__dirname, '..', 'models');
let win = null, coordinator = null, store = null;
// tests can point the app at a throwaway data folder
if (process.env.TRANSCRIBER_DATA) app.setPath('userData', process.env.TRANSCRIBER_DATA);
// tests feed a sound file in as if it were the microphone
if (process.env.TRANSCRIBER_FAKE_MIC) {
  app.commandLine.appendSwitch('use-fake-device-for-media-stream');
  app.commandLine.appendSwitch('use-fake-ui-for-media-stream');
  app.commandLine.appendSwitch('use-file-for-fake-audio-capture', process.env.TRANSCRIBER_FAKE_MIC + '%noloop');
}

const send = (m) => { if (win && !win.isDestroyed()) win.webContents.send('engine', m); };

function spawnWorker(threads) {
  const p = utilityProcess.fork(path.join(__dirname, 'engine.mjs'), [], {
    env: { ...process.env, TRANSCRIBER_MODELS: MODELS, TRANSCRIBER_THREADS: String(threads) },
    serviceName: 'Transcriber worker',
    stdio: 'pipe',
  });
  p.stdout?.on('data', (d) => process.stdout.write('[worker] ' + d));
  p.stderr?.on('data', (d) => process.stderr.write('[worker] ' + d));
  return {
    send: (m) => p.postMessage(m),
    onMessage: (fn) => p.on('message', fn),
    onExit: (fn) => p.on('exit', fn),
    kill: () => { try { p.kill(); } catch {} },
  };
}

async function getCoordinator() {
  if (!coordinator) {
    const { createCoordinator } = await import('./coordinator.mjs');
    coordinator = createCoordinator(spawnWorker, send);
  }
  return coordinator;
}

ipcMain.handle('run', async (_e, msg) => {
  const c = await getCoordinator();
  msg.voices = store.listVoices();      // so saved voices can be recognised
  c.start(msg);
  return true;
});

// recent transcripts and saved voices
ipcMain.handle('store', async (_e, op, ...args) => {
  const ops = ['listTranscripts', 'getTranscript', 'saveTranscript', 'deleteTranscript', 'listVoices', 'saveVoice', 'deleteVoice', 'renameVoice'];
  if (!ops.includes(op)) throw new Error('unknown store op');
  return store[op](...args);
});

// the original recording, for playback when a saved transcript is reopened
const AUDIO_EXT = /\.(mp3|wav|m4a|aac|ogg|oga|opus|flac|webm|mp4|m4v|mov|mkv|wma|amr|3gp|aif|aiff|caf)$/i;
ipcMain.handle('read-audio', async (_e, p) => {
  if (typeof p !== 'string' || !AUDIO_EXT.test(p) || !fs.existsSync(p)) return null;
  const b = fs.readFileSync(p);
  return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
});

ipcMain.handle('save-docx', async (_e, data) => {
  const buf = await buildDocx(data);
  let file = process.env.TRANSCRIBER_SAVE_PATH;
  if (!file) {
    const r = await dialog.showSaveDialog(win, { title: 'Save as Word document', defaultPath: (data.title || 'Transcript').replace(/[\\/:*?"<>|]/g, '') + '.docx', filters: [{ name: 'Word document', extensions: ['docx'] }] });
    if (r.canceled || !r.filePath) return null;
    file = r.filePath;
  }
  fs.writeFileSync(file, buf);
  return file;
});

ipcMain.handle('find', async (_e, text, opts) => {
  if (!win) return;
  if (!text) { win.webContents.stopFindInPage('clearSelection'); return; }
  win.webContents.findInPage(text, opts || {});
});
ipcMain.handle('cancel', async () => { const c = await getCoordinator(); c.cancel(); return true; });

// live transcription from the microphone
ipcMain.handle('live-start', async (_e, msg) => { const c = await getCoordinator(); msg.voices = store.listVoices(); c.liveStart(msg); return true; });
ipcMain.handle('live-chunk', async (_e, msg) => { const c = await getCoordinator(); c.liveChunk(msg); return true; });
ipcMain.handle('live-finish', async (_e, msg) => { const c = await getCoordinator(); c.liveFinish(msg); return true; });
ipcMain.handle('save-recording', async (_e, bytes, ext) => {
  const dir = path.join(app.getPath('userData'), 'recordings');
  fs.mkdirSync(dir, { recursive: true });
  const stamp = new Date().toISOString().slice(0, 19).replace('T', ' ').replace(/:/g, '-');
  const file = path.join(dir, 'Recording ' + stamp + '.' + (ext === 'ogg' ? 'ogg' : 'webm'));
  fs.writeFileSync(file, Buffer.from(bytes));
  return file;
});
ipcMain.handle('export-recording', async (_e, src) => {
  if (typeof src !== 'string' || !fs.existsSync(src) || !src.startsWith(path.join(app.getPath('userData'), 'recordings'))) return null;
  const r = await dialog.showSaveDialog(win, { title: 'Save recording', defaultPath: path.basename(src), filters: [{ name: 'Audio', extensions: [path.extname(src).slice(1)] }] });
  if (r.canceled || !r.filePath) return null;
  fs.copyFileSync(src, r.filePath);
  return r.filePath;
});
ipcMain.handle('open-external', async (_e, url) => { if (/^https:\/\//.test(url)) await shell.openExternal(url); });

function createWindow() {
  win = new BrowserWindow({
    width: 1000, height: 820, minWidth: 560, minHeight: 500,
    title: 'Transcriber',
    icon: path.join(__dirname, 'icon.png'),
    backgroundColor: '#f6f4ef',
    autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, sandbox: true, nodeIntegration: false, spellcheck: true },
  });
  win.webContents.setWindowOpenHandler(({ url }) => { if (/^https:\/\//.test(url)) shell.openExternal(url); return { action: 'deny' }; });
  win.webContents.on('will-navigate', (e) => e.preventDefault());
  win.webContents.on('found-in-page', (_e, r) => send({ type: 'found', matches: r.matches, active: r.activeMatchOrdinal }));
  win.loadFile(path.join(__dirname, 'index.html'));
}

Menu.setApplicationMenu(null);
if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => { if (win) { if (win.isMinimized()) win.restore(); win.focus(); } });
  app.whenReady().then(async () => {
    store = createStore(app.getPath('userData'));
    // the microphone is only ever allowed for audio, and only for the app's own window
    const allowMic = (wc, permission, details) => permission === 'media' && wc === win?.webContents && (!details?.mediaTypes || details.mediaTypes.every(t => t === 'audio'));
    session.defaultSession.setPermissionRequestHandler((wc, permission, cb, details) => cb(allowMic(wc, permission, details)));
    session.defaultSession.setPermissionCheckHandler((wc, permission) => permission === 'media' && wc === win?.webContents);
    if (!fs.existsSync(MODELS)) console.error('Models folder missing: ' + MODELS);
    createWindow();
    (await getCoordinator()).warmUp();   // load the speech model in the background so the first run starts sooner
  });
  app.on('window-all-closed', async () => { if (coordinator) coordinator.cancel(); app.quit(); });
}
