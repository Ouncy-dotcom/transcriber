// Electron main process: opens the window and runs the engine in a background process.
const { app, BrowserWindow, ipcMain, utilityProcess, shell, Menu } = require('electron');
const path = require('node:path');
const fs = require('node:fs');

const isPackaged = app.isPackaged;
const MODELS = isPackaged ? path.join(process.resourcesPath, 'models') : path.join(__dirname, '..', 'models');
let win = null, engine = null, engineReady = null;

function startEngine() {
  engine = utilityProcess.fork(path.join(__dirname, 'engine.mjs'), [], {
    env: { ...process.env, TRANSCRIBER_MODELS: MODELS },
    serviceName: 'Transcriber engine',
    stdio: 'pipe',
  });
  engine.stdout?.on('data', (d) => process.stdout.write('[engine] ' + d));
  engine.stderr?.on('data', (d) => process.stderr.write('[engine] ' + d));
  engineReady = new Promise((resolve) => {
    const h = (m) => { if (m && m.type === 'ready') { resolve(); } };
    engine.on('message', h);
  });
  engine.on('message', (m) => { if (win && !win.isDestroyed() && m.type !== 'ready') win.webContents.send('engine', m); });
  engine.on('exit', (code) => {
    const crashed = engine && !engine._stopping;
    engine = null;
    if (crashed && win && !win.isDestroyed()) win.webContents.send('engine', { type: 'error', text: 'The transcription engine stopped unexpectedly (code ' + code + '). If this keeps happening, the computer may be running out of memory.' });
  });
}

function stopEngine() {
  if (engine) { engine._stopping = true; engine.kill(); engine = null; }
}

ipcMain.handle('run', async (_e, msg) => {
  if (!engine) startEngine();
  await engineReady;
  engine.postMessage(msg);
  return true;
});
ipcMain.handle('cancel', async () => { stopEngine(); return true; });
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
  win.loadFile(path.join(__dirname, 'index.html'));
}

Menu.setApplicationMenu(null);
const single = app.requestSingleInstanceLock();
if (!single) app.quit();
else {
  app.on('second-instance', () => { if (win) { if (win.isMinimized()) win.restore(); win.focus(); } });
  app.whenReady().then(() => {
    if (!fs.existsSync(MODELS)) console.error('Models folder missing: ' + MODELS);
    createWindow();
    startEngine(); // warm up in the background so the first run starts quicker
  });
  app.on('window-all-closed', () => { stopEngine(); app.quit(); });
}
