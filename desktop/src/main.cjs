// Electron main process: opens the window and runs transcription in background worker processes.
const { app, BrowserWindow, ipcMain, utilityProcess, shell, Menu } = require('electron');
const path = require('node:path');
const fs = require('node:fs');

const MODELS = app.isPackaged ? path.join(process.resourcesPath, 'models') : path.join(__dirname, '..', 'models');
let win = null, coordinator = null;

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

ipcMain.handle('run', async (_e, msg) => { const c = await getCoordinator(); c.start(msg); return true; });
ipcMain.handle('cancel', async () => { const c = await getCoordinator(); c.cancel(); return true; });
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
if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => { if (win) { if (win.isMinimized()) win.restore(); win.focus(); } });
  app.whenReady().then(async () => {
    if (!fs.existsSync(MODELS)) console.error('Models folder missing: ' + MODELS);
    createWindow();
    (await getCoordinator()).warmUp();   // load the speech model in the background so the first run starts sooner
  });
  app.on('window-all-closed', async () => { if (coordinator) coordinator.cancel(); app.quit(); });
}
