const { contextBridge, ipcRenderer, webUtils } = require('electron');

let listener = null;
ipcRenderer.on('engine', (_e, m) => { if (listener) listener(m); });

contextBridge.exposeInMainWorld('engine', {
  run: (msg) => ipcRenderer.invoke('run', msg),
  cancel: () => ipcRenderer.invoke('cancel'),
  onMessage: (fn) => { listener = fn; },
  openExternal: (url) => ipcRenderer.invoke('open-external', url),
  store: (op, ...args) => ipcRenderer.invoke('store', op, ...args),
  readAudio: (p) => ipcRenderer.invoke('read-audio', p),
  saveDocx: (data) => ipcRenderer.invoke('save-docx', data),
  find: (text, opts) => ipcRenderer.invoke('find', text, opts),
  pathOf: (file) => { try { return webUtils.getPathForFile(file) || null; } catch { return null; } },
});
