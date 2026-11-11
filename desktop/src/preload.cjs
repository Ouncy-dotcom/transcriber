const { contextBridge, ipcRenderer } = require('electron');

let listener = null;
ipcRenderer.on('engine', (_e, m) => { if (listener) listener(m); });

contextBridge.exposeInMainWorld('engine', {
  run: (msg) => ipcRenderer.invoke('run', msg),
  cancel: () => ipcRenderer.invoke('cancel'),
  onMessage: (fn) => { listener = fn; },
  openExternal: (url) => ipcRenderer.invoke('open-external', url),
});
