import { fork } from 'node:child_process';
async function trial(procs, th){
  const kids=Array.from({length:procs},(_,p)=>fork('prof4.mjs',[th,p,procs],{stdio:['ignore','ignore','ignore','ipc']}));
  await Promise.all(kids.map(k=>new Promise(r=>k.once('message',r))));
  const t=Date.now(); kids.forEach(k=>k.send('go'));
  await Promise.all(kids.map(k=>new Promise(r=>k.once('message',r))));
  kids.forEach(k=>k.kill()); return Date.now()-t;
}
console.log('1 engine x 2 threads, 4 windows:', await trial(1,2),'ms');
console.log('2 engines x 1 thread, 4 windows:', await trial(2,1),'ms');
