// Taps the microphone and hands the raw samples to the window in ~100 ms batches.
class MicTap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(1600); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.buf[this.n++] = ch[i];
        if (this.n === this.buf.length) { this.port.postMessage(this.buf); this.buf = new Float32Array(1600); this.n = 0; }
      }
    }
    return true;
  }
}
registerProcessor('mic-tap', MicTap);
