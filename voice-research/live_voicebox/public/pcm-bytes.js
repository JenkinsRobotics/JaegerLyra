export class PcmBytes {
  constructor() { this.carry = new Uint8Array(0); }
  push(chunk) {
    if (!(chunk instanceof Uint8Array)) throw new Error('Invalid audio packet');
    const joined = new Uint8Array(this.carry.length + chunk.length);
    joined.set(this.carry); joined.set(chunk, this.carry.length);
    const count = Math.floor(joined.length / 4), view = new DataView(joined.buffer);
    const output = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      output[i] = view.getFloat32(i * 4, true);
      if (!Number.isFinite(output[i])) throw new Error('Invalid waveform received');
    }
    this.carry = joined.slice(count * 4);
    return output;
  }
  finish() { if (this.carry.length) throw new Error('Incomplete audio packet'); }
}
