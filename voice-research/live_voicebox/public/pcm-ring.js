export class PcmRing {
  constructor(rate = 24000, prebuffer = 960, capacity = 48000) {
    if (!Number.isInteger(rate) || rate <= 0 || !Number.isInteger(prebuffer) || prebuffer < 1 || !Number.isInteger(capacity) || capacity < prebuffer) throw new Error('Invalid audio buffer');
    this.rate = rate; this.prebuffer = prebuffer; this.data = new Float32Array(capacity);
    this.read = 0; this.write = 0; this.count = 0; this.playing = false;
    this.ended = false; this.stopped = false; this.drained = false;
    this.underruns = 0; this.maximumQueued = 0; this.consumed = 0;
    this.fadeLength = Math.max(1, Math.round(rate * .01));
    this.fadeIn = 0; this.fadeOut = 0; this.fadeStart = 0; this.last = 0;
  }
  enqueue(samples) {
    if (this.stopped || this.ended) return;
    if (!(samples instanceof Float32Array) || samples.some(v => !Number.isFinite(v))) throw new Error('Invalid waveform');
    if (samples.length > this.data.length - this.count) throw new Error('Audio arrived faster than it could be played');
    for (const value of samples) { this.data[this.write] = value; this.write = (this.write + 1) % this.data.length; }
    this.count += samples.length; this.maximumQueued = Math.max(this.maximumQueued, this.count);
  }
  end() { this.ended = true; }
  stop() {
    this.stopped = true; this.ended = true; this.count = 0; this.playing = false;
    this.data.fill(0); this.fadeStart = this.last; this.fadeOut = this.fadeLength;
  }
  pull(output) {
    for (let i = 0; i < output.length; i++) {
      let value = 0;
      if (!this.fadeOut && !this.playing && !this.stopped && (this.count >= this.prebuffer || (this.ended && this.count))) {
        this.playing = true; this.fadeIn = this.fadeLength;
      }
      if (!this.fadeOut && this.playing && !this.count) {
        this.playing = false; this.fadeStart = this.last; this.fadeOut = this.fadeLength;
        if (!this.ended) this.underruns++;
      }
      if (this.fadeOut) {
        value = this.fadeStart * (--this.fadeOut / this.fadeLength);
      } else if (this.playing && this.count) {
        value = this.data[this.read]; this.data[this.read] = 0;
        this.read = (this.read + 1) % this.data.length; this.count--; this.consumed++;
        if (this.fadeIn) value *= 1 - (--this.fadeIn / this.fadeLength);
      }
      output[i] = value; this.last = value;
      if (this.ended && !this.count && !this.playing && !this.fadeOut) this.drained = true;
    }
  }
  stats() { return {queuedSamples:this.count, maximumQueuedSamples:this.maximumQueued, consumedSamples:this.consumed, underruns:this.underruns, drained:this.drained}; }
}
