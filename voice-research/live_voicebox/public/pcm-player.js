import {PcmRing} from './pcm-ring.js';
class VoiceboxPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    if (sampleRate !== 24000) throw new Error('The live voicebox requires a 24 kHz audio context');
    this.ring = new PcmRing(); this.ticks = 0; this.reportedEnd = false;
    this.port.onmessage = ({data}) => {
      try {
        if (data.type === 'audio') this.ring.enqueue(data.samples);
        else if (data.type === 'end') this.ring.end();
        else if (data.type === 'stop') this.ring.stop();
      } catch (error) { this.ring.stop(); this.port.postMessage({type:'error',message:error.message}); }
    };
  }
  process(inputs, outputs) {
    const channels = outputs[0]; if (!channels?.length) return true;
    this.ring.pull(channels[0]);
    for (let i = 1; i < channels.length; i++) channels[i].set(channels[0]);
    if (++this.ticks % 10 === 0) this.port.postMessage({type:'stats',...this.ring.stats(),wave:Array.from(channels[0])});
    if (this.ring.drained && !this.reportedEnd) { this.reportedEnd = true; this.port.postMessage({type:'drained',...this.ring.stats()}); }
    return !this.ring.drained;
  }
}
registerProcessor('voicebox-player',VoiceboxPlayer);
