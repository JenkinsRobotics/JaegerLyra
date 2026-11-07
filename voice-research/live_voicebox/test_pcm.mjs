import test from 'node:test';
import assert from 'node:assert/strict';
import {PcmBytes} from './public/pcm-bytes.js';
import {PcmRing} from './public/pcm-ring.js';

test('PCM decoding survives arbitrary byte boundaries and rejects incomplete data',()=>{
  const expected=new Float32Array([.125,-.75,0,.9,-.001]),bytes=new Uint8Array(expected.buffer),parser=new PcmBytes(),actual=[];
  let at=0;for(const size of [1,2,3,7,2,5]){actual.push(...parser.push(bytes.slice(at,at+size)));at+=size;}
  parser.finish();assert.deepEqual(actual,[...expected]);
  const incomplete=new PcmBytes();incomplete.push(bytes.slice(0,3));assert.throws(()=>incomplete.finish());
  assert.throws(()=>new PcmBytes().push(new Uint8Array(new Float32Array([NaN]).buffer)));
});
test('prebuffer prevents premature output and ring wrapping preserves sample order',()=>{
  const ring=new PcmRing(24000,64,512);ring.enqueue(new Float32Array(63).fill(.25));const silent=new Float32Array(20);ring.pull(silent);
  assert.ok(silent.every(x=>x===0));assert.equal(ring.consumed,0);
  ring.enqueue(new Float32Array(193).fill(.25));const first=new Float32Array(256);ring.pull(first);
  assert.ok(first.slice(240).every(x=>x===.25));assert.equal(ring.underruns,0);
  const next=Float32Array.from({length:400},(_,i)=>i/1000);ring.enqueue(next);const output=new Float32Array(400);ring.pull(output);
  assert.deepEqual(output,next);assert.equal(ring.consumed,656);assert.equal(ring.underruns,0);
});
test('underrun is counted once, recovers, and drains a short ended clip',()=>{
  const ring=new PcmRing(24000,960,4800);ring.enqueue(new Float32Array(960).fill(.2));ring.pull(new Float32Array(1300));ring.pull(new Float32Array(128));
  assert.equal(ring.underruns,1);ring.enqueue(new Float32Array(300).fill(.3));ring.end();const output=new Float32Array(700);ring.pull(output);
  assert.ok(output.some(x=>x>0));assert.ok(output.slice(-100).every(x=>x===0));assert.ok(ring.drained);assert.equal(ring.consumed,1260);
});
test('stop clears queued audio and fades promptly to silence',()=>{
  const ring=new PcmRing();ring.enqueue(new Float32Array(2000).fill(.4));ring.pull(new Float32Array(400));ring.stop();
  ring.enqueue(new Float32Array(1000).fill(.8));const output=new Float32Array(500);ring.pull(output);
  assert.ok(output.slice(240).every(x=>x===0));assert.equal(ring.count,0);assert.ok(ring.data.every(x=>x===0));assert.ok(ring.drained);
});
test('overflow and nonfinite input fail rather than silently lose samples',()=>{
  const ring=new PcmRing(24000,10,20);assert.throws(()=>ring.enqueue(new Float32Array(21)));
  assert.throws(()=>ring.enqueue(new Float32Array([Infinity])));assert.equal(ring.count,0);
});
test('worklet handles changing render sizes, mirrored channels and termination',async()=>{
  let Processor;const messages=[];
  globalThis.sampleRate=24000;globalThis.AudioWorkletProcessor=class{constructor(){this.port={postMessage:m=>messages.push(m)};}};
  globalThis.registerProcessor=(name,klass)=>{Processor=klass;};await import('./public/pcm-player.js');
  const p=new Processor();p.port.onmessage({data:{type:'audio',samples:new Float32Array(2400).fill(.2)}});p.port.onmessage({data:{type:'end'}});
  let running=true;
  for(let i=0;i<40&&running;i++){const size=[64,128,256][i%3],a=new Float32Array(size),b=new Float32Array(size);running=p.process([],[[a,b]]);assert.deepEqual(a,b);}
  assert.equal(running,false);assert.equal(messages.filter(m=>m.type==='drained').length,1);assert.equal(p.ring.consumed,2400);
});
