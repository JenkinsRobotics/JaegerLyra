import test from 'node:test';
import assert from 'node:assert/strict';
import {VoicePair} from './public/voice-pair.js';

function setup(fetchOverride){
  const requests=[],sources=[];
  const context={currentTime:4,destination:{},resume:async()=>{},close:async()=>{},
    decodeAudioData:async data=>({duration:30,sampleRate:48000,key:data.key}),
    createGain(){return {gain:{value:1,setTargetAtTime(v){this.value=v;}},connect(){}};},
    createBufferSource(){const source={connect(gain){this.gain=gain;},disconnect(){},
      start(when,offset){this.when=when;this.offset=offset;},stop(){this.stopped=true;}};sources.push(source);return source;}};
  const fetcher=async url=>{requests.push(url);return fetchOverride?fetchOverride(url):{ok:true,arrayBuffer:async()=>({key:url})};};
  const player=new VoicePair({urls:{lead:'/lead.wav',reply:'/reply.wav'},duration:30,contextFactory:()=>context,fetcher});
  return {player,context,requests,sources};
}

test('two separate buffers start on exactly the same clock and offset',async()=>{
  const {player,requests,sources}=setup();await player.seek(8);await player.play();
  assert.deepEqual(requests,['/lead.wav','/reply.wav']);assert.equal(sources.length,2);
  assert.notEqual(sources[0].buffer,sources[1].buffer);assert.notEqual(sources[0].gain,sources[1].gain);
  assert.equal(sources[0].when,sources[1].when);assert.equal(sources[0].offset,8);assert.equal(sources[1].offset,8);
  assert.equal(player.snapshot().activeSources,2);
});
test('solo and volume changes leave both sources running',async()=>{
  const {player,sources}=setup();await player.play();
  player.setVoice('reply',{enabled:false});player.setVoice('lead',{level:.35});
  assert.equal(sources[0].gain.gain.value,.35);assert.equal(sources[1].gain.gain.value,0);
  assert.equal(sources.length,2);assert.ok(sources.every(s=>!s.stopped));
  player.setVoice('reply',{enabled:true});assert.equal(sources[1].gain.gain.value,1);
});
test('seek reschedules both voices at the same offset without fetching again',async()=>{
  const {player,context,sources,requests}=setup();await player.play();context.currentTime=10;
  await player.seek(17.5);
  assert.ok(sources.slice(0,2).every(s=>s.stopped));assert.equal(sources.length,4);
  assert.equal(sources[2].when,sources[3].when);assert.equal(sources[2].offset,17.5);assert.equal(sources[3].offset,17.5);
  assert.equal(requests.length,2);
});
test('pause preserves shared position and stop resets both',async()=>{
  const {player,context,sources}=setup();await player.play();context.currentTime=9.04;player.pause();
  assert.ok(Math.abs(player.position-5)<1e-10);assert.equal(player.playing,false);
  await player.play();assert.ok(Math.abs(sources[2].offset-5)<1e-10);player.stop();
  assert.equal(player.position,0);assert.equal(player.snapshot().activeSources,0);
  assert.ok(sources.every(s=>s.stopped));
});
test('stop during loading cannot start late audio',async()=>{
  let release;const delayed=new Promise(resolve=>{release=resolve;});
  const {player,sources}=setup(async url=>{await delayed;return {ok:true,arrayBuffer:async()=>({key:url})};});
  const pending=player.play();player.stop();release();await pending;
  assert.equal(sources.length,0);assert.equal(player.playing,false);assert.equal(player.loading,false);
});
test('both sources must finish before completion; output mute is independent',async()=>{
  const {player,sources}=setup();player.setMuted(true);await player.play();assert.equal(player.monitor.gain.value,0);
  sources[0].onended();assert.equal(player.ended,false);sources[1].onended();
  assert.equal(player.ended,true);assert.equal(player.playing,false);assert.equal(player.position,30);
  assert.equal(player.snapshot().activeSources,0);player.setMuted(false);assert.equal(player.monitor.gain.value,1);
});
test('a missing voice fails the pair without playing a partial performance',async()=>{
  const {player,sources}=setup(async url=>({ok:url!=='/reply.wav',arrayBuffer:async()=>({key:url})}));
  await assert.rejects(player.play(),/reply/);assert.equal(sources.length,0);assert.equal(player.loading,false);
});
