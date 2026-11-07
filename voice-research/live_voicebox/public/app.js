import {PcmBytes} from './pcm-bytes.js';
const $ = id => document.getElementById(id);
let catalog, unit, phrase, run = null, revealed = false, queued = null, sending = false, sequence = 0;
const players = [$('audio-a'), $('audio-b')];
const names = ['C','C♯','D','D♯','E','F','F♯','G','G♯','A','A♯','B'];
const shortcuts = ['a','s','d','f','g','h','j','k'];
const shapeIds=['resonance','brightness','pace','vibrato'];
const shapeValues=()=>({resonance:Number($('resonance').value)/100,brightness:Number($('brightness').value)/10,pace:Number($('pace').value)/100,vibrato:Number($('vibrato').value)});
function displayShape(){
  const s=shapeValues();
  $('resonance-value').textContent=s.resonance===1?'Natural':(s.resonance>1?'+':'')+Math.round((s.resonance-1)*100)+'%';
  $('brightness-value').textContent=s.brightness===0?'Natural':(s.brightness>0?'+':'')+s.brightness.toFixed(1)+' dB/oct';
  $('pace-value').textContent=$('playback').value==='syllable'?'Whole words / phrases':s.pace.toFixed(2)+'×';
  $('vibrato-value').textContent=s.vibrato?s.vibrato+' cents · 5 Hz':'Off';
  document.querySelectorAll('[data-shape]').forEach(b=>b.removeAttribute('aria-pressed'));
}
function setShape(values){shapeIds.forEach(id=>$(id).value=values[id]);displayShape();}
const pitchBounds = () => $('playback').value==='word'&&unit ? [Math.max(60,unit.baseHz/2),Math.min(800,unit.baseHz*2)] : [60,800];
const hz = () => {const [low,high]=pitchBounds();return low*(high/low)**(Number($('pitch').value)/1000);};
function noteName(frequency) { const midi = Math.round(69 + 12 * Math.log2(frequency / 440)); return names[(midi % 12 + 12) % 12] + (Math.floor(midi / 12) - 1); }
function status(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
function displayPitch(frequency = hz()) { $('note').textContent = noteName(frequency); $('hz-label').textContent = Math.round(frequency) + ' Hz'; $('pitch-value').textContent = noteName(frequency) + ' · ' + Math.round(frequency) + ' Hz'; }
function setPitch(frequency) { const [low,high]=pitchBounds();$('pitch').value = 1000 * Math.log(Math.min(high,Math.max(low,frequency)) / low) / Math.log(high / low); displayPitch(); control({hz:hz()}); }
function interfaceState() {
  const busy = !!run, vocal = run?.mode === 'syllable', ready = !!catalog;
  for (const id of ['voice','word','playback','start','speech']) $(id).disabled = !ready || busy;
  $('phrase').disabled = !ready; $('reveal').disabled = !ready;
  $('pitch').disabled = !ready || (busy && !vocal); $('volume').disabled = !ready;
  shapeIds.forEach(id=>$(id).disabled=!ready||busy||(id==='pace'&&$('playback').value==='syllable'));
  $('reset-shape').disabled=!ready||busy;
  document.querySelectorAll('[data-shape]').forEach(b=>b.disabled=!ready||busy);
  $('stop').disabled = !busy; $('release').disabled = !vocal || !run?.id || run?.released;
  document.querySelectorAll('#keys button').forEach(button => button.disabled = !vocal || !run?.id || run?.released);
  const wholeWord=$('playback').value==='word';
  $('start').textContent=wholeWord?'Reconstruct word':'Hold a note';
  $('release').hidden=wholeWord;document.querySelector('.buttons').classList.toggle('word-playback',wholeWord);
  $('keys').hidden=wholeWord;
  $('stage-caption').textContent=wholeWord?'The complete word plays once and finishes automatically.':'Hold a note, then play the keys. A · S · D · F · G · H · J · K';
  $('playback-help').textContent=wholeWord?'Plays the full word and finishes automatically. Choose pitch and voice character before playback; volume can change while it plays.':'Intentionally sustains the vowel until you press Finish word. Pitch and volume can change live; voice character is chosen before starting. Stops after one minute.';
  displayShape();
}
function chooseUnit() {
  unit = catalog.units.find(u => u.voice === $('voice').value && u.word === $('word').value);
  $('voice-label').textContent = unit.voice.toUpperCase() + ' / ' + unit.word.toUpperCase();
  const midi = Math.round(69 + 12 * Math.log2(unit.baseHz / 440)); $('keys').replaceChildren();
  [0,2,4,5,7,9,11,12].forEach((offset,i) => {
    const f = 440 * 2 ** ((midi + offset - 69) / 12), button = document.createElement('button');
    button.type = 'button'; button.dataset.hz = f; button.setAttribute('aria-label', 'Play ' + noteName(f));
    button.innerHTML = noteName(f) + '<small>' + shortcuts[i].toUpperCase() + '</small>';
    button.addEventListener('click', () => playNote(button)); $('keys').append(button);
  });
  setPitch(unit.baseHz); interfaceState();
}
function playNote(button) { if (button.disabled) return; setPitch(Number(button.dataset.hz)); document.querySelectorAll('#keys button').forEach(b => b.classList.toggle('active', b === button)); }
async function api(path, body) {
  const response = await fetch(path, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Local engine request failed'); return result;
}
function control(change) {
  if (!run?.id || run.ending) return;
  queued = {...queued,...change,sequence:++sequence}; sendControls();
}
async function sendControls() {
  if (sending) return; sending = true;
  try {
    while (queued && run?.id && !run.ending) {
      const current = run, body = queued; queued = null;
      await api('/api/control', {id:current.id,...body});
      if (run !== current) break;
    }
  } catch (error) { if (run && !run.ending) status(error.message, true); }
  finally { sending = false; if(queued&&run?.id&&!run.ending)queueMicrotask(sendControls); }
}
function draw(wave = []) {
  const canvas = $('scope'), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);ctx.strokeStyle='#b9a4c31c';ctx.lineWidth=1;
  for (const y of [.2,.5,.8]) {ctx.beginPath();ctx.moveTo(0,h*y);ctx.lineTo(w,h*y);ctx.stroke();}
  ctx.beginPath();ctx.strokeStyle='#d4badf';ctx.lineWidth=2;
  const data = wave.length ? wave : Array(128).fill(0);
  data.forEach((v,i) => {const x=i/(data.length-1)*w,y=h/2-v*h*.8;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  document.querySelector('.orb').style.setProperty('--level', Math.min(1,Math.max(0,...data.map(Math.abs))*3));
}
async function finish(current, message = 'Ready for another performance.') {
  if (run !== current || current.ending) return;
  current.ending = true; queued = null;clearInterval(current.poll);
  current.abort.abort(); current.node?.port.postMessage({type:'stop'});
  if (current.id) api('/api/stop',{id:current.id}).catch(()=>{});
  await new Promise(resolve=>setTimeout(resolve,25));
  current.node?.disconnect(); current.node?.port.close(); await current.context?.close().catch(()=>{});
  if (run === current) {run=null; $('state').textContent='Ready';$('queue').textContent='0 ms';interfaceState();draw();status(message);}
}
async function start(mode) {
  if (run) return;players.forEach(player=>player.pause());
  const current={mode,id:null,startedAt:performance.now(),abort:new AbortController(),ending:false,released:false};
  run=current;sequence=0;queued=null;interfaceState();$('state').textContent='Starting';status('Starting the local output engine…');
  for (const id of ['first-packet','queue','underruns','compute']) $(id).textContent='—';
  try {
    const context=new AudioContext({sampleRate:catalog.sampleRate,latencyHint:'interactive'});current.context=context;
    await context.resume(); if (context.sampleRate!==catalog.sampleRate) throw new Error('This browser cannot open the live audio rate. The listening comparisons still work.');
    await context.audioWorklet.addModule('/pcm-player.js');
    if (run!==current || current.ending) return;
    const node=new AudioWorkletNode(context,'voicebox-player',{numberOfInputs:0,numberOfOutputs:1,outputChannelCount:[1]});current.node=node;
    const monitor=context.createGain();monitor.gain.value=$('mute').checked?0:1;current.monitor=monitor;node.connect(monitor);monitor.connect(context.destination);
    node.onprocessorerror=()=>finish(current,'Audio playback stopped unexpectedly. Try starting again.');
    node.port.onmessage=({data})=>{
      if(run!==current || current.ending)return;
      if(data.type==='error')finish(current,data.message);
      if(data.type==='stats'||data.type==='drained'){$('queue').textContent=Math.round(data.queuedSamples/catalog.sampleRate*1000)+' ms';$('underruns').textContent=data.underruns;draw(data.wave);}
      if(data.type==='drained')finish(current,mode==='word'?'Word finished.':'Performance finished.');
    };
    const body={mode,item:mode==='speech'?phrase.id:unit.id,gain:Number($('volume').value)/100,shape:shapeValues()};
    if(mode==='syllable')body.shape.pace=1;
    if(mode!=='speech')body.hz=hz();
    const created=await api('/api/session',body);current.id=created.id;
    if(current.ending || run!==current){api('/api/stop',{id:current.id}).catch(()=>{});return;}
    interfaceState();$('state').textContent=mode==='syllable'?'Opening':'Speaking';
    status(mode==='speech'?'Lyra is reconstructing the captured phrase with your voice-character settings.':mode==='word'?'Reconstructing the complete word. It will finish automatically.':'Hold mode is on. Change pitch while the vowel holds; Finish word releases its ending.');
    current.poll=setInterval(async()=>{
      try{const response=await fetch('/api/status?id='+current.id);const data=await response.json();
        if(run!==current||current.ending)return;
        if(data.packetP95Ms!==null&&data.packetP95Ms!==undefined)$('compute').textContent=data.packetP95Ms.toFixed(2)+' ms · p95';
        if(data.releaseRequested || data.closed)$('state').textContent='Finishing';else if(data.holding)$('state').textContent='Holding';
      }catch{}
    },500);
    const response=await fetch('/api/stream?id='+current.id,{signal:current.abort.signal});
    if(!response.ok)throw new Error((await response.json()).error || 'Could not open audio stream');
    const reader=response.body.getReader(),parser=new PcmBytes();let first=true;
    while(true){const {value,done}=await reader.read();if(done)break;if(current.ending)break;
      const samples=parser.push(value);if(!samples.length)continue;
      if(first){$('first-packet').textContent=Math.round(performance.now()-current.startedAt)+' ms';first=false;}
      node.port.postMessage({type:'audio',samples},[samples.buffer]);
    }
    parser.finish();if(!current.ending)node.port.postMessage({type:'end'});
  }catch(error){if(!current.ending){await finish(current,error.message);status(error.message,true);}}
}
function choosePhrase() {
  phrase=catalog.phrases[Number($('phrase').value)];revealed=false;$('phrase-text').textContent='“'+phrase.text+'”';
  players.forEach((player,i)=>{player.pause();player.src=phrase.choices[i].url;player.load();$('label-'+['a','b'][i]).textContent='Label hidden';});
  $('reveal').textContent='Reveal source / reconstruction';
}
$('voice').addEventListener('change',chooseUnit);$('word').addEventListener('change',chooseUnit);
$('playback').addEventListener('change',chooseUnit);
shapeIds.forEach(id=>$(id).addEventListener('input',displayShape));
$('reset-shape').addEventListener('click',()=>{setShape({resonance:100,brightness:0,pace:100,vibrato:0});setPitch(unit.baseHz);});
const presets={warm:{resonance:92,brightness:-15,pace:94,vibrato:0},light:{resonance:110,brightness:13,pace:105,vibrato:0},expressive:{resonance:100,brightness:0,pace:93,vibrato:25}};
document.querySelectorAll('[data-shape]').forEach(button=>button.addEventListener('click',()=>{setShape(presets[button.dataset.shape]);button.setAttribute('aria-pressed','true');}));
$('pitch').addEventListener('input',()=>{displayPitch();control({hz:hz()});});
$('volume').addEventListener('input',()=>{$('volume-value').textContent=$('volume').value+'%';control({gain:Number($('volume').value)/100});});
$('mute').addEventListener('change',()=>{if(run?.monitor)run.monitor.gain.setValueAtTime($('mute').checked?0:1,run.context.currentTime);});
$('start').addEventListener('click',()=>start($('playback').value));$('speech').addEventListener('click',()=>start('speech'));
$('stop').addEventListener('click',()=>run&&finish(run,'Stopped.'));
$('release').addEventListener('click',()=>{if(!run?.id)return;run.released=true;control({release:true});interfaceState();$('state').textContent='Finishing';});
$('phrase').addEventListener('change',choosePhrase);
$('reveal').addEventListener('click',()=>{revealed=!revealed;players.forEach((_,i)=>$('label-'+['a','b'][i]).textContent=revealed?phrase.choices[i].label:'Label hidden');$('reveal').textContent=revealed?'Hide labels':'Reveal source / reconstruction';});
players.forEach(player=>player.addEventListener('play',()=>{players.forEach(other=>{if(other!==player)other.pause();});if(run)finish(run,'Listening comparison.');}));
document.addEventListener('keydown',event=>{
  if(event.key==='Escape'&&run){event.preventDefault();finish(run,'Stopped.');return;}
  if(event.repeat||event.metaKey||event.ctrlKey||event.altKey||['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName)||event.target.isContentEditable)return;
  const index=shortcuts.indexOf(event.key.toLowerCase());if(index>=0&&run?.mode==='syllable'){event.preventDefault();playNote($('keys').children[index]);}
});
window.addEventListener('pagehide',()=>{if(run){run.abort.abort();run.context?.close();if(run.id)fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:run.id}),keepalive:true}).catch(()=>{});}});
draw();
try{
  const response=await fetch('/api/catalog');if(!response.ok)throw new Error('The local engine is unavailable');catalog=await response.json();
  for(const voice of [...new Set(catalog.units.map(u=>u.voice))])$('voice').add(new Option(voice,voice));
  for(const word of [...new Set(catalog.units.map(u=>u.word))])$('word').add(new Option(word,word));
  catalog.phrases.forEach((p,i)=>$('phrase').add(new Option(p.voice+' · Phrase '+(i%3+1),i)));
  chooseUnit();choosePhrase();$('connection').textContent='Lyra decoder ready';$('connection').classList.add('ready');status('Ready to reconstruct. Audio starts only when you press a playback button.');interfaceState();
}catch(error){status(error.message,true);$('connection').textContent='Engine unavailable';}
