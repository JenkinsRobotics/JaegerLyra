import {VoicePair} from './voice-pair.js';
const byId=id=>document.getElementById(id);
let manifest,player,frame=null;
const clock=s=>`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
function message(text){byId('duet-status').textContent=text;}
function update(){
  if(!player)return;
  const state=player.snapshot(),t=state.position;
  byId('duet-time').textContent=`${clock(t)} / ${clock(state.duration)}`;
  byId('duet-seek').value=t;byId('duet-seek').setAttribute('aria-valuetext',clock(t));
  byId('duet-score').style.setProperty('--playhead',`${Math.min(100,t/state.duration*100)}%`);
  const active=manifest.phrases.filter(p=>p.start_s<=t&&p.end_s>t&&state.enabled[p.part]);
  byId('duet-now').textContent=active.length?active.map(p=>(p.part==='lead'?'Heart: ':'Michael: ')+p.text).join('  /  '):(state.ended?'Duet finished.':t===0?'Ready for the first response.':'A breath between phrases.');
  document.querySelectorAll('.duet-phrase').forEach(b=>b.classList.toggle('active',active.some(p=>p.id===b.dataset.phrase)));
  const part=state.enabled.lead?(state.enabled.reply?'duet':'lead'):(state.enabled.reply?'reply':'none');
  document.querySelectorAll('[data-part]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.part===part)));
  for(const key of ['lead','reply']){
    byId('duet-'+key+'-mute').textContent=state.enabled[key]?'Mute':'Unmute';
    byId('duet-'+key+'-mute').setAttribute('aria-pressed',String(!state.enabled[key]));
    byId('duet-'+key+'-level-value').textContent=Math.round(state.levels[key]*100)+'%';
    byId('duet-'+key+'-state').textContent=!state.enabled[key]?'Muted':state.playing?'Playing':state.loading?'Loading':'Ready';
  }
  byId('duet-play').textContent=state.loading?'Loading both voices…':state.playing?'Ⅱ Pause':'▶ Play voices';
  byId('duet-play').disabled=state.loading;
  byId('duet-mute').textContent=state.muted?'Unmute output':'Mute output';
  byId('duet-mute').setAttribute('aria-pressed',String(state.muted));
  byId('duet-transport-note').textContent=state.playing?'Two separate tracks running in sync':'Two separate tracks · synchronized playback';
  if(state.ended)message('Both vocal tracks finished automatically.');
  if(state.playing&&frame===null)frame=requestAnimationFrame(()=>{frame=null;update();});
}
async function play(){
  if(!player)return;
  if(player.playing){player.pause();message('Paused both voices.');return;}
  window.dispatchEvent(new Event('lyra-duet-play'));
  message('Preparing the two independent vocal tracks…');
  try{await player.play();if(player.playing)message('Playing separate lead and response tracks together.');}
  catch(error){message(error.message);}
}
function stop(){if(player){player.stop();message('Stopped both voices. Ready to replay.');}}
async function seek(seconds){try{await player.seek(seconds);}catch(error){message(error.message);}}
document.querySelectorAll('[data-part]').forEach(button=>button.addEventListener('click',()=>{
  if(!player)return;const part=button.dataset.part;
  player.setVoice('lead',{enabled:part!=='reply'});player.setVoice('reply',{enabled:part!=='lead'});
}));
for(const key of ['lead','reply']){
  byId('duet-'+key+'-mute').addEventListener('click',()=>player?.setVoice(key,{enabled:!player.enabled[key]}));
  byId('duet-'+key+'-level').addEventListener('input',event=>player?.setVoice(key,{level:Number(event.target.value)/100}));
}
byId('duet-play').addEventListener('click',play);byId('duet-stop').addEventListener('click',stop);
byId('duet-mute').addEventListener('click',()=>player?.setMuted(!player.muted));
byId('duet-seek').addEventListener('input',event=>seek(Number(event.target.value)));
window.addEventListener('lyra-other-play',()=>{if(player&&(player.playing||player.loading)){player.pause();message('Paused both voices for other playback.');}});
window.addEventListener('pagehide',()=>{if(frame!==null)cancelAnimationFrame(frame);player?.close().catch(()=>{});});
document.addEventListener('keydown',event=>{if(event.key==='Escape')stop();});
try{
  const response=await fetch('/api/duet');if(!response.ok)throw new Error('No local duet');
  manifest=await response.json();
  if(manifest.available){
    player=new VoicePair({urls:{lead:'/duet-audio/lead.wav',reply:'/duet-audio/reply.wav'},duration:manifest.duration_s,onChange:update});
    byId('duet').hidden=false;byId('duet-song').textContent=manifest.title;
    byId('duet-arrangement').textContent=manifest.arrangement;byId('duet-attribution').textContent=manifest.attribution;
    byId('duet-tempo').textContent=`${manifest.tempo_bpm} BPM · ${Math.round(manifest.duration_s)} seconds · dry vocals`;
    byId('duet-seek').max=manifest.duration_s;
    for(const track of ['lead','reply']){
      const row=document.createElement('div');row.className='duet-track '+track;
      const label=document.createElement('span');label.textContent=track==='lead'?'HEART':'MICHAEL';row.append(label);
      const lane=document.createElement('div');lane.className='duet-lane';row.append(lane);
      for(const phrase of manifest.phrases.filter(p=>p.part===track)){
        const button=document.createElement('button');button.className='duet-phrase';button.dataset.phrase=phrase.id;
        button.style.left=phrase.start_s/manifest.duration_s*100+'%';button.style.width=(phrase.end_s-phrase.start_s)/manifest.duration_s*100+'%';
        button.title=phrase.text;button.setAttribute('aria-label',`Seek to ${track==='lead'?'Heart':'Michael'}: ${phrase.text}`);
        button.textContent=phrase.text;button.addEventListener('click',()=>seek(phrase.start_s));lane.append(button);
      }
      byId('duet-score').append(row);
    }
    for(const phrase of [...manifest.phrases].sort((a,b)=>a.start_s-b.start_s)){
      const p=document.createElement('p');p.textContent=`${clock(phrase.start_s)} · ${phrase.part==='lead'?'Heart':'Michael'} — ${phrase.text}`;byId('duet-lines').append(p);
    }
    update();
  }
}catch(error){console.info('Optional local duet is not prepared:',error.message);}
