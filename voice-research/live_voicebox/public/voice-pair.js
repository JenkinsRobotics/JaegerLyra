// Two independent sources on one audio clock. No combined buffer is fetched
// or constructed; each voice keeps its own buffer, source and gain node.
export class VoicePair {
  constructor({urls,duration,contextFactory=()=>new AudioContext(),fetcher=(...args)=>fetch(...args),onChange=()=>{}}){
    this.urls=urls;this.duration=duration;this.contextFactory=contextFactory;this.fetcher=fetcher;this.onChange=onChange;
    this.context=null;this.buffers=null;this.loadingPromise=null;this.sources=[];this.gains={};this.monitor=null;
    this.levels={lead:1,reply:1};this.enabled={lead:true,reply:true};this.muted=false;
    this.offset=0;this.startedAt=0;this.playing=false;this.loading=false;this.ended=false;this.epoch=0;
  }
  get position(){return this.playing?Math.min(this.duration,this.offset+Math.max(0,this.context.currentTime-this.startedAt)):this.offset;}
  snapshot(){return {position:this.position,duration:this.duration,playing:this.playing,loading:this.loading,ended:this.ended,
    activeSources:this.sources.length,enabled:{...this.enabled},levels:{...this.levels},muted:this.muted};}
  changed(){this.onChange(this.snapshot());}
  async prepare(){
    if(this.buffers)return;
    if(!this.loadingPromise)this.loadingPromise=(async()=>{
      const entries=await Promise.all(Object.entries(this.urls).map(async([key,url])=>{
        const response=await this.fetcher(url);if(!response.ok)throw new Error('Could not load the '+key+' voice.');
        const buffer=await this.context.decodeAudioData(await response.arrayBuffer());
        if(Math.abs(buffer.duration-this.duration)>1/buffer.sampleRate)throw new Error('The vocal tracks have different timing.');
        return [key,buffer];
      }));
      this.buffers=Object.fromEntries(entries);
    })().catch(error=>{this.loadingPromise=null;throw error;});
    await this.loadingPromise;
  }
  async play(){
    if(this.playing||this.loading)return;
    const epoch=++this.epoch;this.loading=true;this.ended=false;
    if(this.offset>=this.duration)this.offset=0;
    this.changed();
    try{
      if(!this.context){
        this.context=this.contextFactory();this.monitor=this.context.createGain();
        this.monitor.gain.value=this.muted?0:1;this.monitor.connect(this.context.destination);
        for(const key of Object.keys(this.urls)){
          const gain=this.context.createGain();gain.gain.value=this.enabled[key]?this.levels[key]:0;
          gain.connect(this.monitor);this.gains[key]=gain;
        }
      }
      await this.context.resume();await this.prepare();
      if(epoch!==this.epoch)return;
      const when=this.context.currentTime+.04;this.startedAt=when;
      let remaining=Object.keys(this.urls).length;
      for(const key of Object.keys(this.urls)){
        const source=this.context.createBufferSource();source.buffer=this.buffers[key];source.connect(this.gains[key]);
        source.onended=()=>{
          source.disconnect();
          if(epoch!==this.epoch)return;
          if(--remaining===0){this.offset=this.duration;this.playing=false;this.ended=true;this.sources=[];this.changed();}
        };
        this.sources.push(source);source.start(when,this.offset);
      }
      this.playing=true;this.loading=false;this.changed();
    }catch(error){if(epoch===this.epoch){this.pause();throw error;}}
  }
  pause(){
    this.offset=this.position;this.epoch++;this.playing=false;this.loading=false;
    for(const source of this.sources){source.onended=null;source.stop();source.disconnect();}
    this.sources=[];this.changed();
  }
  stop(){this.pause();this.offset=0;this.ended=false;this.changed();}
  async seek(seconds){
    if(!Number.isFinite(seconds))throw new Error('Invalid playback position.');
    const resume=this.playing||this.loading;this.pause();this.offset=Math.max(0,Math.min(this.duration,seconds));this.ended=false;this.changed();
    if(resume&&this.offset<this.duration)await this.play();
  }
  setVoice(key,{enabled=this.enabled[key],level=this.levels[key]}={}){
    if(!(key in this.levels)||typeof enabled!=='boolean'||!Number.isFinite(level)||level<0||level>1)throw new Error('Invalid voice control.');
    this.enabled[key]=enabled;this.levels[key]=level;
    if(this.gains[key])this.gains[key].gain.setTargetAtTime(enabled?level:0,this.context.currentTime,.008);
    this.changed();
  }
  setMuted(muted){this.muted=!!muted;if(this.monitor)this.monitor.gain.setTargetAtTime(this.muted?0:1,this.context.currentTime,.008);this.changed();}
  async close(){this.stop();if(this.context)await this.context.close();}
}
