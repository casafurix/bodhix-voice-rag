export class PcmRecorder {
  private ctx?: AudioContext; private stream?: MediaStream; private source?: MediaStreamAudioSourceNode; private processor?: ScriptProcessorNode; private chunks: Float32Array[]=[]; private length=0;
  async start(){
    this.stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
    this.ctx=new AudioContext({sampleRate:16000}); await this.ctx.resume();
    this.source=this.ctx.createMediaStreamSource(this.stream); this.processor=this.ctx.createScriptProcessor(4096,1,1);
    this.processor.onaudioprocess=e=>{const data=new Float32Array(e.inputBuffer.getChannelData(0));this.chunks.push(data);this.length+=data.length};
    this.source.connect(this.processor); this.processor.connect(this.ctx.destination);
  }
  async stop():Promise<Blob>{
    this.processor?.disconnect();this.source?.disconnect();this.stream?.getTracks().forEach(t=>t.stop());await this.ctx?.close();
    const pcm=new Float32Array(this.length);let off=0;for(const c of this.chunks){pcm.set(c,off);off+=c.length}
    return encodeWav(pcm,16000);
  }
}
function encodeWav(samples:Float32Array,sampleRate:number){const buffer=new ArrayBuffer(44+samples.length*2),view=new DataView(buffer);const write=(o:number,s:string)=>{for(let i=0;i<s.length;i++)view.setUint8(o+i,s.charCodeAt(i))};write(0,'RIFF');view.setUint32(4,36+samples.length*2,true);write(8,'WAVE');write(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,sampleRate,true);view.setUint32(28,sampleRate*2,true);view.setUint16(32,2,true);view.setUint16(34,16,true);write(36,'data');view.setUint32(40,samples.length*2,true);let o=44;for(const s of samples){const v=Math.max(-1,Math.min(1,s));view.setInt16(o,v<0?v*0x8000:v*0x7fff,true);o+=2}return new Blob([buffer],{type:'audio/wav'})}
