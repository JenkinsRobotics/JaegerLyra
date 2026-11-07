"""Compare cold loaders using identical checkpoints, requests and sampling seeds.

Run each mode in its own process. Does not change the app's loader or weights.
"""
import argparse,copy,hashlib,json,os,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
os.environ['NLTK_DATA']=str(ROOT/'.cache/nltk')
sys.path.insert(0,str(ROOT))
started=time.perf_counter()
from scripts.try_soulx import make_prompt,make_sung_prompt,SOURCE
import numpy as np
import torch
from omegaconf import OmegaConf
from cli.inference import build_model
from soulxsinger.models.soulxsinger import SoulXSinger
from soulxsinger.utils.data_processor import DataProcessor
from accelerate import init_empty_weights
from singing_model import load_singer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['stock','assign','meta']);args=parser.parse_args()
    folder=ROOT/'evidence/singer-loading';folder.mkdir(exist_ok=True)
    report={'mode':args.mode,'importSeconds':round(time.perf_counter()-started,3),'samples':[]}
    torch.set_num_threads(4);torch.manual_seed(42)
    device='mps';config=OmegaConf.load(ROOT/'models/soulx-singer/config.yaml')
    checkpoint_path=ROOT/'models/soulx-singer/model.pt'
    before=time.perf_counter()
    try:
        if args.mode=='stock':
            model=build_model(str(checkpoint_path),config,device=device)
        else:
            if args.mode=='meta':model=load_singer(checkpoint_path,config,device)
            else:
                model=SoulXSinger(config)
                report['constructSeconds']=round(time.perf_counter()-before,3)
                checkpoint=torch.load(checkpoint_path,weights_only=True,mmap=True,map_location='cpu')
                model.load_state_dict(checkpoint['state_dict'],strict=True,assign=True)
                model.eval().to(device)
        torch.mps.synchronize()
        report['loadSeconds']=round(time.perf_counter()-before,3)
        checkpoint=torch.load(checkpoint_path,weights_only=True,mmap=True,map_location='cpu')
        mismatches=[name for name,value in model.state_dict().items() if not torch.equal(value.cpu(),checkpoint['state_dict'][name])]
        report['checkpointMismatchNames']=mismatches
        report['gpuAllocatedAfterLoadBytes']=torch.mps.current_allocated_memory()
        report['bufferHashes']={name:hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for name,value in model.named_buffers()}
        print('loaded',json.dumps({k:v for k,v in report.items() if k not in ('bufferHashes','samples')}),flush=True)
        processor=DataProcessor(config.audio.hop_size,config.audio.sample_rate,str(SOURCE/'soulxsinger/utils/phoneme/phone_set.json'),device=device)
        for name,sung in [('heart-spoken',False),('heart-sung',True)]:
            request=json.loads((ROOT/'evidence/integrated-singer'/f'{name}.json').read_text())
            fixture=ROOT/'.cache/singing-references'/('v1-af_heart-'+('sung' if sung else 'speech'))
            prompt,wav=make_sung_prompt(fixture) if sung else make_prompt(fixture=fixture)
            data={'prompt':processor.process(copy.deepcopy(prompt),str(wav)),
                  'target':processor.process(copy.deepcopy(request['score']),None)}
            torch.manual_seed(42);before=time.perf_counter()
            with torch.inference_mode():
                audio=model.infer(data,auto_shift=False,pitch_shift=0,n_steps=32,cfg=3.,control='score').squeeze().cpu().numpy()
            elapsed=time.perf_counter()-before
            np.save(folder/(args.mode+'-'+name+'.npy'),audio)
            row={'name':name,'seconds':round(elapsed,3),'samples':len(audio),'peak':float(np.max(np.abs(audio))),'finite':bool(np.isfinite(audio).all())}
            baseline=folder/('stock-'+name+'.npy')
            if args.mode!='stock' and baseline.exists():
                ref=np.load(baseline);error=audio-ref
                row['maxAbsoluteDifference']=float(np.max(np.abs(error)))
                row['differenceRms']=float(np.sqrt(np.mean(error**2)))
                row['sampleIdentical']=bool(np.array_equal(ref,audio))
                row['signalToDifferenceDb']=None if not np.any(error) else float(10*np.log10(np.sum(ref**2)/np.sum(error**2)))
            report['samples'].append(row);print(json.dumps(row),flush=True)
        report['gpuAllocatedAfterRenderBytes']=torch.mps.current_allocated_memory()
        report['maxRssBytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception as error:
        report['error']=str(error)
        print('ERROR',repr(error),flush=True)
        raise
    finally:
        report['wallSeconds']=round(time.perf_counter()-started,3)
        (folder/(args.mode+'.json')).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
