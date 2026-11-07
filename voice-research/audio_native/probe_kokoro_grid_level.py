"""Offline level-only diagnostic after first-upsampler grid cancellation.

Rescales the sustained interior to the baseline's measured RMS, with guarded
cosine ramps. This uses complete rendered clips; it is not live gain control.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import validate_kokoro_control_generalization as common

ROOT=common.ROOT
PRIOR=ROOT/'evidence/kokoro-control-voicebox/upsampler-grid-v1'
OUT=ROOT/'evidence/kokoro-control-voicebox/grid-level-v1'


def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def setup():
    prior=json.loads((PRIOR/'report.json').read_text())
    config=json.loads((PRIOR/'protocol.json').read_text())
    if not prior['complete']:raise ValueError('Incomplete correction study')
    config.update(scope='Post-validation offline gain-only diagnostic after full grid-response cancellation',
        methods=['baseline','full','level'],newControlledRenders=36,baselineReplays=0,
        previousGridReportSha256=common.sha(PRIOR/'report.json'),
        levelRule='Constant interior gain is baseline RMS / corrected RMS on hold +200 ms to end -200 ms. Cosine ramps run from +50 to +200 ms and -200 to -50 ms; no gain outside the hold.',
        limitation='Full source and corrected clips are used to select one gain; not real-time adaptive level control, new training or naturalness acceptance.')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        if json.loads((OUT/'protocol.json').read_text())!=config:raise ValueError('Changed gain protocol')
    else:write('protocol.json',config)
    common.OUT=OUT
    return config,prior


def render(config,prior):
    report={'protocolSha256':common.sha(OUT/'protocol.json'),'sources':prior['sources'],'failures':[],
        'results':[r for r in prior['results'] if r['method'] in ('baseline','full')],
        'expectedEligibleRenders':108,'complete':False}
    if (OUT/'report.json').exists():report=common.read('report.json')
    if report['complete']:print('Already complete; retained.');return
    done={r['name'] for r in report['results']}
    for row in prior['results']:
        if row['method']!='full':continue
        name=row['name']+'-level'
        if name in done:continue
        baseline=next(r for r in prior['results'] if r['method']=='baseline' and
            (r['speaker'],r['word'],r['mode'])==(row['speaker'],row['word'],row['mode']))
        arrays=[]
        for item in (baseline,row):
            if common.sha(ROOT/item['file'])!=item['sha256']:raise ValueError('Changed gain inputs')
            rate,audio=wavfile.read(ROOT/item['file']);assert rate==24000
            arrays.append(np.asarray(audio,dtype=float))
        original,corrected=arrays;t=np.arange(len(corrected))/rate
        start=row['plan']['holdStartSeconds'];end=row['plan']['holdEndSeconds']
        interior=(t>=start+.2)&(t<=end-.2)
        source_rms=float(np.sqrt(np.mean(original[interior]**2)))
        corrected_rms=float(np.sqrt(np.mean(corrected[interior]**2)))
        if min(source_rms,corrected_rms)<=1e-8:raise ValueError('Cannot compensate silent interior')
        gain=source_rms/corrected_rms
        attack=np.clip((t-start-.05)/.15,0.,1.);release=np.clip((end-.05-t)/.15,0.,1.)
        gate=(.5-.5*np.cos(np.pi*attack))*(.5-.5*np.cos(np.pi*release))
        output=corrected*(1+(gain-1)*gate)
        if not np.isfinite(output).all():raise ValueError('Nonfinite level diagnostic')
        file=OUT/(name+'.wav');wavfile.write(file,rate,output.astype(np.float32))
        report['results'].append({**row,'name':name,'method':'level','file':str(file.relative_to(ROOT)),
            'sha256':common.sha(file),'levelCompensation':{'constantGain':gain,'gainDb':float(20*np.log10(gain)),
                'baselineCoreRms':source_rms,'correctedCoreRms':corrected_rms,'coreStartSeconds':start+.2,'coreEndSeconds':end-.2,
                'postLevelCoreRms':float(np.sqrt(np.mean(output[interior]**2))),
                'maximumOutsideGateDifference':float(max(abs(output[gate==0]-corrected[gate==0]))),
                'wholeClipInformationRequired':True},
            'rawPeak':float(max(abs(output))),'overFullScaleSamples':int(np.sum(abs(output)>=1)),'finite':True,
            'wholeDecoderComputeSeconds':None})
        write('report.json',report);print(name,'gain dB',round(20*np.log10(gain),2),'peak',round(max(abs(output)),3),flush=True)
    report['complete']=len(report['results'])==108;write('report.json',report)


if __name__=='__main__':
    config,prior=setup()
    if sys.argv[1]=='render':render(config,prior)
    elif sys.argv[1]=='evaluate':
        if not (OUT/'evaluation.json').exists():
            previous=json.loads((PRIOR/'evaluation.json').read_text())
            if not previous['complete']:raise ValueError('Incomplete previous evaluation')
            reused=[r for r in previous['results'] if r['method'] in ('baseline','full')]
            for row in reused+previous['sources']:
                if common.sha(ROOT/row['file'])!=row['sha256']:raise ValueError('Changed reused audio')
            write('evaluation.json',{'protocolSha256':common.sha(OUT/'protocol.json'),'sources':previous['sources'],
                'results':reused,'reusedAsrEvaluations':90,'complete':False})
        common.evaluate(config)
    else:raise ValueError('Unknown stage')
