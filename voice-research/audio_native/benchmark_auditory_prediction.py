"""Compare persistent, linear and local-PC forecasts at useful audio timescales.

Features are extracted causally by benchmark_voicebox.py. Normalization and
initial weights use only synthetic reference clips; evaluation uses held-out
utterances. Adaptive evaluation updates only after each forecast target arrives.
This is teacher-forced feature prediction, not free-running audio or semantics.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import copy
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from lab import Config, LinearPredictor, LocalPredictor
from auditory_prediction import AuditoryForecast, LAGS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'evidence/streaming-voicebox'


def run(model, data, horizon, adaptive):
    stream = AuditoryForecast(model, horizon, adaptive)
    rows, times = [], []
    for feature in data:
        started = time.perf_counter()
        _, completed = stream.push(feature)
        times.append((time.perf_counter()-started)*1000)
        if completed is not None:
            rows.append(completed)
    return rows, times


def evaluate(rows, mean, scale, raw_f0):
    prediction = np.array([row['prediction'] for row in rows])*scale+mean
    truth = np.array([row['target'] for row in rows])*scale+mean
    indices = np.array([row['targetIndex'] for row in rows])
    active = truth[:,16]>-55
    voiced = raw_f0[indices]>0
    # All scores use predictions committed BEFORE the evaluated target arrived.
    normalized = (prediction-truth)/scale
    return {'activeFrames':int(sum(active)),
        'activeNormalizedMse':float(np.mean(normalized[active]**2)),
        'envelopeMeanAbsoluteDb':float(np.mean(np.abs(prediction[active,:16]-truth[active,:16]))),
        'energyMeanAbsoluteDb':float(np.mean(np.abs(prediction[active,16]-truth[active,16]))),
        'voicedPitchMedianAbsoluteCents':float(np.median(np.abs(prediction[voiced,17]-truth[voiced,17]))*1200),
        'periodicityMeanAbsoluteError':float(np.mean(np.abs(prediction[active,18]-truth[active,18]))),
        'finite':bool(np.isfinite(prediction).all())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validation',action='store_true')
    args=parser.parse_args()
    speakers = ['af_heart','am_michael']
    data = {(speaker, role):dict(np.load(OUT/f'{speaker}-{role}-features.npz'))
            for speaker in speakers for role in ('reference','heldout')}
    training = np.concatenate([data[(s,'reference')]['features'] for s in speakers])
    mean = np.mean(training,axis=0)
    scale = np.maximum(np.std(training,axis=0), np.r_[np.ones(16),3.,.25,.1])
    normalized = {key:(value['features']-mean)/scale for key,value in data.items()}
    evaluation = {s:(normalized[(s,'heldout')],data[(s,'heldout')]['f0']) for s in speakers}
    if args.validation:
        manifest=json.loads((OUT/'validation/manifest.json').read_text())
        evaluation={}
        for clip in manifest['clips']:
            item=np.load(OUT/'validation'/clip['featureFile'])
            evaluation[clip['name']]=((item['features']-mean)/scale,item['f0'])
    cfg = Config(history=len(LAGS)*19, block=19, hidden=64, settling_steps=12, learning_rate=.04,input_gain=1.,seed=17)
    report = {'scope':'Causal feature forecasts only; no semantic understanding or free-running audio. Feature errors are not perceptual voice-quality ratings.',
        'features':['16 relative envelope dB values','energy dB','log2 F0 / 160Hz (0 when unvoiced)','periodicity'],
        'historyLagsMs':[10*lag for lag in LAGS], 'horizonsMs':[20,50,100],
        'feedback':'Each prediction is stored before its target arrives; adaptation waits for the target timestamp. True intervening features are observed normally.',
        'normalization':'Mean and scale fitted only on Heart and Michael reference clips; held-out samples excluded',
        'training':'One chronological pass through the two reference clips for each model/horizon/seed; no held-out tuning or training',
        'config':asdict(cfg),'seeds':[17,29,41], 'results':[]}
    report['evaluationSet']='Six new clips, including previously unseen Bella' if args.validation else 'Original held-out sentence in Heart and Michael'
    for horizon in (2,5,10):
        for name in ('persist','linear-nlms','local-pc'):
            seeds = (17,29,41) if name=='local-pc' else (None,)
            for seed in seeds:
                configuration = Config(**{**asdict(cfg),'seed':seed or 17})
                model = None if name=='persist' else LinearPredictor(configuration) if name=='linear-nlms' else LocalPredictor(configuration)
                if model is not None:
                    for speaker in speakers:
                        run(model, normalized[(speaker,'reference')], horizon, True)
                for speaker,(features, f0) in evaluation.items():
                    for adaptive in ((False,True) if model is not None else (False,)):
                        rows, times = run(copy.deepcopy(model),features,horizon,adaptive)
                        metrics = evaluate(rows,mean,scale,f0)
                        metrics.update(model=name,seed=seed,speaker=speaker,horizonMs=horizon*10,adaptive=adaptive,
                                       processingMsP95=float(np.percentile(times,95)))
                        report['results'].append(metrics)
                        print(json.dumps(metrics),flush=True)
    (OUT/('validation-prediction-report.json' if args.validation else 'prediction-report.json')).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
