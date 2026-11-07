"""Reproduce the old LPC defect and compare covariance residuals on all inputs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import json,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.linalg import toeplitz
from streaming_voicebox import lpc,VoiceboxConfig,LPC_VERSION,StreamingSynthesis
from aligned_analysis import AlignedAnalysis
from legacy_lpc_v1 import lpc as old_lpc
from lpc_coordinates import to_reflection

ROOT=Path(__file__).resolve().parents[2]


def main():
    report={'scope':'Numerical LPC correction, not naturalness acceptance',
        'analysisVersion':LPC_VERSION,
        'defect':'Clamping intermediate reflection coefficients changes the covariance solution without correcting subsequent prediction-error recursion',
        'change':'Solve the diagonally loaded autocorrelation system without clipping; then apply the existing bandwidth expansion',
        'legacyCode':'voice-research/audio_native/legacy_lpc_v1.py','results':[]}
    paths=list((ROOT/'evidence/voice-fixtures').glob('*.wav'))
    paths+=list((ROOT/'evidence/streaming-voicebox/validation').glob('*.wav'))
    for folder in ('fresh-inputs','human-inputs'):paths+=list((ROOT/f'evidence/source-detail/{folder}').glob('*.wav'))
    cfg=VoiceboxConfig(order=32);chirp=cfg.bandwidth**np.arange(33)
    for path in sorted(paths):
        audio,rate=sf.read(path);assert rate==24000
        padded=np.pad(audio,(360,600));values={'old':[],'new':[]};failures={'old':0,'new':0};max_coefficient={'old':0.,'new':0.}
        changed=0;frames=0;example=None
        for at in range(0,len(audio),240):
            window=padded[at:at+960]
            signal=(window-window.mean())*np.hanning(960)
            r=np.array([signal[k:]@signal[:len(signal)-k] for k in range(33)])
            r[0]+=1e-6+r[0]*1e-4;covariance=toeplitz(r[:32])
            before=old_lpc(window,cfg);after=lpc(window,cfg)
            changed+=int(not np.array_equal(before,after));frames+=1
            for label,coefficients in (('old',before),('new',after)):
                raw=coefficients/chirp
                values[label].append(float(np.linalg.norm(covariance@raw[1:]+r[1:])/r[0]))
                max_coefficient[label]=max(max_coefficient[label],float(abs(coefficients).max()))
                try:to_reflection(coefficients)
                except ValueError:
                    failures[label]+=1
                    if example is None:example={'sample':at,'oldCoefficients':before.tolist(),'newCoefficients':after.tolist()}
        analysis=AlignedAnalysis(cfg);synthesis=StreamingSynthesis(cfg)
        packets=analysis.push(audio)+analysis.finish()
        reconstructed=np.concatenate([synthesis.push(frame) for frame in packets])
        difference=float(abs(reconstructed-audio).max())
        assert difference<1e-8
        row={'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'frames':frames,'changedFrames':changed,'reflectionConversionFailures':failures,
             'maximumCoefficientMagnitude':max_coefficient,
             'normalizedNormalEquationResidualP95':{k:float(np.percentile(v,95)) for k,v in values.items()},
             'firstFailureExample':example,
             'fullResidualRoundtripMaximumDifference':difference,
             'fullResidualPcm16Identical':bool(np.array_equal(np.rint(reconstructed*32768),np.rint(audio*32768)))}
        report['results'].append(row)
        output=ROOT/'evidence/source-detail/lpc-v2/solver-audit.json';output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('sha256','firstFailureExample')}),flush=True)


if __name__=='__main__':main()
