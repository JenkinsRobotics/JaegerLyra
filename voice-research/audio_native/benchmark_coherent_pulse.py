"""Paired phase-canonical ablation; banks and all prior schedules remain frozen."""
from benchmark_pulse_basis import main,ROOT
from coherent_pulse_basis import CoherentPulseBasisStream
from functools import partial


if __name__=='__main__':
    main(out=ROOT/'evidence/pulse-basis/coherent-v2',stream_factory=CoherentPulseBasisStream,
        extra_report={'scope':'Periodized, phase-canonical pulse generation from the frozen captured waveform bank',
            'parentReport':'evidence/pulse-basis/v1/report.json',
            'ablation':'Central reference-length cycle extraction, Nyquist restriction and fundamental phase alignment; no bank refitting',
            'qualityReplacementAccepted':False})
    main(out=ROOT/'evidence/pulse-basis/periodic-only-v2',
        stream_factory=partial(CoherentPulseBasisStream,align=False),
        variants={'correlated-seed-0':(1.,True,0)},
        extra_report={'scope':'Periodized pulses without fundamental-phase correction; paired seed-0 ablation',
            'parentReport':'evidence/pulse-basis/v1/report.json',
            'ablation':'Same central cycle extraction and Nyquist restriction as coherent-v2, but phase is left unchanged',
            'qualityReplacementAccepted':False})
