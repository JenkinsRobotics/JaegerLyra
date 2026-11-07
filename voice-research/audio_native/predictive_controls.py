"""Continuous control prediction that can run without target audio for a gap.

Classical reflection coefficients provide a stable *fixed* all-pole denominator;
that alone does not prove arbitrary time-varying synthesis stability. This is
short acoustic continuation, not language understanding or independent TTS.
"""
from collections import deque
import numpy as np
from excitation import VocalControls
from auditory_prediction import LAGS
from vocal_energy import source_filter_power
from lpc_coordinates import to_reflection, from_reflection


class ControlSpace:
    """84 continuous controls: reflection, pitch/voice/energy, PCA source.

Encoding is stateless so an unseen frame cannot affect a later observation via
an encoder's hidden pitch history. Unvoiced pitch uses a fixed 160 Hz convention.
"""
    def __init__(self, source, order=32, output_energy=False):
        if output_energy not in (False,True,'window'):
            raise ValueError('Unknown energy control mode')
        self.source = source
        self.order = order
        self.size = order + 4 + len(source.basis)
        self.output_energy = output_energy

    def encode(self, frame):
        control = VocalControls.from_frame(frame)
        energy=(frame.window_rms if self.output_energy=='window' else
                frame.rms if self.output_energy else control.excitation_rms)
        return np.r_[np.arctanh(to_reflection(control.coefficients)),
                     np.log2((frame.f0 or 160.)/160.), frame.periodicity,
                     float(frame.f0 > 0), np.log(energy + 1e-6),
                     self.source.encode(frame)]

    def decode(self, vector, start, samples):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.size,) or not np.isfinite(vector).all():
            raise ValueError('Invalid continuous vocal vector')
        order = self.order
        reflection = np.tanh(np.clip(vector[:order], -5., 5.))
        coefficients = from_reflection(reflection)
        f0 = float(np.clip(160.*2**np.clip(vector[order], -2., 2.), 50., 500.))
        periodicity = float(np.clip(vector[order+1], 0., 1.))
        voice = float(np.clip(vector[order+2], 0., 1.))
        gain = float(np.exp(np.clip(vector[order+3], np.log(1e-6), 0.))-1e-6)
        harmonic_mix = voice*float(np.clip(periodicity, .65, .98))
        harmonics=self.source.decode(vector[order+4:])
        if self.output_energy:
            power=source_filter_power(coefficients,reflection,f0,harmonics,harmonic_mix)
            gain/=np.sqrt(power['mixedPowerGain'])
        control = VocalControls(start, samples, coefficients, f0, periodicity, gain)
        return control, harmonics, harmonic_mix


def visible_outside_gap(frames, gap_start, gap_end, config):
    """Exclude every aligned-analysis frame whose input window touches the gap.

    This includes the 15 ms lookahead and 15 ms past side; simply dropping frames
    whose output timestamps lie in the gap would leak withheld waveform samples.
    """
    if gap_start < 0 or gap_end <= gap_start:
        raise ValueError('Invalid missing-audio interval')
    lookahead = (config.history-config.hop)//2
    past = config.history-config.hop-lookahead
    return np.array([f.available_at <= gap_start or f.start-past >= gap_end for f in frames])


class FreeRunningControls:
    """Predict one frame before observing it; feed predictions back during loss.

    Missing data is None, not a numeric placeholder. No observe() call is made
    during a gap, and learning resumes only after the full context is observed
    again. Predicted control ranges are bounded by reference-only training stats.
    """
    def __init__(self, model, center, scale, lower, upper, adaptive=True, lags=LAGS):
        self.model = model
        self.center = np.asarray(center, dtype=np.float64).copy()
        self.scale = np.asarray(scale, dtype=np.float64).copy()
        self.lower = np.asarray(lower, dtype=np.float64).copy()
        self.upper = np.asarray(upper, dtype=np.float64).copy()
        if (self.center.ndim != 1 or any(x.shape != self.center.shape for x in (self.scale,self.lower,self.upper))
                or not all(np.isfinite(x).all() for x in (self.center,self.scale,self.lower,self.upper))
                or np.any(self.scale<=0) or np.any(self.lower>self.upper)):
            raise ValueError('Invalid reference statistics')
        if not lags or list(lags) != sorted(set(lags)) or min(lags)!=0:
            raise ValueError('Invalid past lags')
        self.lags = lags
        self.history = deque(maxlen=max(lags)+1)
        self.observed = deque(maxlen=max(lags)+1)
        self.adaptive = adaptive
        self.learned_frames = 0
        self.generated_frames = 0
        self.clamped_values = 0

    def step(self, observation):
        if observation is not None:
            observation = np.asarray(observation, dtype=np.float64)
            if observation.shape != self.center.shape or not np.isfinite(observation).all():
                raise ValueError('Invalid observed controls')
        elif not self.history:
            raise ValueError('Cannot continue before any observed controls')
        complete = len(self.history)>max(self.lags)
        base = self.history[-1].copy() if self.history else np.zeros_like(self.center)
        context = np.concatenate([self.history[-lag-1] for lag in self.lags]) if complete else None
        # Prediction is committed before the current observation enters state.
        prediction = base.copy()
        if self.model is not None and complete:
            prediction += self.model.predict(context)
        if observation is None:
            if self.model is not None:
                bounded = np.clip(prediction, self.lower, self.upper)
                self.clamped_values += int(np.count_nonzero(bounded != prediction))
                prediction = bounded
            value = prediction
            self.generated_frames += 1
        else:
            value = (observation-self.center)/self.scale
            if self.model is not None and complete and all(self.observed) and self.adaptive:
                self.model.observe(context, value-base, learn=True)
                self.learned_frames += 1
        self.history.append(value.copy())
        self.observed.append(observation is not None)
        return self.center + self.scale*value, self.center + self.scale*prediction


class AnchoredControls(FreeRunningControls):
    """Direct multi-horizon forecasts from the last fully observed context.

    Does not recursively treat its generated controls as observed input. Each
    horizon has a local predictor; forecasts are committed before targets exist.
    Missing targets are discarded. Interpolation is between predicted horizons,
    never between known audio on opposite sides of a gap.
    """
    def __init__(self, models, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        if not models or min(models)!=1 or any(not isinstance(h,int) or h<1 for h in models):
            raise ValueError('Direct horizons must include one frame')
        self.models = dict(sorted(models.items()))
        self.horizons = np.array(list(self.models))
        self.pending = {}
        self.index = -1
        self.anchor = None

    def step(self, observation):
        if observation is not None:
            observation = np.asarray(observation, dtype=np.float64)
            if observation.shape != self.center.shape or not np.isfinite(observation).all():
                raise ValueError('Invalid observed controls')
        elif not self.history:
            raise ValueError('Cannot continue before any observation')
        self.index += 1
        due = self.pending.pop(self.index, [])
        base = self.history[-1].copy() if self.history else np.zeros_like(self.center)
        prediction = next((task['prediction'].copy() for task in due if task['horizon']==1),base.copy())
        if observation is None:
            if self.anchor is None:
                complete = len(self.history)>max(self.lags) and all(self.observed)
                if complete:
                    context = np.concatenate([self.history[-lag-1] for lag in self.lags])
                    curve = np.array([base + model.predict(context) for model in self.models.values()])
                else:
                    curve = np.array([base.copy() for _ in self.models])
                self.anchor = (self.index,curve)
            at,curve = self.anchor
            horizon = self.index-at+1
            right = int(np.searchsorted(self.horizons,horizon))
            if right == 0:prediction = curve[0].copy()
            elif right >= len(curve):prediction = curve[-1].copy()
            else:
                left=right-1
                mix=(horizon-self.horizons[left])/(self.horizons[right]-self.horizons[left])
                prediction=(1-mix)*curve[left]+mix*curve[right]
            bounded=np.clip(prediction,self.lower,self.upper)
            self.clamped_values+=int(np.count_nonzero(bounded!=prediction))
            value=bounded
            self.generated_frames+=1
        else:
            value=(observation-self.center)/self.scale
            # Every queued context was fully observed when emitted. A target
            # that lands after a gap may update that older valid forecast.
            if self.adaptive:
                for task in due:
                    self.models[task['horizon']].observe(task['context'],value-task['base'],learn=True)
                    self.learned_frames+=1
            self.anchor=None
        self.history.append(value.copy());self.observed.append(observation is not None)
        if observation is not None and len(self.history)>max(self.lags) and all(self.observed):
            context=np.concatenate([self.history[-lag-1] for lag in self.lags])
            for horizon,model in self.models.items():
                task={'horizon':horizon,'context':context.copy(),'base':value.copy(),
                      'prediction':value+model.predict(context)}
                self.pending.setdefault(self.index+horizon,[]).append(task)
        return self.center+self.scale*value,self.center+self.scale*prediction
