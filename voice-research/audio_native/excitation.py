"""Generated harmonic/noise excitation with a small reference-fitted profile.

Classical source/filter research, NOT a trained TTS model or accepted cloning.
The decoder only receives continuous filter/pitch/energy controls; it cannot read
the original PCM residual. The filter controls still come from the input speech,
so this is analysis/resynthesis, not autonomous content or language generation.
No global backpropagation or discrete codebook. Profiles use running statistics.
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter, lfiltic
from streaming_voicebox import VoiceboxConfig, resonance_coefficients
from lpc_coordinates import to_reflection


@dataclass(frozen=True)
class VocalControls:
    start: int
    samples: int
    coefficients: np.ndarray
    f0: float
    periodicity: float
    excitation_rms: float

    @classmethod
    def from_frame(cls, frame):
        return cls(frame.start, len(frame.residual), frame.coefficients.copy(),
                   frame.f0, frame.periodicity,
                   float(np.sqrt(np.mean(frame.residual**2))))


class ReferenceExcitation:
    """Running harmonic-power and phase statistics from one reference.

    Phase is aligned to each observed frame's fundamental. The residual/filter
    decomposition is approximate; this is not a measured glottal waveform.
    Stored profile size is independent of reference duration. No audio segments
    or discrete units are retained in the resulting profile.
    """
    def __init__(self, harmonics=32, rate=24000):
        self.harmonics = harmonics
        self.rate = rate
        self.history = np.zeros(960)
        self.power = np.zeros(harmonics)
        self.phase = np.zeros(harmonics, dtype=np.complex128)
        self.count = 0

    def observe(self, frame):
        self.history = np.r_[self.history, frame.residual][-len(self.history):]
        if frame.f0 <= 0 or frame.periodicity < .8:
            return
        k = np.arange(1,self.harmonics+1)
        valid = k*frame.f0 < self.rate*.47
        window = np.hanning(len(self.history))
        times = np.arange(len(self.history))/self.rate
        coefficients = 2*(np.exp(-2j*np.pi*k[:,None]*frame.f0*times)@(window*self.history))/sum(window)
        coefficients[~valid] = 0
        energy = np.sqrt(.5*np.sum(np.abs(coefficients)**2))
        if energy < 1e-5 or abs(coefficients[0])/energy < .05:
            return
        aligned = coefficients*np.exp(-1j*k*np.angle(coefficients[0]))/energy
        self.count += 1
        step = 1/self.count
        self.power += step*(np.abs(aligned)**2-self.power)
        self.phase += step*(aligned/(np.abs(aligned)+1e-12)-self.phase)

    def profile(self):
        if self.count < 8:
            raise ValueError('Reference needs more clear voiced frames')
        result = np.sqrt(self.power)*np.exp(1j*np.angle(self.phase))
        return {'real':result.real.tolist(),'imag':result.imag.tolist(),
                'voicedFrames':self.count,'harmonics':self.harmonics,
                'method':'Running aligned harmonic power/phase statistics; no neural training'}


def pulse_profile(harmonics=32):
    # A band-limited pulse-like baseline with gently declining harmonics.
    values = 1/np.arange(1,harmonics+1)**.35
    return {'real':values.tolist(),'imag':np.zeros(harmonics).tolist(),
            'voicedFrames':0,'harmonics':harmonics,'method':'Fixed pulse-like harmonic baseline'}


class GeneratedVoicebox:
    """Stateful PCM synthesis from compact continuous controls, no input PCM.

    Continuous oscillator phase, interpolated F0/voicing/energy, Nyquist-tapered
    harmonics and independent noise. Filter state uses previously generated
    samples. A profile changes source shape; it does not guarantee identity.
    """
    def __init__(self, profile=None, config=VoiceboxConfig(), seed=79, smooth_source=True,
                 filter_mode='direct', smooth_filter=False, shape_noise=False):
        if filter_mode not in ('direct','normalized-lattice'):
            raise ValueError('Unknown synthesis filter mode')
        self.filter_mode = filter_mode
        if smooth_filter and filter_mode!='normalized-lattice':
            raise ValueError('Smooth resonance requires the normalized lattice')
        self.smooth_filter=smooth_filter
        self.last_reflection=None
        self.noise_shaper=None
        if shape_noise:
            from noise_shape import ShapedNoise
            self.noise_shaper=ShapedNoise()
        if filter_mode == 'normalized-lattice':
            # Keep the existing default path independent of the optional JIT.
            from lattice_filter import normalized_lattice
            self.lattice = normalized_lattice
        self.config = config
        profile = profile or pulse_profile()
        self.harmonics = np.asarray(profile['real'])+1j*np.asarray(profile['imag'])
        if self.harmonics.ndim!=1 or not len(self.harmonics) or not np.isfinite(self.harmonics).all():
            raise ValueError('Invalid excitation profile')
        self.last_harmonics=self.harmonics.copy()
        self.smooth_source=smooth_source
        self.phase = 0.
        self.last_f0 = 160.
        self.last_voicing = 0.
        self.last_gain = 0.
        self.history = np.zeros(config.order)
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def push(self, control, pitch_ratio=1., resonance=1., harmonics=None, voicing=None,
             noise_bands=None):
        if control.start != self.position or not 0 < control.samples <= self.config.hop:
            raise ValueError('Invalid or out-of-order controls')
        if (not np.isfinite(pitch_ratio) or not .5 <= pitch_ratio <= 2.
                or not np.isfinite(control.f0) or control.f0<0
                or not np.isfinite(control.excitation_rms) or control.excitation_rms<0
                or not np.isfinite(control.periodicity)):
            raise ValueError('Invalid vocal control')
        if len(control.coefficients)!=self.config.order+1 or not np.isfinite(control.coefficients).all():
            raise ValueError('Invalid filter coefficients')
        if noise_bands is not None and self.noise_shaper is None:
            raise ValueError('Noise controls require shape_noise=True')
        n = control.samples
        target_f0 = np.clip(control.f0*pitch_ratio,50.,1000.) if control.f0 else self.last_f0
        if voicing is not None and (not np.isfinite(voicing) or not 0<=voicing<=1):
            raise ValueError('Voicing power fraction must be 0–1')
        target_voicing = (float(voicing) if voicing is not None else
                          np.clip(control.periodicity, .65, .98) if control.f0 else 0.)
        # Use a sample ramp, retaining the exact last emitted endpoint.
        blend = (np.arange(n)+1)/n
        f0 = self.last_f0+(target_f0-self.last_f0)*blend
        voicing = self.last_voicing+(target_voicing-self.last_voicing)*blend
        coefficients = resonance_coefficients(control.coefficients,resonance)
        target_gain = control.excitation_rms
        if self.filter_mode == 'normalized-lattice':
            reflection = to_reflection(coefficients)
            # Interpolate normalized drive, not unnormalized drive divided by
            # a new filter scale halfway through a frame. The latter pumps
            # source energy when resonance changes.
            target_gain /= np.exp(.5*np.sum(np.log1p(-reflection*reflection)))
        gain = self.last_gain+(target_gain-self.last_gain)*blend
        phase = self.phase+2*np.pi*np.cumsum(f0)/self.config.sample_rate
        harmonics=self.harmonics if harmonics is None else np.asarray(harmonics)
        if harmonics.shape!=self.harmonics.shape or not np.isfinite(harmonics).all():
            raise ValueError('Invalid continuous excitation descriptor')
        k = np.arange(1,len(harmonics)+1)[:,None]
        taper = np.clip((self.config.sample_rate*.49-k*f0[None,:])/(self.config.sample_rate*.04),0,1)
        values=(self.last_harmonics[:,None]+(harmonics-self.last_harmonics)[:,None]*blend
                if self.smooth_source else harmonics[:,None])*taper
        periodic = np.sum(np.real(values*np.exp(1j*k*phase[None,:])),axis=0)
        # Interpolate endpoint powers, not instantaneous interpolated amplitude:
        # normalizing by the latter would undo a fade through a phase inversion.
        power=((1-blend)*np.abs(self.last_harmonics[:,None])**2
               +blend*np.abs(harmonics[:,None])**2 if self.smooth_source
               else np.abs(harmonics[:,None])**2)
        expected_rms = np.sqrt(.5*np.sum(power*taper*taper,axis=0))
        periodic /= np.maximum(expected_rms,1e-8)
        noise = self.rng.normal(0,1,n)
        if self.noise_shaper is not None:
            noise=self.noise_shaper.process(noise,np.zeros(8) if noise_bands is None else noise_bands)
        excitation = gain*(np.sqrt(voicing)*periodic+np.sqrt(1-voicing)*noise)
        if self.filter_mode == 'normalized-lattice':
            output = self.lattice(excitation,reflection,self.history,
                                  previous_reflection=self.last_reflection if self.smooth_filter else None)
        else:
            initial = lfiltic([1.],coefficients,self.history[::-1])
            output,_ = lfilter([1.],coefficients,excitation,zi=initial)
        if not np.isfinite(output).all() or np.max(np.abs(output))>10:
            raise ValueError('Unstable or excessive generated voicebox output')
        self.phase = float(phase[-1]%(2*np.pi))
        self.last_f0 = float(f0[-1]); self.last_voicing = float(voicing[-1]); self.last_gain = float(gain[-1])
        if self.filter_mode == 'direct':
            self.history = np.r_[self.history,output][-self.config.order:]
        else:
            self.last_reflection=reflection.copy()
        self.last_harmonics=harmonics.copy()
        self.position += n
        return output
