"""Reusable captured-vowel instrument around the frozen neural decoder.

A short, explicitly captured voiced patch supplies timbre. Authored pitch,
duration, vibrato and amplitude supply the new performance. This is a neural
sample-based instrument, not a learned articulatory model or general cloning.
No original PCM, transcript, full target recording or encoder is used by the
runtime. Encoded source controls and a duration-dependent pulse plan remain
resident; only the decoded waveform cache is bounded.
"""
from dataclasses import dataclass
import numpy as np
from neural_waveform import RATE, HOP
from neural_performance import EncodedPerformance, compile_performance, validate_pulses
from neural_performance_stream import NeuralWindowReader, PerformancePacket, Grain


def compile_vowel(audio, encoder):
    """Select a stable source-only region, retaining at most 200 ms of audio.

    Selection uses source period and energy variation, never output pitch,
    identity, reconstruction error or a listener score. Intended vowel identity
    comes from the fixture label; this function does not recognize phonemes.
    """
    audio = np.asarray(audio, dtype=float)
    performance = compile_performance(audio, encoder)
    pulses = performance.pulses
    candidates = []
    for group in np.split(pulses, np.flatnonzero(np.diff(pulses) > RATE * .020) + 1):
        if len(group) < 8:
            continue
        for i in range(1, len(group)-6):
            last = np.searchsorted(group, group[i] + RATE * .150, side='right') - 1
            last = min(last, len(group)-2)
            selected = group[i:last+1]
            if len(selected) < 6 or selected[-1] - selected[0] < RATE * .060:
                continue
            periods = np.diff(selected)
            if np.max(periods) > 1.25 * np.min(periods):
                continue
            energies = np.array([np.sqrt(np.mean(audio[int(a):int(b)]**2))
                for a, b in zip(selected[:-1], selected[1:])])
            if np.min(energies) < .003:
                continue
            score = float(np.std(np.log(periods)) + .15 * np.std(np.log(energies)))
            # Include complete neighbouring source periods for grain support.
            start = max(0, int(np.floor(group[i-1])))
            stop = min(len(audio), int(np.ceil(group[last+1])) + 1)
            if stop-start <= RATE*.200:
                candidates.append((score, start, stop, selected))
    if not candidates:
        raise ValueError('No stable 60–200 ms voiced patch; reference rejected')
    score, start, stop, selected = min(candidates, key=lambda x: (x[0], x[1]))
    matrix = encoder.encode(audio[start:stop])
    local = (selected-start).copy()
    for item in (matrix.values, matrix.gains, local):
        item.setflags(write=False)
    return EncodedPerformance(matrix, local), {
        'sourceStartSeconds': start/RATE, 'sourceEndSeconds': stop/RATE,
        'retainedSeconds': (stop-start)/RATE, 'pulseCount': len(local),
        'sourceMedianPitchHz': float(RATE/np.median(np.diff(selected))),
        'sourceOnlySelectionScore': score,
        'selectionCandidates': len(candidates),
        'capturedTimbre': True, 'recognizedVowel': False,
    }


@dataclass(frozen=True)
class VowelScore:
    seconds: float = 2.
    times: tuple = (0.,)
    hz: tuple = (220.,)
    gain: float = 1.
    vibrato_cents: float = 0.
    vibrato_hz: float = 5.
    fade_seconds: float = .035

    def validate(self):
        scalar = (self.seconds, self.gain, self.vibrato_cents, self.vibrato_hz, self.fade_seconds)
        t = np.asarray(self.times, dtype=float); f = np.asarray(self.hz, dtype=float)
        if (not np.isfinite(scalar).all() or not .02 <= self.seconds <= 30
                or not 0 <= self.gain <= 2 or not 0 <= self.vibrato_cents <= 100
                or not 0 <= self.vibrato_hz <= 12 or not .005 <= self.fade_seconds <= .5
                or t.ndim != 1 or not len(t) or f.shape != t.shape
                or not np.isfinite(t).all() or not np.isfinite(f).all()
                or t[0] != 0 or np.any(np.diff(t) <= 0) or t[-1] > self.seconds
                or np.any(f < 60) or np.any(f > 800)):
            raise ValueError('Invalid vowel score')

    def frequency(self, seconds):
        seconds = np.asarray(seconds)
        pitch = np.interp(seconds, self.times, np.log2(self.hz))
        return 2**(pitch + self.vibrato_cents/1200 * np.sin(2*np.pi*self.vibrato_hz*seconds))

    def envelope(self, samples, count):
        fade = min(self.fade_seconds*RATE, (count-1)/2)
        attack = np.clip(samples/fade, 0., 1.)
        release = np.clip((count-1-samples)/fade, 0., 1.)
        return self.gain * (.5-.5*np.cos(np.pi*attack)) * (.5-.5*np.cos(np.pi*release))


class NeuralVowelInstrument:
    """Generate new note durations from a <=200 ms captured vowel patch.

    Pulse timing is planned at a fixed 1 kHz control rate. Source pulse indices
    move back and forth to avoid jumping from the patch end to its start. This
    repeats captured microvariation; it does not create novel natural prosody.
    Reads are synchronous and at most 10 ms; cancellation affects future reads.
    """
    def __init__(self, patch, decoder, score=VowelScore(), cache_frames=16):
        score.validate()
        if not 0 < patch.matrix.sample_count <= RATE*.200:
            raise ValueError('Expected a short captured vowel patch')
        pulses = validate_pulses(patch.pulses, patch.matrix.sample_count).copy()
        if len(pulses) < 4 or np.min(np.diff(pulses)) < RATE/1000 or np.max(np.diff(pulses)) > RATE*.020:
            raise ValueError('Invalid periodic vowel patch')
        self.score = VowelScore(float(score.seconds), tuple(score.times), tuple(score.hz),
            float(score.gain), float(score.vibrato_cents), float(score.vibrato_hz), float(score.fade_seconds))
        self.reader = NeuralWindowReader(patch.matrix, decoder, cache_frames)
        self.count = round(score.seconds*RATE)
        # Metadata only. Extending the time axis beyond the requested endpoint
        # supplies the right neighbour needed by the final grain window.
        times = np.arange(int(np.ceil((score.seconds+.06)*1000))+1)/1000
        frequency = self.score.frequency(times)
        phase = np.r_[0., np.cumsum((frequency[:-1]+frequency[1:])*.0005)]
        centers = np.interp(np.arange(int(np.floor(phase[-1]))+1), phase, times)*RATE
        periods = np.diff(pulses)
        choices = np.r_[np.arange(1, len(pulses)-1), np.arange(len(pulses)-3, 1, -1)]
        self.grains = []
        for j, center in enumerate(centers[:-1]):
            index = choices[j % len(choices)]
            a = min(periods[index-1], center-centers[j-1] if j else centers[1]-center)
            b = min(periods[index], centers[j+1]-center)
            start = max(0, int(np.ceil(center-a))); stop = min(self.count, int(np.floor(center+b))+1)
            if stop > start:
                self.grains.append(Grain(float(center), float(pulses[index]), float(a), float(b), start, stop))
        self.order = sorted(range(len(self.grains)), key=lambda i: self.grains[i].start)
        self.next_grain = 0; self.active = []; self.position = 0; self.closed = False
        self.maximum_active_grains = 0

    def read(self, samples=HOP):
        if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or not 1 <= samples <= HOP:
            raise ValueError('Read 1–240 samples')
        if self.closed:
            return None
        start = self.position; stop = min(self.count, start+samples)
        self.active = [i for i in self.active if self.grains[i].stop > start]
        while self.next_grain < len(self.order) and self.grains[self.order[self.next_grain]].start < stop:
            self.active.append(self.order[self.next_grain]); self.next_grain += 1
        self.active.sort()
        self.maximum_active_grains = max(self.maximum_active_grains, len(self.active))
        audio = np.zeros(stop-start); weights = np.zeros(stop-start)
        for index in self.active:
            grain = self.grains[index]
            indices = np.arange(max(start, grain.start), min(stop, grain.stop))
            offsets = indices-grain.center
            widths = np.where(offsets < 0, grain.left, grain.right)
            window = .5+.5*np.cos(np.pi*offsets/widths)
            where = indices-start
            audio[where] += self.reader.sample(grain.source+offsets)*window
            weights[where] += window
        audio = np.divide(audio, weights, out=np.zeros_like(audio), where=weights > 1e-12)
        audio *= self.score.envelope(np.arange(start, stop), self.count)
        self.position = stop
        if stop == self.count:
            self.closed = True; self.reader.clear(); self.active.clear()
        return PerformancePacket(start, audio)

    def cancel(self):
        self.closed = True; self.reader.clear(); self.active.clear()

    def __iter__(self):
        while (packet := self.read()) is not None:
            yield packet
