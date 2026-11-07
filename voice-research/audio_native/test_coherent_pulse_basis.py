import unittest
import numpy as np
from coherent_pulse_basis import canonical_pulse,CoherentPulseBasisStream
from pulse_basis_instrument import OFFSETS,pulse_window,RADIUS
from compact_vocal_instrument import RATE,VocalFrame
from test_pulse_basis_instrument import analytic_model


class CoherentPulseTests(unittest.TestCase):
    def test_phase_canonicalization_of_shifted_analytic_waveforms(self):
        period=120.;window=pulse_window(period)
        def source(phase):
            x=2*np.pi*OFFSETS/period+phase
            return (np.cos(x)+.25*np.cos(3*x+.4))*window
        target=source(0.)
        for phase in (-2.4,-1.1,0.,.9,2.8):
            actual=canonical_pulse(source(phase),period)
            # Linear interpolation is explicitly part of the renderer.
            self.assertLess(np.sqrt(np.mean((actual-target)**2)),.001)
        self.assertGreater(np.linalg.norm(canonical_pulse(source(1.1),period,align=False)-target),1.)

    def test_silence_and_invalid_values(self):
        np.testing.assert_array_equal(canonical_pulse(np.zeros_like(OFFSETS),120.),0.)
        for period in (0.,float('nan'),401.):
            with self.assertRaises(ValueError):canonical_pulse(OFFSETS,period)
        with self.assertRaises(ValueError):canonical_pulse(np.full_like(OFFSETS,np.inf),120.)

    def test_stream_retains_tone_gain_seed_and_delay(self):
        model=analytic_model()
        stream=CoherentPulseBasisStream([model],amount=0.)
        blocks=[stream.push(VocalFrame(200.,(1.,))) for _ in range(100)]
        output=np.r_[np.concatenate(blocks),stream.finish()]
        self.assertEqual(len(output),RATE//2+RADIUS)
        t=np.arange(RATE//2)/RATE
        target=.04*np.sqrt(2)*np.cos(2*np.pi*200*t)
        self.assertLess(np.sqrt(np.mean((output[RADIUS:][1200:-1200]-target[1200:-1200])**2)),1e-5)
        a=CoherentPulseBasisStream([model],seed=4)
        b=CoherentPulseBasisStream([model],seed=4)
        for _ in range(20):
            np.testing.assert_allclose(a.push(VocalFrame(200.,(1.,)))*.5,
                b.push(VocalFrame(200.,(1.,),gain=.5)),atol=1e-14)
        a.cancel();self.assertFalse(np.any(a.pending));self.assertFalse(np.any(a.state))
        with self.assertRaises(RuntimeError):a.push(VocalFrame(200.,(1.,)))


if __name__=='__main__':unittest.main()
