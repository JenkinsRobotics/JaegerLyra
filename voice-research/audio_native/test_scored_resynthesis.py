import unittest
import numpy as np
from neural_waveform import RATE
from pcm_reference_decoder import pcm_performance,PcmFrameDecoder
from scored_resynthesis import render_score


class ScoredResynthesisTests(unittest.TestCase):
    def setUp(self):
        t=np.arange(RATE)/RATE
        self.source=.2*np.sin(2*np.pi*200*t)
        self.performance=pcm_performance(self.source,np.arange(120,RATE-120,120,dtype=float))
        self.decoder=PcmFrameDecoder()

    def render(self,**changes):
        values=dict(source_s=[0.,1.],output_s=[0.,1.5],pitch_time_s=[0.,1.5],pitch_hz=[300.,300.],vibrato_cents=0.)
        values.update(changes)
        return render_score(self.performance,self.decoder,**values)[0]

    def test_authored_pitch_and_duration(self):
        result=self.render();self.assertEqual(len(result),round(RATE*1.5))
        middle=result[RATE//3:RATE]
        bins=np.fft.rfftfreq(len(middle),1/RATE)
        peak=bins[np.argmax(abs(np.fft.rfft(middle*np.hanning(len(middle)))))]
        self.assertLess(abs(peak-300),3.)
        self.assertTrue(np.isfinite(result).all());self.assertLess(max(abs(result)),.25)

    def test_nonuniform_time_map(self):
        result=self.render(source_s=[0.,.25,.75,1.],output_s=[0.,.15,1.35,1.5])
        self.assertEqual(len(result),36000)
        self.assertGreater(float(np.sqrt(np.mean(result[8000:28000]**2))),.08)

    def test_timing_must_be_monotonic(self):
        with self.assertRaises(ValueError):self.render(output_s=[0.,-.5])

    def test_source_map_must_cover_capture(self):
        with self.assertRaises(ValueError):self.render(source_s=[0.,.9])

    def test_out_of_range_pitch_rejected(self):
        with self.assertRaises(ValueError):self.render(pitch_hz=[0.,900.])

    def test_vibrato_changes_audio(self):
        plain=self.render();vibrato=self.render(vibrato_cents=25.)
        self.assertEqual(len(plain),len(vibrato))
        self.assertGreater(float(np.sqrt(np.mean((plain-vibrato)**2))),.005)


if __name__=='__main__':unittest.main()
