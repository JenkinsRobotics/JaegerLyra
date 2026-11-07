import unittest
import numpy as np
from kokoro_singing import plan_score, validate_request


class KokoroScoreTests(unittest.TestCase):
 def test_vowels_hold_notes_while_consonants_keep_their_articulation(self):
  phones='stˈeɪ wɪð mi'
  natural=np.full(len(phones)+2,3)
  durations,pitch,notes=plan_score(phones,natural,104)
  self.assertEqual(len(pitch),int(sum(durations))*2)
  self.assertTrue(np.isfinite(pitch).all())
  self.assertTrue(all(durations[n['nucleus']]>=6 for n in notes))
  self.assertEqual(durations[1],3) # /s/ is not stretched like a vowel.
  self.assertEqual(durations[0],3)
  self.assertEqual(durations[-1],5)
  self.assertTrue(all(a['end']==b['start'] for a,b in zip(notes,notes[1:])))
 def test_tempo_changes_timing_and_octave_changes_pitch_independently(self):
  phones='la la la';natural=np.full(len(phones)+2,3)
  slow,p1,_=plan_score(phones,natural,88)
  fast,_,_=plan_score(phones,natural,120)
  low,p2,_=plan_score(phones,natural,88,transpose=-12)
  self.assertGreater(sum(slow),sum(fast))
  np.testing.assert_array_equal(slow,low)
  np.testing.assert_allclose(p1/2,p2,rtol=1e-5)
 def test_reference_voice_and_unbounded_lyrics_are_not_silently_substituted(self):
  for request in [{'lyrics':''},{'lyrics':'a'*181},{'lyrics':'Hello','tempo':float('nan')},
                  {'lyrics':'Hello','speaker':'Serena'},{'lyrics':'Hello','melody':'unknown'}]:
   with self.assertRaises(ValueError):validate_request(request)
  request=validate_request({'lyrics':'[Verse]\nStay with me','speaker':'bf_emma'})
  self.assertEqual(request['speaker'],'bf_emma')
  self.assertEqual(request['lyrics'],'Stay with me')
  with self.assertRaises(ValueError):plan_score('shh',np.ones(5))
