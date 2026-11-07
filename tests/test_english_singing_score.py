import unittest
from english_singing_score import syllables,plan_english

WORDS={'little':['L','IH1','T','AH0','L'],'lights':['L','AY1','T','S'],
       'are':['AA1','R'],'shining':['SH','AY1','N','IH0','NG'],
       'quiet':['K','W','AY1','AH0','T']}
class EnglishScoreTests(unittest.TestCase):
 def test_syllables_keep_every_consonant_and_vowel_once(self):
  for phones in WORDS.values():
   parts=syllables(phones);self.assertEqual([p for part in parts for p in part],phones)
   self.assertTrue(all(sum(p[-1].isdigit() for p in part)==1 for part in parts))
  self.assertEqual(syllables(WORDS['shining']),[['SH','AY1'],['N','IH0','NG']])
 def test_new_lyrics_get_notes_per_syllable_and_exact_frame_boundaries(self):
  score=plan_english('Little lights are shining',WORDS.__getitem__)
  voiced=[n for n in score['notes'] if n['midi']]
  self.assertEqual(len(voiced),6)
  self.assertTrue(all(a['end']==b['start'] for a,b in zip(score['notes'],score['notes'][1:])))
  self.assertEqual(score['time'][1],sum(n['frames'] for n in score['notes'])*20)
 def test_tempo_and_voice_range_are_independent(self):
  high=plan_english('Little lights',WORDS.__getitem__,tempo=120)
  low=plan_english('Little lights',WORDS.__getitem__,tempo=120,transpose=-12)
  slow=plan_english('Little lights',WORDS.__getitem__,tempo=88)
  self.assertEqual(high['duration'],low['duration'])
  self.assertGreater(slow['time'][1],high['time'][1])
  self.assertTrue(all(a['midi']-b['midi']==12 for a,b in zip(high['notes'],low['notes']) if a['midi']))
 def test_backend_word_conditioning_continues_a_word_across_notes(self):
  score=plan_english('Little lights',WORDS.__getitem__)
  self.assertEqual(score['phoneme'].split()[1:3],['en_L-IH1-T-AH0-L','en_L-IH1-T-AH0-L'])
  self.assertEqual(score['note_type'].split()[1:4],['2','3','2'])
  self.assertEqual([p for n in score['notes'][1:3] for p in n['phonemes']],WORDS['little'])
 def test_unsupported_text_is_not_silently_dropped(self):
  for text in ['','123 stars','hello 🌟','a\nb\nc\nd\ne']:
   with self.assertRaises(ValueError):plan_english(text,WORDS.__getitem__)

if __name__=='__main__':unittest.main()
