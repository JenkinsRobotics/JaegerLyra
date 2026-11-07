"""Original English lyric-to-note planner for research model adapters.

Pronunciation is supplied by the caller. This module does not load a model or
learn a voice. It separates lyric articulation, melody, tempo and voice range.
"""
import math
import re

MELODIES={'glow':(60,62,64,67,65,64,62,60),'lullaby':(64,62,60,62,64,62,59,60)}
ONSETS={tuple(x.split()) for x in ('P R','B R','T R','D R','K R','G R','F R','TH R','SH R','P L','B L','K L','G L','F L','S L','S M','S N','S W','T W','K W','S P','S T','S K','S P L','S P R','S T R','S K R','S K W')}

def syllables(phones):
    vowels=[i for i,p in enumerate(phones) if re.fullmatch(r'[A-Z]+[012]',p)]
    if not vowels:raise ValueError('No vowel in pronunciation')
    boundaries=[0]
    for left,right in zip(vowels,vowels[1:]):
        between=phones[left+1:right]
        onset=min(1,len(between))
        for length in range(2,len(between)+1):
            if tuple(between[-length:]) in ONSETS:onset=length
        boundaries.append(right-onset)
    boundaries.append(len(phones))
    return [phones[a:b] for a,b in zip(boundaries,boundaries[1:])]

def plan_english(lyrics,pronounce,tempo=104,melody='glow',transpose=0):
    if not isinstance(lyrics,str) or not lyrics.strip() or len(lyrics)>180:raise ValueError('Use up to 180 characters of English lyrics.')
    lines=[line.strip() for line in lyrics.splitlines() if line.strip()]
    if len(lines)>4 or any(len(line)>90 for line in lines):raise ValueError('Use one to four short lyric lines.')
    if not math.isfinite(tempo) or not 72<=tempo<=132:raise ValueError('Tempo must be 72–132 BPM.')
    if melody not in MELODIES or not isinstance(transpose,int) or not -12<=transpose<=12:raise ValueError('Unsupported melody or voice range.')
    if re.search(r"[^A-Za-z\s.,!?;:'’\-()\"]",lyrics):raise ValueError('Use English words; write numbers out in words.')
    notes=[];frame=0
    def append(word,phones,frames,midi,kind=2,word_phones=None):
        nonlocal frame
        notes.append({'word':word,'phonemes':phones,'wordPhonemes':word_phones or phones,'frames':frames,'start':frame/50,'end':(frame+frames)/50,'midi':midi,'type':kind});frame+=frames
    append('<SP>',[],10,0,1)
    for line_index,line in enumerate(lines):
        units=[]
        for word in re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*",line):
            phones=[p for p in pronounce(word.replace('’',"'").lower()) if re.fullmatch(r'[A-Z]+[012]?',p)]
            if not phones:raise ValueError('Could not pronounce '+word)
            units.extend((word,part,phones,index) for index,part in enumerate(syllables(phones)))
        if not units:raise ValueError('Each lyric line needs words.')
        if len(units)>16:raise ValueError('Keep each line to sixteen sung syllables or fewer.')
        for index,(word,phones,word_phones,syllable_index) in enumerate(units):
            final=index==len(units)-1;beats=1.7 if final else (1. if index%2==0 else .8)
            frames=max(8,round(50*60/tempo*beats))
            midi=(60 if line_index%2 else 64) if final else MELODIES[melody][index%len(MELODIES[melody])]
            append(word,phones,frames,midi+transpose,3 if syllable_index else 2,word_phones)
        append('<SP>',[],13,0,1)
    if frame>1500:raise ValueError('Keep the performance under thirty seconds.')
    return {'index':0,'language':'English','time':[0,frame*20],
            'text':' '.join(n['word'] for n in notes),
            # SoulX conditions on complete English words, repeated across
            # continuation notes. Emitting each syllable as a separate word
            # introduces false word boundaries (e.g. "to night"). Keep the
            # syllable articulation separately in the backend-neutral notes.
            'phoneme':' '.join('en_'+'-'.join(n['wordPhonemes']) if n['phonemes'] else '<SP>' for n in notes),
            'duration':' '.join(f"{n['frames']/50:.2f}" for n in notes),
            'note_pitch':' '.join(str(n['midi']) for n in notes),
            'note_type':' '.join(str(n['type']) for n in notes),
            'lyrics':lyrics,'notes':notes,'tempo':tempo,'melody':melody,'formatVersion':2}
