import json 
import re

from core.settings import VOWELS_FILEPATH,HEAD_CONSONANTS_FILEPATH, FINAL_CONSONANTS_FILEPATH
from core.utils.text2sequence.vn.normalizers import TextNormalizer


ACCENTS = ['1', '2', '3', '4', '5']
BREAKS = {
    ".": " chấm ",
    ",": " phẩy ",
}

with open(VOWELS_FILEPATH, "r", encoding="utf-8") as file:
    VOWELS = sorted(json.load(file), key=len, reverse=True)

with open(HEAD_CONSONANTS_FILEPATH, "r", encoding="utf-8") as file:
    HEAD_CONSONANTS = sorted(json.load(file), key=len, reverse=True)

with open(FINAL_CONSONANTS_FILEPATH, "r", encoding="utf-8") as file:
    FINAL_CONSONANTS = sorted(json.load(file), key=len, reverse=True)

PHONEMES = sorted(VOWELS + HEAD_CONSONANTS + FINAL_CONSONANTS + ACCENTS + list(BREAKS.keys()) + [" "], key=len, reverse=True)
PHONEME_TO_ID = {s: i for i, s in enumerate(PHONEMES)}


class VietnameseText2Sequence(object):

    def __init__(self, phonemes=PHONEMES, normalizer=TextNormalizer(), spliter=" "):
        self.phonemes = phonemes
        self.normalizer = normalizer
        self.spliter = spliter

    def _parse_head_constants(self, word):
        pattern = r'^(' + '|'.join(HEAD_CONSONANTS) + ')'
        match = re.match(pattern, word)
        head_consonant = None
        if match:
            head_consonant = r'\b' + match.group(1)
        return re.sub(pattern, '', word), head_consonant
    
    def _parse_vowels(self, word):
        pattern = r'^(' + '|'.join(VOWELS) + ')'
        match = re.match(pattern, word)
        vowel = None
        if match:
            vowel =  match.group(1)
        return re.sub(pattern, '', word), vowel

    def _parse_final_constants(self, word):
        pattern = r'^(' + '|'.join(FINAL_CONSONANTS) + ')'
        match = re.match(pattern, word)
        final_consonant = None
        if match:
            final_consonant =  match.group(1)
        return re.sub(pattern, '', word), final_consonant

    def word2vec(self, word:str):
        embedding_vector = []
        
        word, head_consonant = self._parse_head_constants(word)
        word, vowel = self._parse_vowels(word)
        word, final_consonant = self._parse_final_constants(word)

        if head_consonant is not None:
            embedding_vector.append(PHONEME_TO_ID[head_consonant])
            
        if vowel is not None:
            embedding_vector.append(PHONEME_TO_ID[vowel])
            
        if final_consonant is not None:
            embedding_vector.append(PHONEME_TO_ID[final_consonant])
        
        if len(word) > 0 and word[-1] in PHONEMES:
            accent_or_break = word[-1]
            embedding_vector.append(PHONEME_TO_ID[accent_or_break])

        return {
            "head_consonant": head_consonant,
            "final_consonant": final_consonant,
            "vowel": vowel,
            "emmbedding_vector": embedding_vector
        }

    def embedding(self, text):
        text = self.normalizer.normalize(text)
        words = text.split(self.spliter)
        sequence = []
        for word in words:
            sequence.extend(self.word2vec(word)["emmbedding_vector"])
            sequence.append(PHONEME_TO_ID[self.spliter])
        return sequence[:-1]

    def __call__(self, text):
        return self.embedding(text)

