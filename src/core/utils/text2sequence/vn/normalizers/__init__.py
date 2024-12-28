from core.utils.text2sequence.vn.normalizers.acronym_normalizer import AcronymNormalizer
from core.utils.text2sequence.vn.normalizers.breaker_normalizer import BreakNormalizer
from core.utils.text2sequence.vn.normalizers.character_normalizer import CharacterNormalizer
from core.utils.text2sequence.vn.normalizers.date_normalizer import DateNormalizer
from core.utils.text2sequence.vn.normalizers.letter_normalizer import LetterNormalizer
from core.utils.text2sequence.vn.normalizers.number_normalizer import NumberNomalizer
from core.utils.text2sequence.vn.normalizers.phoneme_normalizer import PhonemeNormalizer
from core.utils.text2sequence.vn.normalizers.symbol_normalizer import SymbolNormalizer
from core.utils.text2sequence.vn.normalizers.unit_normalizer import UnitNormalizer
from core.utils.text2sequence.vn.normalizers.tone_normalizer import ToneNormalizer

DEFAULT_PIPELINE = [
    DateNormalizer,
    NumberNomalizer,
    LetterNormalizer,
    AcronymNormalizer,
    SymbolNormalizer,
    UnitNormalizer,
    PhonemeNormalizer,
    ToneNormalizer,
    CharacterNormalizer,
    BreakNormalizer,
]



class TextNormalizer(object):

    def __init__(self, pipeline=DEFAULT_PIPELINE, lower=True):
        self.pipeline = pipeline
        self.lower = lower

    def normalize(self, text):
        if self.lower:
            text = text.lower()

        for processor in self.pipeline:
            text = processor.normalize(text)

        return text

    def __call__(self, text):
        return self.normalize(text)