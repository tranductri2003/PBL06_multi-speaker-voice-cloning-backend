import json
from core.settings import SAME_PHONEMES_FILEPATH
from core.utils.text2sequence.vn.normalizers.sort_by_key import sort_by_key
import re

with open(SAME_PHONEMES_FILEPATH, "r", encoding="utf-8") as file:
    SAME_PHONEMES = sort_by_key(json.load(file).items())



class PhonemeNormalizer(object):

    pattern = re.compile(r"(" + "|".join(map(re.escape, SAME_PHONEMES)) + r")")

    @classmethod
    def normalize(cls, text: str):

        def replace_symbol(match):
            return SAME_PHONEMES[match.group(0)]

        return cls.pattern.sub(replace_symbol, text)
