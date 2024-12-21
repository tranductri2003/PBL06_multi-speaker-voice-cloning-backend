import json

from core.utils.tokenizer.normalizers.sort_by_key import sort_by_key
from core.settings import LETTERS_FILEPATH
import re

with open(LETTERS_FILEPATH, "r", encoding="utf-8") as file:
    LETTERS = sort_by_key(json.load(file).items())



class LetterNormalizer(object):

    pattern = re.compile(r"\b(" + "|".join(map(re.escape, LETTERS)) + r")\b")

    @classmethod
    def normalize(cls, text: str):

        def replace_unit(match):
            return LETTERS[match.group(0)]

        return cls.pattern.sub(replace_unit, text)
