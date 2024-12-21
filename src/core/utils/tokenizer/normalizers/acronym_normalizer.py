import json
from core.utils.tokenizer.normalizers.sort_by_key import sort_by_key
from core.settings import ACRONYMS_FILEPATH

import re

with open(ACRONYMS_FILEPATH, "r", encoding="utf-8") as file:
    ACRONYMS = sort_by_key(json.load(file).items())

class AcronymNormalizer(object):

    pattern = re.compile(r"\b(" + "|".join(map(re.escape, ACRONYMS)) + r")\b")

    @classmethod
    def normalize(cls, text: str):
        def replace_unit(match):
            return ACRONYMS[match.group(0)]

        return cls.pattern.sub(replace_unit, text)