import json
from core.utils.text2sequence.vn.normalizers.sort_by_key import sort_by_key
from core.settings import UNITS_FILEPATH
import re

with open(UNITS_FILEPATH, "r", encoding="utf-8") as file:
    UNITS = sort_by_key(json.load(file).items())



class UnitNormalizer(object):

    pattern = re.compile(r"\b(" + "|".join(map(re.escape, UNITS)) + r")\b")

    @classmethod
    def normalize(cls, text):

        def replace_unit(match):
            return UNITS[match.group(0)]

        return cls.pattern.sub(replace_unit, text)
