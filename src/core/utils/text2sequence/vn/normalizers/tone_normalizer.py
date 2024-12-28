import json
import re

from core.settings import TONES_FILEPATH

with open(TONES_FILEPATH, "r", encoding="utf-8") as file:
    TONES = json.load(file)



class ToneNormalizer(object):
    pattern = re.compile(r"(\w*)([áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ])(\w*)")

    @classmethod
    def normalize(cls, text):

        def replace(match):
            accented = match.group(2)
            base, tone = TONES[accented]
            return f"{match.group(1)}{base}{match.group(3)}{tone}"

        text = cls.pattern.sub(replace, text)
        return text
