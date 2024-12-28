import re


class CharacterNormalizer(object):

    pattern = re.compile(r"[^a-zA-Z0-9\sđâăêôơư.,]")

    @classmethod
    def normalize(cls, text: str):

        return cls.pattern.sub("", text)
