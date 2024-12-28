import re


class ENAcronymNormalizer(object):
    
    _acronyms = [(re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1]) for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]]


    def normalize_acronym(self, text):
        for regex, replacement in self._acronyms:
            text = re.sub(regex, replacement, text)
            
        return text

    def __call__(self, text):
        return self.normalize_acronym(text)