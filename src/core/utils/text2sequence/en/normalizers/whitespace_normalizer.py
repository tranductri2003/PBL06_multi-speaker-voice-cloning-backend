import re


class WhiteSpaceNormalizer(object):

    _whitespace_re = re.compile(r"\s+")

    def collapse_whitespace(self, text):
        return re.sub(self._whitespace_re, " ", text)
    
    def __call__(self, text):
        return self.collapse_whitespace(text)
