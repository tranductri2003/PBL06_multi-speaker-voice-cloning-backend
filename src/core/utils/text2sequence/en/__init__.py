import re
from unidecode import unidecode
from core.utils.text2sequence.en.normalizers import ENAcronymNormalizer, ENNumberNormalizer, WhiteSpaceNormalizer

class EnglishText2Sequence(object):
    
    _pad        = "_"
    _eos        = "~"
    _characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? "

    symbols = [_pad, _eos] + list(_characters)

    _symbol_to_id = {s: i for i, s in enumerate(symbols)}
    _id_to_symbol = {i: s for i, s in enumerate(symbols)}
    _curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

    _acronym_normalizer = ENAcronymNormalizer()
    _number_normalizer = ENNumberNormalizer()
    _whitespace_normalizer = WhiteSpaceNormalizer()

    def lowercase(self, text):
        return text.lower()


    def convert_to_ascii(self, text):
        return unidecode(text)


    def normalize(self, text):
        text = self.convert_to_ascii(text)
        text = self.lowercase(text)
        text = self._number_normalizer(text)
        text = self._acronym_normalizer(text)
        text = self._whitespace_normalizer(text)
        
        return text
    
    def _symbols_to_sequence(self, symbols):
        return [self._symbol_to_id[s] for s in symbols if s in self._symbol_to_id and s not in ("_", "~")]

    
    def text_to_sequence(self, text):
        sequence = []

        while len(text):
            m = self._curly_re.match(text)
            if not m:
                sequence += self._symbols_to_sequence(self.normalize(text))
                break
            
            sequence += self._symbols_to_sequence(self.normalize(m.group(1)))
            sequence += self._arpabet_to_sequence(m.group(2))
            text = m.group(3)

        sequence.append(self._symbol_to_id["~"])
        return sequence
    
    def __call__(self, text):
        return self.text_to_sequence(text)