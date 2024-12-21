import re

BREAKS = {
    ".": " chấm ",
    ",": " phẩy ",
}



class BreakNormalizer(object):

    BREAKS = BREAKS

    duplicate_dot_comma_pattern = re.compile(r"([,.]){2,}")
    adjacent_symbols_pattern = re.compile(r"(\S)([,.])(\S)")
    left_symbol_pattern = re.compile(r"(\S)([,.])")
    right_symbol_pattern = re.compile(r"([,.])(\S)")

    @classmethod
    def normalize(cls, text):
        text = cls.duplicate_dot_comma_pattern.sub(lambda m: m.group(1), text)

        def replace_dot_and_comma(match):
            return match.group(1) + cls.BREAKS[match.group(2)] + match.group(3)

        text = cls.adjacent_symbols_pattern.sub(replace_dot_and_comma, text)
        text = cls.left_symbol_pattern.sub(r"\1 \2", text)
        text = cls.right_symbol_pattern.sub(r"\1 \2", text)

        return text