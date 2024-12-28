import json
import re
from core.settings import BASE_NUMBERS_FILEPATH, NUMBER_LEVELS_FILEPATH

with open(BASE_NUMBERS_FILEPATH, "r", encoding="utf-8") as file:
    BASE_NUMBERS = {int(key): value for key, value in json.load(file).items()}

with open(NUMBER_LEVELS_FILEPATH, "r", encoding="utf-8") as file:
    NUMBER_LEVELS = {int(key): value for key, value in json.load(file).items()}

class NumberNomalizer(object):

    pattern = re.compile(r"\d+")

    @classmethod
    def _convert_number_2_digits(cls, number: int):
        if number in BASE_NUMBERS:
            return BASE_NUMBERS[number]

        tens = number // 10
        base = number % 10
        if base > 0:
            return f"{BASE_NUMBERS[tens]} mươi {BASE_NUMBERS[base]}"

        return f"{BASE_NUMBERS[tens]} mươi"

    @classmethod
    def _convert_number_3_digits(cls, number: int):
        if number == 0:
            return ""

        remainder = number % 100
        hundred = number // 100
        if remainder == 0:
            return f"{BASE_NUMBERS[hundred]} trăm"

        if remainder < 10:
            return f"{BASE_NUMBERS[number // 100]} trăm linh {BASE_NUMBERS[remainder]}"

        return f"{BASE_NUMBERS[hundred]} trăm {cls._convert_number_2_digits(remainder)}"

    @classmethod
    def number_to_vietnamese(cls, number: int):
        if number == 0:
            return "không"

        if number in BASE_NUMBERS:
            return BASE_NUMBERS[number]

        if number < 100:
            return cls._convert_number_2_digits(number)

        result = cls._convert_number_3_digits(number % 1000)
        current_level = None

        for current_level in NUMBER_LEVELS:
            next_level = current_level * 1000
            if number // (next_level) == 0:
                break
            level_base = number % (next_level) // current_level
            result = f"{cls._convert_number_3_digits(level_base)} {NUMBER_LEVELS[current_level]} {result}"

        level_base = number // current_level

        if level_base == 0:
            return result

        if level_base in BASE_NUMBERS:
            return f"{BASE_NUMBERS[level_base]} {NUMBER_LEVELS[current_level]} {result}"

        if level_base > 99:
            return f"{cls._convert_number_3_digits(level_base)} {NUMBER_LEVELS[current_level]} {result}"

        if level_base > 11:
            return f"{cls._convert_number_2_digits(level_base)} {NUMBER_LEVELS[current_level]} {result}"

    @classmethod
    def normalize(cls, text: str) -> str:

        replaced_text = cls.pattern.sub(lambda x: cls.number_to_vietnamese(int(x.group())), text)

        return replaced_text
