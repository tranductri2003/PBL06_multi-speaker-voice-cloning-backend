import json
from core.settings import DATE_PREFIXES_FILEPATH
import re


with open(DATE_PREFIXES_FILEPATH, "r", encoding="utf-8") as file:
    DATE_PREFIXS = sorted(json.load(file), key=len, reverse=True)




class DateNormalizer(object):

    DATE_PREFIXS = DATE_PREFIXS

    date_pattern1 = re.compile(r"(\b\w{0,4}\b)\s*([12][0-9]|3[01]|0?[1-9])\/(1[0-2]|0?[1-9])\/(\d{1,4})")
    date_pattern2 = re.compile(r"(\b\w{0,4}\b)\s*([12][0-9]|3[01]|0?[1-9])\-(1[0-2]|0?[1-9])\-(\d{1,4})")
    date_pattern3 = re.compile(r"(\b\w{0,5}\b)\s*(0?[1-9]|1[0,1,2])[\/|\-](\d{4})")
    prefixs = "|".join(DATE_PREFIXS)
    date_pattern4 = re.compile(r"(" + prefixs + r")\s([12][0-9]|3[01]|0?[1-9])[\-|\/](1[0-2]|0?[1-9])")

    @classmethod
    def normalize_date_pattern1(cls, text: str):
        # Date pattern 1
        # Example: 11/12/2002

        def replace(match):
            prefix = match.group(1).strip()
            day = match.group(2)
            month = match.group(3)
            year = match.group(4)

            if prefix == "ngày":
                return f"{prefix} {day} tháng {month} năm {year}"
            else:
                return f'{prefix + " " if prefix != "" else ""}ngày {day} tháng {month} năm {year}'

        return cls.date_pattern1.sub(replace, text)

    @classmethod
    def normalize_date_pattern2(cls, text: str):
        # Date pattern 1
        # Example: 11-12-2002

        def replace(match):
            prefix = match.group(1).strip()
            day = match.group(2)
            month = match.group(3)
            year = match.group(4)

            if prefix == "ngày":
                return f"{prefix} {day} tháng {month} năm {year}"
            else:
                return f'{prefix + " " if prefix != "" else ""}ngày {day} tháng {month} năm {year}'

        return cls.date_pattern2.sub(replace, text)

    @classmethod
    def normalize_date_pattern3(cls, text: str):
        # Date pattern 3
        # Example: 12/2022 -> tháng 12 năm 2002

        def replace(match):
            prefix = match.group(1)
            month = match.group(2)
            year = match.group(3)

            if prefix == "tháng":
                return f"tháng {month} năm {year}"
            else:
                return f'{prefix + " " if prefix != "" else ""}tháng {month} năm {year}'

        return cls.date_pattern3.sub(replace, text)

    @classmethod
    def normalize_date_pattern4(cls, text: str):
        # Date pattern 4
        # Example: ngày 11/12

        def replace(match):
            prefix = match.group(1)
            day = match.group(2)
            month = match.group(3)

            if prefix == "ngày":
                return f"ngày {day} tháng {month}"
            else:
                return f'{prefix + " " if prefix != "" else ""}ngày {day} tháng {month}'

        return cls.date_pattern4.sub(replace, text)

    @classmethod
    def normalize(cls, text: str):
        text = cls.normalize_date_pattern4(text)
        text = cls.normalize_date_pattern1(text)
        text = cls.normalize_date_pattern2(text)
        text = cls.normalize_date_pattern3(text)
        return text
