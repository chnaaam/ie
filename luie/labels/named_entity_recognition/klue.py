# KLUE
# ['PS', 'LC', 'OG', 'DT', 'TI', 'QT']

from typing import Dict


class NerKlueLabels:
    SIZE: int = 26
    PAD = "[PAD]"
    LABEL2IDX: Dict[str, int] = {
        PAD: 0,
        "O": 1,
        "B-DT": 2,
        "I-DT": 3,
        "E-DT": 4,
        "S-DT": 5,
        "B-LC": 6,
        "I-LC": 7,
        "E-LC": 8,
        "S-LC": 9,
        "B-OG": 10,
        "I-OG": 11,
        "E-OG": 12,
        "S-OG": 13,
        "B-PS": 14,
        "I-PS": 15,
        "E-PS": 16,
        "S-PS": 17,
        "B-QT": 18,
        "I-QT": 19,
        "E-QT": 20,
        "S-QT": 21,
        "B-TI": 22,
        "I-TI": 23,
        "E-TI": 24,
        "S-TI": 25,
    }
    IDX2LABEL: Dict[str, str] = {
        "0": PAD,
        "1": "O",
        "2": "B-DT",
        "3": "I-DT",
        "4": "E-DT",
        "5": "S-DT",
        "6": "B-LC",
        "7": "I-LC",
        "8": "E-LC",
        "9": "S-LC",
        "10": "B-OG",
        "11": "I-OG",
        "12": "E-OG",
        "13": "S-OG",
        "14": "B-PS",
        "15": "I-PS",
        "16": "E-PS",
        "17": "S-PS",
        "18": "B-QT",
        "19": "I-QT",
        "20": "E-QT",
        "21": "S-QT",
        "22": "B-TI",
        "23": "I-TI",
        "24": "E-TI",
        "25": "S-TI",
    }
