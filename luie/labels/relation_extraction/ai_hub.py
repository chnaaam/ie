# AI Hub
from typing import Dict


class ReAiHubLabels:
    SIZE: int = 20
    LABEL2IDX: Dict[str, int] = {
        "isParentOf": 0,
        "isUsedFor": 1,
        "isOpponentOf": 2,
        "no-Relation": 3,
        "isSiblingOf": 4,
        "isPlaceOf": 5,
        "isChildOf": 6,
        "isMemberOf": 7,
        "isSpouseOf": 8,
        "isToBe": 9,
        "hasAttribute": 10,
        "inFavorOf ": 11,
        "isAgainst": 12,
        "isA": 13,
        "isColleagueOf": 14,
        "isPartOf": 15,
        "isCausedBy": 16,
        "isTimeOf": 17,
        "isRelativeOf": 18,
        "inFavorOf": 19,
    }
    IDX2LABEL: Dict[str, str] = {
        "0": "isParentOf",
        "1": "isUsedFor",
        "2": "isOpponentOf",
        "3": "no-Relation",
        "4": "isSiblingOf",
        "5": "isPlaceOf",
        "6": "isChildOf",
        "7": "isMemberOf",
        "8": "isSpouseOf",
        "9": "isToBe",
        "10": "hasAttribute",
        "11": "inFavorOf ",
        "12": "isAgainst",
        "13": "isA",
        "14": "isColleagueOf",
        "15": "isPartOf",
        "16": "isCausedBy",
        "17": "isTimeOf",
        "18": "isRelativeOf",
        "19": "inFavorOf",
    }
