from dataclasses import dataclass
from typing import List


@dataclass
class NerInputs:
    sentence: str
    labels: List[int]
