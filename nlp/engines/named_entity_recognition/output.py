from dataclasses import dataclass


@dataclass
class NerOutput:
    word: str
    label: str
    start_idx: int
    end_idx: int
