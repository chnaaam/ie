from typing import List

from nlp.engines.named_entity_recognition.engine import (
    NerInferenceEngine,
    NerInferenceOnnxEngine,
)
from nlp.engines.named_entity_recognition.output import NerOutput


class NlpEngine:
    def __init__(self, task: str, device_id: int = 0, use_onnx: bool = False) -> None:
        if task == "ner":
            self.module = NerInferenceEngine() if not use_onnx else NerInferenceOnnxEngine()
        else:
            raise ValueError("Unsupported task. 'ner' task is currently supported.")

    def run(self, sentence: str) -> List[NerOutput]:
        return self.module.run(sentence=sentence)
