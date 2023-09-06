from typing import List

from luie.modules.ner_module import NerInferenceEngineModule, NerOutput


class LuieEngine:
    def __init__(self, task: str, device_id: int = 0, use_onnx: bool = False) -> None:
        if task == "ner":
            self.module = NerInferenceEngineModule()
        else:
            raise ValueError("Unsupported task. 'ner' task is currently supported.")

    def run(self, sentence: str) -> List[NerOutput]:
        return self.module.run(sentence=sentence)
