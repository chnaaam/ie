from typing import List

from nlp.engines.engine import NlpEngine
from nlp.engines.named_entity_recognition.output import NerOutput


def test_inference_ner():
    engine = NlpEngine(task="ner")
    outputs = engine.run("홍길동의 아버지는 홍판서이다.")

    assert isinstance(outputs, List)
    assert isinstance(outputs[0], NerOutput)

    for output in outputs:
        print(output)
