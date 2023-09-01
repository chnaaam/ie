from typing import List

from nlp.engines.named_entity_recognition import (
    NerInferenceEngine,
    NerInferenceOnnxEngine,
    NerOutput,
)


def test_inference_engine():
    engine = NerInferenceEngine()
    outputs = engine.run("홍길동의 아버지는 홍판서이다.")

    assert isinstance(outputs, List)
    assert isinstance(outputs[0], NerOutput)

    for output in outputs:
        print(output)


def test_inference_onnx_engine():
    engine = NerInferenceOnnxEngine()
    outputs = engine.run("홍길동의 아버지는 홍판서이다.")

    assert isinstance(outputs, List)
    assert isinstance(outputs[0], NerOutput)

    for output in outputs:
        print(output)
