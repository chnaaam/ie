import fire
from optimum.onnxruntime import ORTModelForTokenClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig


class Optimizers:
    def ner(self, model_path: str, output_path: str) -> None:
        ort_model = ORTModelForTokenClassification.from_pretrained(model_path, from_transformers=True)
        optimizer = ORTOptimizer.from_pretrained(ort_model)  # type: ignore
        optimization_config = OptimizationConfig(optimization_level=99, fp16=True)

        optimizer.optimize(save_dir=output_path, optimization_config=optimization_config)


if __name__ == "__main__":
    fire.Fire(Optimizers)
