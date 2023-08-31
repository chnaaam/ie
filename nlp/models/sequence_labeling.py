from transformers import AutoModelForTokenClassification, PreTrainedModel


class SequenceLabelingModel:
    def __new__(cls, model_name: str, num_labels: int = -1) -> PreTrainedModel:
        params = {"pretrained_model_name_or_path": model_name, "num_labels": num_labels}

        if num_labels == -1:
            del params["num_labels"]

        return AutoModelForTokenClassification.from_pretrained(**params)
