import os
import shutil
from typing import List, Tuple, Union

import torch
from kodali import Kodali, NerAiHubLabels, NerOutputs, NerTags
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from nlp.train.named_entity_recognition.data_model import NerTensor
from nlp.utils.io import get_project_path, save_obj


def split_dataset(dataset: NerOutputs, ratio: float) -> Tuple[NerOutputs, NerOutputs]:
    train_dataset_size = int(dataset.size * ratio)

    return (
        NerOutputs(data=dataset.data[:train_dataset_size]),
        NerOutputs(data=dataset.data[train_dataset_size:]),
    )


class Preprocessor:
    def __init__(self) -> None:
        self.cache_path = os.path.join(get_project_path(), "ner", "dataset")
        self.train_dataset_cache_path = os.path.join(self.cache_path, "ner.train")
        self.valid_dataset_cache_path = os.path.join(self.cache_path, "ner.valid")

        os.makedirs(self.cache_path, exist_ok=True)

    def load_ai_hub(self, path: str, ratio: float = 0.9) -> Tuple[NerOutputs, NerOutputs]:
        return split_dataset(
            dataset=Kodali(path=path, task="ner", source="ai-hub"),  # type: ignore
            ratio=ratio,
        )

    def load_klue(self, path: str) -> Tuple[NerOutputs, NerOutputs]:
        train_path, valid_path = "", ""
        for file_name in os.listdir(path):
            if file_name.endswith("train.tsv"):
                train_path = os.path.join(path, file_name)
            elif file_name.endswith("dev.tsv"):
                valid_path = os.path.join(path, file_name)

        if train_path == "" and valid_path == "":
            raise ValueError("Path of training and validation dataset is not corrected.")

        return (
            Kodali(path=train_path, task="ner", source="klue"),
            Kodali(path=valid_path, task="ner", source="klue"),
        )  # type: ignore

    def is_existed_cache(self) -> bool:
        if os.path.isfile(self.train_dataset_cache_path) and os.path.isfile(self.valid_dataset_cache_path):
            return True

        return False

    def reload(self):
        if os.path.isdir(self.cache_path):
            shutil.rmtree(self.cache_path)

        os.makedirs(self.cache_path, exist_ok=True)

    def build(
        self,
        train_dataset: NerOutputs,
        valid_dataset: NerOutputs,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_seq_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        train_tensors = self.to_tensor(dataset=train_dataset, tokenizer=tokenizer, max_seq_length=max_seq_length)
        valid_tensors = self.to_tensor(dataset=valid_dataset, tokenizer=tokenizer, max_seq_length=max_seq_length)

        save_obj(path=self.train_dataset_cache_path, data=train_tensors)
        save_obj(path=self.valid_dataset_cache_path, data=valid_tensors)

    @staticmethod
    def to_tensor(
        dataset: NerOutputs,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_seq_length: int = 256,
    ) -> List[NerTensor]:
        tensors = []

        for data in tqdm(dataset.data, desc="encode"):
            inputs = tokenizer(
                data.sentence,
                max_length=max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            inputs = {key: value.squeeze(0) for key, value in inputs.items()}

            # TODO : Tokenizer is not based character
            if len(data.labels) < max_seq_length:
                labels = data.labels + [NerAiHubLabels.PAD] * (max_seq_length - len(data.labels))
            else:
                labels = data.labels[:max_seq_length]

                if labels[-1] != str(NerTags.OUTSIDE):
                    labels[-1] = str(NerTags.OUTSIDE)

                    if labels[-2] == str(NerTags.INSIDE):
                        labels[-2] = str(NerTags.END)

            labels = [NerAiHubLabels.LABEL2IDX[l] for l in labels]

            tensors.append(
                NerTensor(
                    inputs=inputs,
                    labels=torch.LongTensor(labels),
                )
            )

        return tensors
