import os
import shutil
from typing import List, Tuple, Union

import torch
from kodali import Kodali, NerOutputs, to_bioes_scheme

from nlp.labels.named_entity_recognition.korean_corpus import NerKoreanCorpusLabels
from nlp.train.named_entity_recognition.data_model import NerInputs
from nlp.utils.io import get_project_path, save_obj


def split_dataset(dataset: NerOutputs, ratio: float) -> Tuple[NerOutputs, NerOutputs]:
    train_dataset_size = int(dataset.size * ratio)

    return (
        NerOutputs(data=dataset.data[:train_dataset_size]),
        NerOutputs(data=dataset.data[train_dataset_size:]),
    )


def to_labels(label: str) -> str:
    if label == NerKoreanCorpusLabels.PAD:
        return label

    # B-OGG_POLITICS -> B-OG
    return label[:4]


class Preprocessor:
    def __init__(self) -> None:
        self.cache_path = os.path.join(get_project_path(), "ner", "dataset")
        self.train_dataset_cache_path = os.path.join(self.cache_path, "ner.train")
        self.valid_dataset_cache_path = os.path.join(self.cache_path, "ner.valid")

        os.makedirs(self.cache_path, exist_ok=True)

    def load_korean_corpus(self, path: str, ratio: float = 0.9) -> Tuple[NerOutputs, NerOutputs]:
        return split_dataset(
            dataset=Kodali(path=path, task="ner", source="korean-corpus"),  # type: ignore
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
        max_seq_length: int = 256,
    ) -> None:
        self.max_seq_length = max_seq_length

        train_tensors = self.to_inputs(dataset=train_dataset)
        valid_tensors = self.to_inputs(dataset=valid_dataset)

        save_obj(path=self.train_dataset_cache_path, data=train_tensors)
        save_obj(path=self.valid_dataset_cache_path, data=valid_tensors)

    def to_inputs(self, dataset: NerOutputs) -> List[NerInputs]:
        inputs = []

        for data in dataset.data:
            sentence = data.sentence
            labels = to_bioes_scheme(data=data)

            inputs.append(
                NerInputs(
                    sentence=sentence,
                    labels=[NerKoreanCorpusLabels.LABEL2IDX[to_labels(label=l)] for l in labels],
                )
            )

        return inputs
