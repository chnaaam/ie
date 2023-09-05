from typing import Any, Dict, Union

import torch
from kodali import NerTags
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from nlp.labels.named_entity_recognition.korean_corpus import NerKoreanCorpusLabels


def to_labels(label: str) -> str:
    if label == NerKoreanCorpusLabels.PAD:
        return label

    # B-OGG_POLITICS -> B-OG
    return label[:4]


class NerDataset(Dataset):
    def __init__(
        self,
        inputs: Any,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_seq_length: int = 256,
    ):
        super().__init__()

        self.inputs = inputs
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index) -> Dict[str, Any]:
        item = self.inputs[index]

        sentence = item.sentence
        labels = item.labels

        inputs = self.tokenizer(
            sentence,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        inputs = {key: value.squeeze(0) for key, value in inputs.items()}

        _max_seq_length = self.max_seq_length - 2  # CLS, SEP Token

        if len(labels) < _max_seq_length:
            labels = labels + [NerKoreanCorpusLabels.LABEL2IDX[NerKoreanCorpusLabels.PAD]] * (
                _max_seq_length - len(labels)
            )
        else:
            labels = labels[:_max_seq_length]

            if labels[-1] != NerKoreanCorpusLabels.LABEL2IDX[str(NerTags.OUTSIDE)]:
                labels[-1] = NerKoreanCorpusLabels.LABEL2IDX[str(NerTags.OUTSIDE)]

                if NerKoreanCorpusLabels.IDX2LABEL[str(labels[-2])].startswith(str(NerTags.INSIDE)):
                    replaced_label = NerKoreanCorpusLabels.IDX2LABEL[str(labels[-2])].replace(
                        str(NerTags.INSIDE),
                        str(NerTags.END),
                    )

                    labels[-2] = NerKoreanCorpusLabels.LABEL2IDX[replaced_label]

        labels = (
            [NerKoreanCorpusLabels.LABEL2IDX[str(NerTags.OUTSIDE)]]
            + labels
            + [NerKoreanCorpusLabels.LABEL2IDX[str(NerTags.OUTSIDE)]]
        )

        return {**inputs, "labels": torch.LongTensor(labels)}
