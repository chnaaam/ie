from typing import Tuple

import fire
from kodali import NerOutputs
from transformers import AutoTokenizer

from nlp.train.named_entity_recognition.preprocessor import Preprocessor
from nlp.train.named_entity_recognition.trainer import Trainer


def fix_seed(seed):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)  # type: ignore

    torch.backends.cudnn.deterministic = False  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    np.random.seed(seed)
    random.seed(seed)


class NamedEntityRecognition:
    def preprocess(
        self,
        tokenizer_name: str,
        max_seq_length: int = 256,
        ai_hub_dataset_path: str = "",
        klue_dataset_path: str = "",
        ratio: float = 0.9,
        reload: bool = False,
        seed: int = 42,
    ):
        fix_seed(seed=seed)

        p = Preprocessor()

        if reload:
            p.reload()

        if not p.is_existed_cache():
            train_dataset, valid_dataset = NerOutputs(data=[]), NerOutputs(data=[])

            if ai_hub_dataset_path:
                td, vd = p.load_ai_hub(path=ai_hub_dataset_path, ratio=ratio)

                train_dataset += td
                valid_dataset += vd

            if klue_dataset_path:
                td, vd = p.load_klue(path=klue_dataset_path)

                train_dataset += td
                valid_dataset += vd

            p.build(
                train_dataset=train_dataset,  # type: ignore
                valid_dataset=valid_dataset,  # type: ignore
                tokenizer=AutoTokenizer.from_pretrained(tokenizer_name),
                max_seq_length=max_seq_length,
            )

    def train(
        self,
        model_name: str,
        train_batch_size: int,
        valid_batch_size: int,
        num_epochs: int,
        learning_rate: float,
        num_train_workers: int = 0,
        num_valid_workers: int = 0,
        device_id: int = 0,
        use_fp16: bool = True,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        seed: int = 42,
    ):
        fix_seed(seed=seed)

        trainer = Trainer(
            model_name=model_name,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            num_train_workers=num_train_workers,
            num_valid_workers=num_valid_workers,
            device_id=device_id,
            use_fp16=use_fp16,
            betas=betas,
            weight_decay=weight_decay,
        )

        trainer.run()


if __name__ == "__main__":
    fire.Fire(NamedEntityRecognition)
