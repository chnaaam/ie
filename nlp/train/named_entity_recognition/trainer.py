import os
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from kodali import NerAiHubLabels, NerOutputs
from torch.utils.data import DataLoader
from tqdm import tqdm

from nlp.train.named_entity_recognition.dataset import NerDataset
from nlp.train.named_entity_recognition.metric import calc_f1_score
from nlp.train.named_entity_recognition.models import SequenceLabelingModel
from nlp.utils.io import get_project_path, load_obj


def to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {k: v.to(device) for k, v in batch.items()}


def load_dataset(train_dataset_path: str, valid_dataset_path: str) -> Tuple[NerOutputs, NerOutputs]:
    return (
        load_obj(path=train_dataset_path),
        load_obj(path=valid_dataset_path),
    )


class Trainer:
    def __init__(
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
    ) -> None:
        self.cache_model_path = os.path.join(get_project_path(), "ner", "fine-tuning")

        os.makedirs(self.cache_model_path, exist_ok=True)

        self.cache_dataset_path = os.path.join(get_project_path(), "ner", "dataset")

        train_dataset, valid_dataset = load_dataset(
            train_dataset_path=os.path.join(self.cache_dataset_path, "ner.train"),
            valid_dataset_path=os.path.join(self.cache_dataset_path, "ner.valid"),
        )

        self.scaler = torch.cuda.amp.GradScaler() if use_fp16 else None  # type: ignore
        self.num_epochs = num_epochs
        self.device = f"cuda:{device_id}" if device_id != -1 else "cpu"

        self.train_data_loader = DataLoader(
            dataset=NerDataset(inputs=train_dataset),
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_train_workers,
        )
        self.valid_data_loader = DataLoader(
            dataset=NerDataset(inputs=valid_dataset),
            batch_size=valid_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_valid_workers,
        )

        self.model = SequenceLabelingModel(
            model_name=model_name,
            num_labels=NerAiHubLabels.SIZE,
        )

        self.model = self.model.to(self.device)  # type: ignore

        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )

    def run(self) -> None:
        for e in range(self.num_epochs):
            self.train(current_epoch=e)
            valid_score = self.valid(current_epoch=e)

            self.model.save_pretrained(os.path.join(self.cache_model_path, f"score-{valid_score:.2f}"))

    def train(self, current_epoch: int) -> None:
        losses = []
        avg_loss = 0.0

        self.model.train()

        train_progress = tqdm(self.train_data_loader)
        for batch in train_progress:
            train_progress.set_description(
                "Training [Epoch : {0}|{1}, Avg Loss : {2:.4f}]".format(
                    current_epoch,
                    self.num_epochs,
                    avg_loss,
                )
            )

            self.optimizer.zero_grad()
            batch = to_device(batch=batch, device=self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():  # type: ignore
                    outputs = self.model(**batch)
                    loss = outputs.loss

                self.scaler.scale(loss).backward()  # type: ignore
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**batch)

                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

            losses.append(loss.tolist())

            if len(losses) > 100:
                losses = losses[len(losses) - 100 :]

            avg_loss = sum(losses) / len(losses)

            # Logging
            # mlflow.log_metrics(
            #     {
            #         "training-loss": outputs.loss.tolist(),
            #     }
            # )

    def valid(self, current_epoch: int) -> float:
        losses, scores = [], []
        avg_loss, avg_score = 0.0, 0.0
        self.model.eval()

        with torch.no_grad():
            valid_progress = tqdm(self.valid_data_loader)
            for batch in valid_progress:
                valid_progress.set_description(
                    "Validation [Epoch : {0}|{1}, Avg Loss : {2:.4f}, Avg Score : {3:.2f}]".format(
                        current_epoch,
                        self.num_epochs,
                        avg_loss,
                        avg_score,
                    )
                )

                batch = to_device(batch=batch, device=self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():  # type: ignore
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)

                loss = outputs.loss
                pred_y_list = torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1).tolist()
                true_y_list = batch["labels"].tolist()

                score = calc_f1_score(
                    pred_y_list=pred_y_list,
                    true_y_list=true_y_list,
                    idx2label=NerAiHubLabels.IDX2LABEL,
                    pad_id=NerAiHubLabels.LABEL2IDX[NerAiHubLabels.PAD],
                )

                losses.append(loss.tolist())
                scores.append(score)

                if len(losses) > 100:
                    losses = losses[len(losses) - 100 :]

                avg_loss = sum(losses) / len(losses)
                avg_score = sum(scores) / len(scores)

                # Logging
                # mlflow.log_metrics(
                #     {
                #         "validation-loss": outputs.loss.tolist(),
                #         "validation-score": score,
                #     }
                # )

        return avg_score
