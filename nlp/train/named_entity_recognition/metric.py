from typing import List

from seqeval.metrics import f1_score
from seqeval.scheme import IOBES


def calc_f1_score(
    pred_y_list: List[List[int]],
    true_y_list: List[List[int]],
    idx2label: dict,
    pad_id: int,
) -> float:
    true_labels, pred_labels = [], []

    for pred_y, true_y in zip(pred_y_list, true_y_list):
        tl, pl = [], []

        for py, ty in zip(pred_y, true_y):
            if ty == pad_id:
                break

            pl.append(idx2label[str(py)])
            tl.append(idx2label[str(ty)])

        true_labels.append(tl)
        pred_labels.append(pl)

    score = f1_score(true_labels, pred_labels, scheme=IOBES)

    if isinstance(score, list):
        return score[0] * 100
    else:
        return score * 100
