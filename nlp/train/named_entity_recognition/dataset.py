from typing import Any, Dict

from torch.utils.data import Dataset


class NerDataset(Dataset):
    def __init__(self, inputs: Any):
        super().__init__()

        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index) -> Dict[str, Any]:
        item = self.inputs[index]

        return {**item.inputs, "labels": item.labels}
