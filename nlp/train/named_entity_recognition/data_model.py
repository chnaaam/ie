from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class NerTensor:
    inputs: Dict[str, torch.Tensor]
    labels: torch.Tensor
