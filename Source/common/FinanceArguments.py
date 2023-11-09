from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelArgs:
    output_length: int
    num_layers: int
    input_size: Tuple[int, int]
    hidden_size: int
    fc_hidden_size: int
    device: str

@dataclass
class DataArgs:
    stock_id: str
    seq_length: int
    output_length: int
    predict_type: str

@dataclass
class TrainerArgs:
    batch_size: int
    learning_rate: float
    num_epoch: int
    do_train: bool
    do_test: bool
    device: str
