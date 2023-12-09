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
    seq_length: int
    embed_dim: int
    resolution: int
    n_head: int
    n_layer: int
    tf_fc_hidden_size: int
    tf_input_size: int

@dataclass
class EmbeddingArgs:
    embed_dim: int
    resolution: int

@dataclass
class DataArgs:
    seq_length: int
    output_length: int
    predict_type: str
    data_path: str

@dataclass
class TrainerArgs:
    batch_size: int
    learning_rate: float
    num_epoch: int
    do_train: bool
    do_test: bool
    device: str
    resolution: int
    stock_type: str
    save_model: bool
    model_type: str
    
@dataclass
class DataBuilderArgs:
    seq_length: int
    output_length: int
    predict_type: str
