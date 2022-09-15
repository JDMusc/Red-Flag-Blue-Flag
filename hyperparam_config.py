from dataclasses import dataclass


@dataclass
class ModelParams:
    embed_size = 50

@dataclass
class CnnModelParams(ModelParams):
    dropout:float = 0
    do_batch_norm:bool = False
    n_filters: int = 36
    kernel_sizes: list = lambda: [1, 2, 3, 5]
    bias: bool = True
